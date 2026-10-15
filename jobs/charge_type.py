"""
Categorize unique charge labels into analytics.charge_mapping (Supabase/Postgres).

Local replacement for the Glue job in charge.py. Classification logic is
unchanged: rule_classify_row from src/taxonomy.py first, then a TF-IDF
char_wb 3-5 gram fallback with SIMILARITY_THRESHOLD = 0.35.

Precedence: client override seed (seeds/charge_mapping_overrides.csv, method 'client_review')
> written rules (REVIEWED_LABELS, PRIORITY_OVERRIDES, TAXONOMY) > TF-IDF similarity. Seed rows
are re-applied on every run, also to labels already in the mapping, so a one-line seed change
takes effect without --rebuild. Lines of excluded carriers / clients (factoring lookups) are
flagged is_excluded in analytics.charge_categorized and are not categorized or counted here.

Incremental: reads aggregated (charge_type, charge_description) combinations
from staging.charge and classifies only the ones not already in the mapping,
appending them. Existing rows keep their category; only row_count /
total_value / updated_at are refreshed. To re-classify a label after fixing a
pattern, delete its row from the mapping and re-run. Everything is written in
a single transaction.

--rebuild instead re-classifies every label and replaces the whole mapping
(TRUNCATE + insert), so all rows reflect the current taxonomy.

Requires SUPABASE_DB_URL (see .env.example).

Run from the repo root:
    python jobs/charge_type.py              # classify new labels only
    python jobs/charge_type.py --rebuild    # re-classify everything

To experiment without touching the shared analytics schema, write to your
own sandbox schema instead:
    python jobs/charge_type.py --schema sandbox_yourname
The first run creates the schema plus its own charge_mapping table and
charge_categorized / charge_review views (from the migration SQL, pointed at
that schema). staging.charge is only ever read.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import overrides  # noqa: E402
from taxonomy import CATCH_ALL, TAXONOMY, normalize, rule_classify_row  # noqa: E402
from tfidf_fallback import SIMILARITY_THRESHOLD, TfidfFallback  # noqa: E402

MIGRATIONS = sorted((REPO_ROOT / "supabase" / "migrations").glob("*.sql"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--schema", default="analytics",
                    help="schema to write charge_mapping into (default: analytics, the shared one)")
parser.add_argument("--rebuild", action="store_true",
                    help="re-classify every label and replace the whole mapping (default: only new labels)")
parser.add_argument("--seed", type=Path, default=REPO_ROOT / "seeds" / "charge_mapping_overrides.csv",
                    help="client override seed (default: seeds/charge_mapping_overrides.csv); "
                         "point it at a header-only CSV to see what the rules alone would produce")
parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD,
                    help=f"minimum TF-IDF similarity to accept a fallback match (default: {SIMILARITY_THRESHOLD})")
args = parser.parse_args()
SCHEMA = args.schema
# The schema name is interpolated into SQL, so only allow plain identifiers.
if not re.fullmatch(r"[a-z_][a-z0-9_]*", SCHEMA):
    sys.exit(f"--schema must be a lowercase identifier like sandbox_yourname, got {SCHEMA!r}")

load_dotenv(REPO_ROOT / ".env")
DB_URL = os.environ.get("SUPABASE_DB_URL")
if not DB_URL:
    sys.exit("SUPABASE_DB_URL is not set (copy .env.example to .env and fill it in)")
# Pin the psycopg2 driver (SQLAlchemy 2.1 defaults to psycopg3) and accept postgres:// URLs.
for prefix in ("postgres://", "postgresql://"):
    if DB_URL.startswith(prefix):
        DB_URL = "postgresql+psycopg2://" + DB_URL[len(prefix):]
        break

engine = create_engine(DB_URL)

if SCHEMA != "analytics":
    # Same DDL as the shared migration (all IF NOT EXISTS / OR REPLACE), just
    # retargeted at the sandbox schema. The shared schema is never re-created here.
    with engine.begin() as conn:
        # Replaying the migrations in order would make the first one's narrower views fail
        # ("cannot drop columns from view") on every run after the first, so drop the views
        # (never the table) and let the migrations re-create them.
        conn.exec_driver_sql(f"DROP VIEW IF EXISTS {SCHEMA}.charge_categorized, {SCHEMA}.charge_review")
        for migration in MIGRATIONS:
            conn.exec_driver_sql(re.sub(r"\banalytics\b", SCHEMA, migration.read_text()))
else:
    # The shared schema is migrated by hand; fail early if the override columns are not there yet.
    with engine.connect() as conn:
        cols = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'analytics' AND table_name = 'charge_mapping_review'"))}
    if not cols:
        sys.exit("analytics.charge_mapping_review does not exist: apply "
                 "supabase/migrations/20261015000000_client_review_overrides.sql first")

seed = overrides.load_seed(args.seed)
print(f"Override seed: {len(seed)} rows ({seed['status'].value_counts().to_dict()})")

# ---------------------------------------------------------------------------
# Load unique (charge_type, charge_description) combinations, aggregated in SQL
# ---------------------------------------------------------------------------
AGG_SQL = text("""
    SELECT coalesce(charge_type, '')        AS charge_type,
           coalesce(charge_description, '') AS charge_description,
           count(*)                         AS row_count,
           sum(charge_value)                AS total_value
    FROM staging.charge
    GROUP BY 1, 2
""")

with engine.connect() as conn:
    unique_labels = pd.read_sql(AGG_SQL, conn)

unique_labels = unique_labels.rename(columns={
    "charge_type": "Charge Type",
    "charge_description": "Charge Description",
})
unique_labels["row_count"] = unique_labels["row_count"].astype("int64")
unique_labels["total_value"] = pd.to_numeric(unique_labels["total_value"], errors="coerce")

# ---------------------------------------------------------------------------
# Incremental: keep only labels that are not in the mapping yet. all_labels keeps
# the full aggregate so counts on existing rows can be refreshed at the end.
# ---------------------------------------------------------------------------
all_labels = unique_labels
if args.rebuild:
    existing_keys = set()
    print(f"--rebuild: re-classifying all {len(all_labels):,} labels")
else:
    with engine.connect() as conn:
        existing = pd.read_sql(text(f"SELECT charge_type, charge_description FROM {SCHEMA}.charge_mapping"), conn)
    existing_keys = set(zip(existing["charge_type"], existing["charge_description"]))
    is_new = [k not in existing_keys for k in zip(all_labels["Charge Type"], all_labels["Charge Description"])]
    unique_labels = all_labels[is_new].reset_index(drop=True)
    print(f"{len(all_labels):,} labels in staging.charge, {len(existing):,} already mapped, {len(unique_labels):,} new")


def refresh_counts(conn):
    """Update row_count / total_value / updated_at on existing mapping rows."""
    known = all_labels[[k in existing_keys for k in zip(all_labels["Charge Type"], all_labels["Charge Description"])]]
    if known.empty:
        return 0
    conn.execute(text(f"""
        UPDATE {SCHEMA}.charge_mapping m
        SET row_count = v.row_count, total_value = v.total_value, updated_at = now()
        FROM unnest(CAST(:ct AS text[]), CAST(:cd AS text[]), CAST(:rc AS bigint[]), CAST(:tv AS numeric[]))
             AS v(charge_type, charge_description, row_count, total_value)
        WHERE m.charge_type = v.charge_type AND m.charge_description = v.charge_description
    """), {
        "ct": known["Charge Type"].tolist(),
        "cd": known["Charge Description"].tolist(),
        "rc": [int(x) for x in known["row_count"]],
        "tv": [None if pd.isna(x) else float(x) for x in known["total_value"]],
    })
    return len(known)


def apply_seed_to_existing(conn):
    """Re-apply the override seed to labels already in the mapping (incremental runs never
    re-classify them). confirmed / client_assigned set the category; disputed keeps the mapping
    and keeps its method; excluded keeps the mapping and becomes client_review. Status and
    note live in charge_mapping_review (sync_review_table), not on the mapping."""
    if args.rebuild:
        return 0   # the mapping was just rebuilt with the seed applied
    known = seed[[k in existing_keys for k in zip(seed["charge_type"], seed["label"])]]
    for r in known.itertuples():
        sets_category = r.status in overrides.SETS_CATEGORY
        conn.execute(text(f"""
            UPDATE {SCHEMA}.charge_mapping
            SET major_category = CASE WHEN :sets THEN :category ELSE major_category END,
                subcategory    = CASE WHEN :sets THEN :subcategory ELSE subcategory END,
                method         = CASE WHEN :status = 'disputed' THEN method ELSE :cr END,
                similarity     = CASE WHEN :sets THEN NULL ELSE similarity END,
                updated_at = now()
            WHERE charge_type = :ctype AND charge_description = :label
        """), {"status": r.status, "sets": sets_category, "category": r.category,
               "subcategory": r.subcategory, "cr": overrides.CLIENT_REVIEW, "ctype": r.charge_type, "label": r.label})
    return len(known)


def sync_review_table(conn):
    """Make <schema>.charge_mapping_review equal to the seed (replace; the seed is the source of truth)."""
    conn.execute(text(f"TRUNCATE {SCHEMA}.charge_mapping_review"))
    review = seed.rename(columns={"label": "charge_description", "status": "review_status"})[
        ["charge_type", "charge_description", "review_status", "note", "source", "reviewed_on"]]
    review = review.assign(note=review["note"].replace("", None), reviewed_on=pd.to_datetime(review["reviewed_on"]).dt.date)
    review.to_sql("charge_mapping_review", conn, schema=SCHEMA, if_exists="append", index=False)


def report_unmatched_seed_rows(conn):
    """Seed rows that match no label in the mapping after this run (expected: none)."""
    mapped = set(map(tuple, pd.read_sql(
        text(f"SELECT charge_type, charge_description FROM {SCHEMA}.charge_mapping"), conn).values))
    missing = [k for k in zip(seed["charge_type"], seed["label"]) if k not in mapped]
    print(f"Seed rows that matched no label: {len(missing)}")
    for k in missing:
        print(f"  {k}")


if unique_labels.empty:
    with engine.begin() as conn:
        n_overridden = apply_seed_to_existing(conn)
        n = refresh_counts(conn)
        sync_review_table(conn)
        report_unmatched_seed_rows(conn)
    print(f"No new labels. Refreshed counts on {n:,} existing rows, re-applied {n_overridden} seed rows.")
    sys.exit(0)

n_major = len(TAXONOMY) + 1  # + catch-all
print(f"{n_major} major categories (incl. catch-all)")
for major, subcats in TAXONOMY.items():
    print(f"  {major}: {len(subcats)} subcategories")

print(f"{len(unique_labels):,} unique (Charge Type, Charge Description) combinations covering {unique_labels['row_count'].sum():,} rows")

# ---------------------------------------------------------------------------
# Apply rules to unique (Charge Type, Charge Description) combinations
# ---------------------------------------------------------------------------
classified = unique_labels.apply(lambda r: rule_classify_row(r["Charge Type"], r["Charge Description"]), axis=1)
unique_labels["Major Category"] = [c[0] if c else None for c in classified]
unique_labels["Subcategory"] = [c[1] if c else None for c in classified]
unique_labels["Method"] = ["rule" if c else None for c in classified]
unique_labels["similarity"] = float("nan")

# Client overrides outrank the rules; confirmed / client_assigned labels are decided here and
# skip TF-IDF below. disputed / excluded keep what the rules (or TF-IDF) decide.
unique_labels, _ = overrides.apply_to_labels(unique_labels, seed)

matched_rows = unique_labels.loc[unique_labels["Major Category"].notna(), "row_count"].sum()
print(f"Rule-based coverage: {matched_rows / unique_labels['row_count'].sum() * 100:.1f}% of rows")

# ---------------------------------------------------------------------------
# TF-IDF fallback for labels the rules missed (src/tfidf_fallback.py)
# ---------------------------------------------------------------------------
def label_text(row):
    return (normalize(row["Charge Type"]) + " " + normalize(row["Charge Description"])).strip()


unmatched = unique_labels["Major Category"].isna()
todo = [idx for idx in unique_labels.index[unmatched] if label_text(unique_labels.loc[idx])]
fallback_results = dict(zip(todo, TfidfFallback(args.threshold).classify([label_text(unique_labels.loc[i]) for i in todo])))

for idx, result in fallback_results.items():
    if result:
        (major, sub), sim = result
        unique_labels.loc[idx, "Major Category"] = major
        unique_labels.loc[idx, "Subcategory"] = sub
        unique_labels.loc[idx, "Method"] = "tfidf_fallback"
        unique_labels.loc[idx, "similarity"] = sim

unique_labels = overrides.finalize_after_fallback(unique_labels)

unique_labels["Major Category"] = unique_labels["Major Category"].fillna(CATCH_ALL[0])
unique_labels["Subcategory"] = unique_labels["Subcategory"].fillna(CATCH_ALL[1])
unique_labels["Method"] = unique_labels["Method"].fillna("none")

final_coverage = 1 - unique_labels.loc[unique_labels["Major Category"] == CATCH_ALL[0], "row_count"].sum() / unique_labels["row_count"].sum()
print(f"Final coverage after TF-IDF fallback: {final_coverage * 100:.1f}% of rows categorized")
print(unique_labels["Method"].value_counts())

# ---------------------------------------------------------------------------
# Write <schema>.charge_mapping in one transaction: append the new rows and refresh
# counts on the existing ones (--rebuild: truncate and re-insert everything).
# Never if_exists="replace" -- views depend on the table.
# ---------------------------------------------------------------------------
mapping = unique_labels.rename(columns={
    "Charge Type": "charge_type",
    "Charge Description": "charge_description",
    "Major Category": "major_category",
    "Subcategory": "subcategory",
    "Method": "method",
})[[
    "charge_type", "charge_description", "major_category", "subcategory",
    "method", "similarity", "row_count", "total_value",
]]

dupes = mapping.duplicated(["charge_type", "charge_description"], keep=False)
if dupes.any():
    raise ValueError(f"{dupes.sum()} rows violate (charge_type, charge_description) uniqueness:\n{mapping[dupes]}")

with engine.begin() as conn:
    n_overridden = apply_seed_to_existing(conn)
    if args.rebuild:
        conn.execute(text(f"TRUNCATE {SCHEMA}.charge_mapping"))
    mapping.to_sql("charge_mapping", conn, schema=SCHEMA, if_exists="append", index=False)
    n_refreshed = refresh_counts(conn)
    sync_review_table(conn)
    report_unmatched_seed_rows(conn)
verb = "Rebuilt with" if args.rebuild else "Appended"
print(f"{verb} {len(mapping):,} rows in {SCHEMA}.charge_mapping; refreshed counts on {n_refreshed:,} existing rows"
      f"; re-applied {n_overridden} seed rows to existing labels")
