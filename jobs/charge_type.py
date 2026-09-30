"""
Categorize unique charge labels into analytics.charge_mapping (Supabase/Postgres).

Local replacement for the Glue job in charge.py. Classification logic is
unchanged: rule_classify_row from src/taxonomy.py first, then a TF-IDF
char_wb 3-5 gram fallback with SIMILARITY_THRESHOLD = 0.35.

Reads aggregated (charge_type, charge_description) combinations from
staging.charge, classifies them, and replaces the contents of
analytics.charge_mapping in a single transaction.

Requires SUPABASE_DB_URL (see .env.example).

Run from the repo root:
    python jobs/charge_type.py

To experiment without touching the shared analytics schema, write to your
own sandbox schema instead:
    python jobs/charge_type.py --schema sandbox_yourname
The first run creates the schema plus its own charge_mapping table and
charge_categorized / charge_review views (from the migration SQL, pointed at
that schema). staging.charge is only ever read.

Fallback for labels the rules miss (--fallback):
    tfidf      character n-gram TF-IDF against the taxonomy patterns (default)
    embedding  experimental: multilingual sentence embeddings, k-nearest
               rule-classified labels (src/embedding_fallback.py). Sandbox only.
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

from taxonomy import CATCH_ALL, EXCLUDED, FALLBACK_RULES, TAXONOMY, TFIDF_EXCLUDE, rule_classify_row  # noqa: E402
from tfidf_fallback import TfidfFallback  # noqa: E402

MIGRATION_SQL = REPO_ROOT / "supabase" / "migrations" / "20260929000000_analytics_charge_mapping.sql"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--schema", default="analytics",
                    help="schema to write charge_mapping into (default: analytics, the shared one)")
parser.add_argument("--fallback", choices=["tfidf", "embedding"], default="tfidf",
                    help="how to classify labels the rules miss (default: tfidf)")
parser.add_argument("--threshold", type=float,
                    help="minimum similarity to accept a fallback match "
                         "(default: 0.35 for tfidf, EMBEDDING_THRESHOLD for embedding)")
args = parser.parse_args()
SCHEMA = args.schema
FALLBACK = args.fallback
# The schema name is interpolated into SQL, so only allow plain identifiers.
if not re.fullmatch(r"[a-z_][a-z0-9_]*", SCHEMA):
    sys.exit(f"--schema must be a lowercase identifier like sandbox_yourname, got {SCHEMA!r}")
if FALLBACK != "tfidf" and SCHEMA == "analytics":
    sys.exit("--fallback embedding is experimental; write it to a sandbox with --schema sandbox_yourname")

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

matched_rows = unique_labels.loc[unique_labels["Major Category"].notna(), "row_count"].sum()
print(f"Rule-based coverage: {matched_rows / unique_labels['row_count'].sum() * 100:.1f}% of rows")

# ---------------------------------------------------------------------------
# Fallback for labels the rules missed
# ---------------------------------------------------------------------------
EMBEDDING_THRESHOLD = 0.75  # calibrated on this data: see eval/fallback_eval_set.csv


def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def label_text(row):
    return (normalize(row["Charge Type"]) + " " + normalize(row["Charge Description"])).strip()


unmatched = unique_labels["Major Category"].isna()
todo = [idx for idx in unique_labels.index[unmatched] if label_text(unique_labels.loc[idx])]
if FALLBACK == "tfidf":
    tfidf = TfidfFallback() if args.threshold is None else TfidfFallback(args.threshold)
    fallback_results = dict(zip(todo, tfidf.classify([label_text(unique_labels.loc[i]) for i in todo])))
else:
    from embedding_fallback import EmbeddingFallback

    # References: every label the rules classified (same exclusions as the TF-IDF
    # corpus), plus the subcategory name itself so classes without examples can win.
    # Labels carrying a FALLBACK_RULES prefix are left out: whether their class came
    # from the prefix ("Retourzendingen Surge Fee - Commercieel" -> Returns) or from
    # the charge itself, the prefix words dominate the embedding, so as neighbours
    # they pull plain "Surge Fee - Commercieel" into Returns.
    ruled = unique_labels[~unmatched]
    references = {
        (label_text(r), (r["Major Category"], r["Subcategory"]))
        for _, r in ruled.iterrows()
        if (r["Major Category"], r["Subcategory"]) not in TFIDF_EXCLUDE | {EXCLUDED}
        and not any(re.search(pat, label_text(r)) for pat, _ in FALLBACK_RULES)
    }
    references |= {
        (f"{major}: {sub}", (major, sub))
        for major, subcats in TAXONOMY.items() for sub in subcats
        if (major, sub) not in TFIDF_EXCLUDE
    }
    print(f"Embedding {len(todo):,} unmatched labels against {len(references):,} references...")
    embedder = EmbeddingFallback(sorted(references), threshold=EMBEDDING_THRESHOLD if args.threshold is None else args.threshold)
    fallback_results = dict(zip(todo, embedder.classify([label_text(unique_labels.loc[i]) for i in todo])))

for idx, result in fallback_results.items():
    if result:
        (major, sub), sim = result
        unique_labels.loc[idx, "Major Category"] = major
        unique_labels.loc[idx, "Subcategory"] = sub
        unique_labels.loc[idx, "Method"] = f"{FALLBACK}_fallback"
        unique_labels.loc[idx, "similarity"] = sim

unique_labels["Major Category"] = unique_labels["Major Category"].fillna(CATCH_ALL[0])
unique_labels["Subcategory"] = unique_labels["Subcategory"].fillna(CATCH_ALL[1])
unique_labels["Method"] = unique_labels["Method"].fillna("none")

final_coverage = 1 - unique_labels.loc[unique_labels["Major Category"] == CATCH_ALL[0], "row_count"].sum() / unique_labels["row_count"].sum()
print(f"Final coverage after {FALLBACK} fallback: {final_coverage * 100:.1f}% of rows categorized")
print(unique_labels["Method"].value_counts())

# ---------------------------------------------------------------------------
# Write <schema>.charge_mapping: truncate + append in one transaction.
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
    if SCHEMA != "analytics":
        # Same DDL as the shared migration (all IF NOT EXISTS / OR REPLACE), just
        # retargeted at the sandbox schema. The shared schema is never re-created here.
        ddl = re.sub(r"\banalytics\b", SCHEMA, MIGRATION_SQL.read_text())
        conn.exec_driver_sql(ddl)
    conn.execute(text(f"TRUNCATE {SCHEMA}.charge_mapping"))
    mapping.to_sql("charge_mapping", conn, schema=SCHEMA, if_exists="append", index=False)
print(f"Wrote {len(mapping):,} rows to {SCHEMA}.charge_mapping")
