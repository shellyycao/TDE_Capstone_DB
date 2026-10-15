"""Before/after checks for the client-review overrides, against two sandbox schemas.

    python eval/check_client_review.py --before sandbox_cr_baseline --after sandbox_client_review

--before: charge_type.py run with an empty seed (rules alone). --after: run with the real seed.
Needs SUPABASE_DB_URL; read-only.
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--before", required=True)
ap.add_argument("--after", required=True)
args = ap.parse_args()
for name in (args.before, args.after):
    assert re.fullmatch(r"[a-z_][a-z0-9_]*", name), name

load_dotenv(REPO_ROOT / ".env")
engine = create_engine("postgresql+psycopg2://" + os.environ["SUPABASE_DB_URL"].split("://", 1)[1])
OTHER = "Other / Uncategorized"


def lines(schema, where="true"):
    # unmapped lines count as the catch-all, as in build_dashboard.py
    return pd.read_sql(f"""
        SELECT CASE WHEN major_category = 'Unmapped' THEN '{OTHER}' ELSE major_category END AS category,
               count(*) AS n, coalesce(sum(charge_value), 0) AS usd
        FROM {schema}.charge_categorized WHERE {where} GROUP BY 1""", engine).set_index("category")


before_all, after_all = lines(args.before), lines(args.after)
before, after = lines(args.before, "NOT is_excluded"), lines(args.after, "NOT is_excluded")
cmp = before_all.join(after, how="outer", lsuffix="_before", rsuffix="_after").fillna(0)
cmp["d_lines"] = cmp["n_after"] - cmp["n_before"]
cmp["d_usd"] = (cmp["usd_after"] - cmp["usd_before"]).round(2)
print("lines per category, before (all lines) vs after (excluded lines removed):")
print(cmp.astype({"n_before": int, "n_after": int, "d_lines": int}).to_string())

# 1. client_review matches
m = pd.read_sql(f"SELECT * FROM {args.after}.charge_mapping", engine)
print("\nmethod counts on labels:", m["method"].value_counts().to_dict())
rv = pd.read_sql(f"SELECT * FROM {args.after}.charge_mapping_review", engine)
print("charge_mapping_review:", len(rv), "rows", rv["review_status"].value_counts().to_dict())
seed = pd.read_csv(REPO_ROOT / "seeds" / "charge_mapping_overrides.csv", dtype=str, keep_default_na=False)
j = seed.merge(m, left_on=["charge_type", "label"], right_on=["charge_type", "charge_description"], how="left", indicator=True)
print("seed rows matching no label:", int((j["_merge"] != "both").sum()))
print("client_review labels:", int((m["method"] == "client_review").sum()),
      "| disputed keep their method:", j.loc[j["status"] == "disputed", "method"].tolist())

# 2. lines out of Other / Uncategorized and the new share
mb = pd.read_sql(f"SELECT * FROM {args.before}.charge_mapping", engine)
ab = mb.merge(m, on=["charge_type", "charge_description"], suffixes=("_b", "_a"))
moved = ab[(ab["major_category_b"] == OTHER) & (ab["major_category_a"] != OTHER)]
print(f"\nOther -> elsewhere: {len(moved)} labels, {int(moved['row_count_a'].sum())} lines, ${moved['total_value_a'].sum():,.2f}")
print(moved.groupby(["major_category_a"])["row_count_a"].sum().to_string())
tot_b, tot_a = before_all["n"].sum(), after["n"].sum()
print(f"Other share before: {before_all.loc[OTHER, 'n'] / tot_b:.3%} of {tot_b:,} lines"
      f" | after: {after.loc[OTHER, 'n'] / tot_a:.3%} of {tot_a:,} lines (excluded removed)")

# 3. untouched categories change only where an Other label moved in
changed = ab[(ab["major_category_b"] != ab["major_category_a"]) | (ab["subcategory_b"] != ab["subcategory_a"])]
print(f"\nlabels whose category or subcategory changed: {len(changed)}")
print(changed.groupby(["major_category_b", "major_category_a"])["row_count_a"].agg(["count", "sum"]).to_string())
non_other = changed[changed["major_category_b"] != OTHER]
print("changed labels that were NOT in Other before:", len(non_other))
if len(non_other):
    print(non_other[["charge_type", "charge_description", "major_category_b", "subcategory_b",
                     "major_category_a", "subcategory_a", "row_count_a"]].to_string())
