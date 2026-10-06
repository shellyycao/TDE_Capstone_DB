"""
Build the dashboard summary from Supabase.

Local replacement for the Glue job in legacy/glue/dashboard.py. Runs the same
three aggregations (major/subcategory, per company, per carrier) against
analytics.charge_categorized joined to staging.shipment, and writes
site/summary.json in the exact shape site/index.html expects.

Requires SUPABASE_DB_URL (see .env.example). Run from the repo root:
    python jobs/build_dashboard.py
To preview a sandbox run of charge_type.py (see its --schema flag):
    python jobs/build_dashboard.py --schema sandbox_yourname
Preview:
    cd site && python -m http.server 8000
"""

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from taxonomy import CATCH_ALL  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--schema", default="analytics",
                    help="schema to read charge_categorized from (default: analytics, the shared one)")
parser.add_argument("--out", default="summary.json",
                    help="file name under site/ to write (default: summary.json); view another with "
                         "http://localhost:8000/?data=<name>")
args = parser.parse_args()
SCHEMA = args.schema
# The schema name is interpolated into SQL, so only allow plain identifiers.
if not re.fullmatch(r"[a-z_][a-z0-9_]*", SCHEMA):
    sys.exit(f"--schema must be a lowercase identifier like sandbox_yourname, got {SCHEMA!r}")

# Run the tests first, so the dashboard is never built from incomplete or outdated data.
# The tests check the shared analytics tables, so sandbox previews skip them.
if SCHEMA == "analytics":
    result = subprocess.run([sys.executable, "-m", "pytest", str(REPO_ROOT / "tests"), "-q"])
    if result.returncode != 0:
        sys.exit("Checks failed - dashboard not built. Fix the issues above and run again.")

load_dotenv(REPO_ROOT / ".env")
DB_URL = os.environ.get("SUPABASE_DB_URL")
if not DB_URL:
    sys.exit("SUPABASE_DB_URL is not set (copy .env.example to .env and fill it in)")
for prefix in ("postgres://", "postgresql://"):
    if DB_URL.startswith(prefix):
        DB_URL = "postgresql+psycopg2://" + DB_URL[len(prefix):]
        break

SUMMARY_PATH = REPO_ROOT / "site" / Path(args.out).name

# charge_categorized labels lines with no mapping 'Unmapped'. Fold those into
# the taxonomy catch-all (which index.html shows as "Other / Uncategorized").
# The folding happens in a subquery so lines already classified as the
# catch-all and 'Unmapped' lines land in the same GROUP BY bucket.
QUERY = text(f"""
SELECT major_category, subcategory,
       count(*) AS num_rows,
       coalesce(sum(charge_value), 0) AS total_value
FROM (
  SELECT CASE WHEN major_category = 'Unmapped' THEN :catch_major ELSE major_category END AS major_category,
         CASE WHEN major_category = 'Unmapped' THEN :catch_sub   ELSE subcategory    END AS subcategory,
         charge_value
  FROM {SCHEMA}.charge_categorized
) t
GROUP BY major_category, subcategory
ORDER BY major_category, total_value DESC
""")

ENTITY_QUERY = """
SELECT {entity} AS entity,
       CASE WHEN c.major_category = 'Unmapped' THEN :catch_major ELSE c.major_category END AS major_category,
       count(*) AS num_rows,
       coalesce(sum(c.charge_value), 0) AS total_value
FROM {schema}.charge_categorized c
LEFT JOIN staging.shipment s ON c.shipment_id = s.shipment_id
GROUP BY 1, 2
ORDER BY entity, total_value DESC
"""

COMPANY_QUERY = text(ENTITY_QUERY.format(
    schema=SCHEMA,
    entity="coalesce(nullif(trim(s.client_name), ''), 'Unspecified account')"))
CARRIER_QUERY = text(ENTITY_QUERY.format(
    schema=SCHEMA,
    entity="coalesce(nullif(trim(s.carrier_name), ''), 'Unspecified carrier')"))

# Only ship the top N in the JSON so the dashboard payload (and the
# dropdown lists) stay manageable even with many companies or carriers.
TOP_N_COMPANIES = 24
TOP_N_CARRIERS = 24

engine = create_engine(DB_URL)
params = {"catch_major": CATCH_ALL[0], "catch_sub": CATCH_ALL[1]}

print(f"Reading {SCHEMA}.charge_categorized")
with engine.connect() as conn:
    print("Running major/subcategory summary query...")
    rows = conn.execute(QUERY, params).fetchall()

    print("Running per-company breakdown query...")
    company_rows = conn.execute(COMPANY_QUERY, params).fetchall()

    print("Running per-carrier breakdown query...")
    carrier_rows = conn.execute(CARRIER_QUERY, params).fetchall()

# Group the (major, sub) rows into nested category objects, preserving
# first-seen order of major categories (ORDER BY major_category keeps each
# major category's rows contiguous).
majors = OrderedDict()
for major_name, sub_name, num_rows, total_value in rows:
    num_rows = int(num_rows)
    total_value = round(float(total_value), 2)

    if major_name not in majors:
        majors[major_name] = {"name": major_name, "row_count": 0, "total_value": 0.0, "subs": []}

    majors[major_name]["row_count"] += num_rows
    majors[major_name]["total_value"] = round(majors[major_name]["total_value"] + total_value, 2)
    majors[major_name]["subs"].append({
        "name": sub_name,
        "row_count": num_rows,
        "total_value": total_value,
    })

categories = list(majors.values())


def group_by_entity(entity_rows, top_n):
    """Group (entity_name, major, num_rows, total_value) rows into per-entity
    objects with a nested category breakdown, then split into the top N by
    charge magnitude and everything else."""
    entities_map = OrderedDict()
    for entity_name, major_name, num_rows, total_value in entity_rows:
        num_rows = int(num_rows)
        total_value = round(float(total_value), 2)

        if entity_name not in entities_map:
            entities_map[entity_name] = {
                "name": entity_name,
                "row_count": 0,
                "total_value": 0.0,
                "abs_total_value": 0.0,
                "categories": [],
            }

        e = entities_map[entity_name]
        e["row_count"] += num_rows
        e["total_value"] = round(e["total_value"] + total_value, 2)
        e["abs_total_value"] = round(e["abs_total_value"] + abs(total_value), 2)
        e["categories"].append({
            "name": major_name,
            "row_count": num_rows,
            "total_value": total_value,
        })

    all_entities = sorted(entities_map.values(), key=lambda e: e["abs_total_value"], reverse=True)
    top = all_entities[:top_n]
    rest = all_entities[top_n:]
    for e in top:
        del e["abs_total_value"]
    return all_entities, top, rest


all_companies, top_companies, rest_companies = group_by_entity(company_rows, TOP_N_COMPANIES)
all_carriers, top_carriers, rest_carriers = group_by_entity(carrier_rows, TOP_N_CARRIERS)

summary = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_transactions": sum(c["row_count"] for c in categories),
    "categories": categories,
    "companies": top_companies,
    "other_companies_count": len(rest_companies),
    "other_companies_row_count": sum(c["row_count"] for c in rest_companies),
    "other_companies_total_value": round(sum(c["total_value"] for c in rest_companies), 2),
    "carriers": top_carriers,
    "other_carriers_count": len(rest_carriers),
    "other_carriers_row_count": sum(c["row_count"] for c in rest_carriers),
    "other_carriers_total_value": round(sum(c["total_value"] for c in rest_carriers), 2),
}

SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
print(f"Wrote summary to {SUMMARY_PATH}")
print(f"{len(categories)} major categories, {sum(len(c['subs']) for c in categories)} subcategories")
print(f"{len(all_companies)} companies found, shipping top {len(top_companies)} "
      f"({len(rest_companies)} rolled into 'other')")
print(f"{len(all_carriers)} carriers found, shipping top {len(top_carriers)} "
      f"({len(rest_carriers)} rolled into 'other')")
