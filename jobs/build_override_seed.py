"""
Build seeds/charge_mapping_overrides.csv from the client's charge-mapping review.

Input: seeds/client_review_2026-10-15.csv (category, charge_type, label, lines, review, should_be).
The client file leaves charge_type blank when the type is the same text as the label (how most
charge_mapping rows are keyed), so a blank type is resolved to the label before matching.

  review = Right     -> confirmed        current category / subcategory kept
  review = Assigned  -> client_assigned  category = should_be, subcategory from ASSIGNED below
  review = Wrong     -> disputed         current mapping kept, client's answer in the note
  CLIENT_DECISIONS   -> later answers from the client (excluded, base freight)
  NEEDS_CHECK        -> note = needs_check, for a human to decide

"Current" category / subcategory is read from analytics.charge_mapping (read-only), so run this
before the override seed has been applied, or after -- confirmed rows map to themselves either way.
Requires SUPABASE_DB_URL. Run from the repo root:
    python jobs/build_override_seed.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from overrides import SEED_COLUMNS, load_seed  # noqa: E402

CLIENT_CSV = REPO_ROOT / "seeds" / "client_review_2026-10-15.csv"
SEED_CSV = REPO_ROOT / "seeds" / "charge_mapping_overrides.csv"
SOURCE = "client_review"
REVIEWED_ON = "2026-10-15"

# The client gave only a category for the "Assigned" labels; the subcategory is ours, picked
# from the existing taxonomy. NEW_SUBCATEGORIES are the two we had to add (src/taxonomy.py,
# no patterns) because nothing existing fit -- they go in the PR notes.
GENERAL_FEE = "General Service Fee"                   # new
OTHER_ACCESSORIAL = "Other Accessorial / Special Service"  # new
ASSIGNED = {
    # Administrative & Service Fees
    "Service Charge": GENERAL_FEE,
    "SYMPHONY - VISIBILITY": GENERAL_FEE,
    "PACKING SUPPLIES": "Package Handling / Storage",
    "Weekly Printer Service Fee": GENERAL_FEE,
    "WEEKLY PRINTER SERVICE FEE": GENERAL_FEE,
    "Filler Only Pkg SM": "Package Handling / Storage",
    "Revalidation - Service Fee": GENERAL_FEE,
    "CHC": GENERAL_FEE,
    "P/U": "Pickup Service",
    # Accessorial / Delivery Surcharge
    "Metro": "Delivery Area Surcharge (DAS)",                     # existing pattern: "metro charge"
    "RDC REMOTE DELIVERY CHARGE Tier- 1": "Delivery Area Surcharge (DAS)",
    "PARKING FEES": "Wait Time / Mileage / Toll",
    "Dest. Administrative Handling": "Additional Handling",        # existing pattern: "dest. handling"
    "Fix time frame": "Appointment / Notification",
    # everything else the client put in Accessorial / Delivery Surcharge, Taxes & Customs and
    # Line Haul is resolved by ASSIGNED_DEFAULT_BY_CATEGORY below
}
ASSIGNED_DEFAULT_BY_CATEGORY = {
    "Accessorial / Delivery Surcharge": OTHER_ACCESSORIAL,
    "Taxes & Customs": "Customs / Brokerage",                       # clearance / entry lines / classifications
    "Line Haul / Base Transportation": "General / Mode Not Specified",
}

# Left for a human to decide -- inconsistent with a sibling label or overlapping a subcategory.
NEEDS_CHECK = {
    "ADDL INVOICE", "ADDITIONAL INVOICES",          # Taxes vs Accessorial
    "P/U", "PU AND HOLD",                           # Admin vs Accessorial
    "Reclass", "ADDITIONAL CLASSIFICATIONS",        # Accessorial vs Taxes
    "DOC", "DOCS",                                  # Accessorial, but Admin already has Document Fee
    "Original Invoice Amount",                      # Line Haul; may double count
}

# Later answers from the client, by label.
CLIENT_DECISIONS = {
    "Look Up Amount": ("excluded", "client: factoring lookups, not shipping cost"),
}
BASE_FREIGHT_SUBCATEGORY = "LTL Freight (Commodity Line)"
BASE_FREIGHT_NOTE = "client: base freight"
DISPUTED_NOTES = {
    "PICKUP MILES ADDITIONAL": "client: wrong, no destination given; pending follow-up",
}

load_dotenv(REPO_ROOT / ".env")
db_url = os.environ.get("SUPABASE_DB_URL")
if not db_url:
    sys.exit("SUPABASE_DB_URL is not set (copy .env.example to .env and fill it in)")
db_url = "postgresql+psycopg2://" + db_url.split("://", 1)[1]
current = pd.read_sql("SELECT charge_type, charge_description, major_category, subcategory FROM analytics.charge_mapping",
                      create_engine(db_url))
current = current.set_index(["charge_type", "charge_description"])

client = pd.read_csv(CLIENT_CSV, dtype=str, keep_default_na=False)
rows = []
for r in client.itertuples():
    ctype = r.charge_type or r.label
    if (ctype, r.label) not in current.index:
        sys.exit(f"no charge_mapping row for ({ctype!r}, {r.label!r}) -- seed row would match nothing")
    cur_major, cur_sub = current.loc[(ctype, r.label)]
    status, note = None, ""
    if r.review == "Right":
        status, major, sub = "confirmed", cur_major, cur_sub
        if sub == BASE_FREIGHT_SUBCATEGORY:
            note = BASE_FREIGHT_NOTE
    elif r.review == "Assigned":
        status, major = "client_assigned", r.should_be
        sub = ASSIGNED.get(r.label) or ASSIGNED_DEFAULT_BY_CATEGORY[major]
    elif r.review == "Wrong":
        status, major, sub = "disputed", cur_major, cur_sub
        note = DISPUTED_NOTES.get(r.label, "")   # Look Up Amount is overridden by CLIENT_DECISIONS below
    else:
        sys.exit(f"unknown review value {r.review!r} for {r.label!r}")
    if r.label in CLIENT_DECISIONS:
        status, note = CLIENT_DECISIONS[r.label]
        major, sub = cur_major, cur_sub           # excluded: mapping unchanged
    if r.label in NEEDS_CHECK:
        note = "needs_check"
    rows.append((r.label, ctype, major, sub, status, SOURCE, REVIEWED_ON, note))

seed = pd.DataFrame(rows, columns=SEED_COLUMNS)
seed.to_csv(SEED_CSV, index=False)
load_seed(SEED_CSV)   # validates statuses, duplicate keys and that every subcategory exists in the taxonomy
print(f"Wrote {len(seed)} rows to {SEED_CSV}")
print(seed["status"].value_counts().to_string())
print(f"needs_check: {(seed['note'] == 'needs_check').sum()}, base freight: {(seed['note'] == BASE_FREIGHT_NOTE).sum()}")
