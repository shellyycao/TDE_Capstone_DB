"""Tests for the client-review work: LINEHAUL spellings and the override seed.

No database access needed. Run from the repo root:
    python eval/test_client_review.py
Exits non-zero if any check fails.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import overrides  # noqa: E402
from taxonomy import rule_classify_row  # noqa: E402

BASE_FREIGHT = ("Line Haul / Base Transportation", "General / Mode Not Specified")
failures = []


def check(name, got, expected):
    if got != expected:
        failures.append(f"{name}: expected {expected!r}, got {got!r}")


# --- the client is relabelling cargo-description lines as LINEHAUL: every spelling must be base freight
# (case- and space-insensitive), with and without a generic charge type in front.
SPELLINGS = ["LINEHAUL", "LINE HAUL", "Line Haul", "Linehaul", "linehaul", "line haul",
             "Line  Haul", "  LINEHAUL  ", "LINE\tHAUL", "Line-Haul"]
for spelling in SPELLINGS:
    for ctype in ("", "Freight", "Accessorial"):
        check(f"rule_classify_row({ctype!r}, {spelling!r})", rule_classify_row(ctype, spelling), BASE_FREIGHT)

# a fuel surcharge on a linehaul stays a fuel surcharge
check("linehaul fuel surcharge", rule_classify_row("", "LINEHAUL FUEL SURCHARGE")[0], "Fuel Surcharge")

# --- seed is valid and says what the client said
seed = overrides.load_seed(REPO_ROOT / "seeds" / "charge_mapping_overrides.csv")
by_label = {r.label: r for r in seed.itertuples()}
status_counts = seed["status"].value_counts().to_dict()
check("seed rows", len(seed), 161)
check("status counts", status_counts, {"confirmed": 109, "client_assigned": 50, "excluded": 1, "disputed": 1})
check("source", set(seed["source"]), {"client_review"})
check("reviewed_on", set(seed["reviewed_on"]), {"2026-10-15"})
check("Look Up Amount", (by_label["Look Up Amount"].status, by_label["Look Up Amount"].note),
      ("excluded", "client: factoring lookups, not shipping cost"))
check("PICKUP MILES ADDITIONAL", by_label["PICKUP MILES ADDITIONAL"].status, "disputed")
ltl = seed[seed["subcategory"] == "LTL Freight (Commodity Line)"]
check("LTL rows confirmed as base freight", (set(ltl["status"]), set(ltl["note"])), ({"confirmed"}, {"client: base freight"}))
check("needs_check labels", {r.label for r in seed.itertuples() if r.note == "needs_check"},
      {"ADDL INVOICE", "ADDITIONAL INVOICES", "P/U", "PU AND HOLD", "Reclass", "ADDITIONAL CLASSIFICATIONS",
       "DOC", "DOCS", "Original Invoice Amount"})
check("no seed row left in the catch-all", int(((seed["status"] == "client_assigned") & (seed["category"] == "Other / Uncategorized")).sum()), 0)

# --- precedence: seed > rules > similarity
labels = pd.DataFrame({
    "Charge Type": ["Service Charge", "Look Up Amount", "PICKUP MILES ADDITIONAL", "Saturday Delivery", "Accessorial"],
    "Charge Description": ["Service Charge", "Look Up Amount", "PICKUP MILES ADDITIONAL", "Saturday Delivery", "UNSEEN FEE"],
})
classified = labels.apply(lambda r: rule_classify_row(r["Charge Type"], r["Charge Description"]), axis=1)
labels["Major Category"] = [c[0] if c else None for c in classified]
labels["Subcategory"] = [c[1] if c else None for c in classified]
labels["Method"] = ["rule" if c else None for c in classified]
labels["Charge Type"] = ["Service Charge", "Look Up Amount", "PICKUP MILES ADDITIONAL", "Accessorial", "Accessorial"]
out, hits = overrides.apply_to_labels(labels, seed)
out = overrides.finalize_after_fallback(out)
row = lambda i: (out.at[i, "Major Category"], out.at[i, "Subcategory"], out.at[i, "Method"], out.at[i, "review_status"])
check("client_assigned overrides the unmatched label", row(0), ("Administrative & Service Fees", "General Service Fee", "client_review", "client_assigned"))
check("excluded keeps its mapping, counts as client_review", row(1), (*BASE_FREIGHT, "client_review", "excluded"))
check("disputed keeps mapping and method, gets the note", row(2) + (out.at[2, "note"],),
      ("Administrative & Service Fees", "Pickup Service", "rule", "disputed", "client: wrong, no destination given; pending follow-up"))
check("label not in the seed is untouched", row(3)[:3], (*rule_classify_row("Accessorial", "Saturday Delivery"), "rule"))
check("unseen label stays unmatched for TF-IDF", pd.isna(out.at[4, "Major Category"]), True)
check("matched seed keys", len(hits), 3)

if failures:
    print(f"{len(failures)} FAILED")
    print("\n".join(failures))
    sys.exit(1)
print("all checks passed")
