"""Score the categorization pipeline on the hand-labelled evaluation set.

eval/fallback_eval_set.csv holds 200 labels the taxonomy rules missed when it was
built (the 150 most frequent plus 50 random ones from the tail), each with the
subcategory it should get -- several separated by ";" when more than one is
defensible -- or UNSURE (not scored). Labels were assigned by reading the label text, not verified against
source invoices, so treat the numbers as a regression check, not ground truth.

Runs the current rules, then the TF-IDF fallback at several thresholds, and
reports how many labels get a category and how many of those are right. No
database access needed. Run from the repo root:
    python eval/evaluate_fallback.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from taxonomy import normalize, rule_classify_row  # noqa: E402
from tfidf_fallback import SIMILARITY_THRESHOLD, TfidfFallback  # noqa: E402

ev = pd.read_csv(REPO_ROOT / "eval" / "fallback_eval_set.csv", keep_default_na=False)
ev = ev[ev.expected != "UNSURE"].copy()
ev["text"] = [(normalize(a) + " " + normalize(b)).strip() for a, b in zip(ev.charge_type, ev.charge_description)]
ev["rule"] = [rule_classify_row(a, b) for a, b in zip(ev.charge_type, ev.charge_description)]
raw = TfidfFallback(threshold=0).classify(list(ev.text))


def is_correct(category, expected):
    return category is not None and category[1] in expected.split(";")


print(f"{len(ev)} scored labels (UNSURE left out)")
rule_hits = ev.rule.notna()
rule_ok = sum(is_correct(c, e) for c, e in zip(ev.rule[rule_hits], ev.expected[rule_hits]))
print(f"rules now match {rule_hits.sum()} of them, {rule_ok} correctly\n")

rows = []
for threshold in (0.25, 0.30, 0.35, 0.40):
    final = [c if c is not None else (r[0] if r[1] >= threshold else None) for c, r in zip(ev.rule, raw)]
    assigned = sum(c is not None for c in final)
    correct = sum(is_correct(c, e) for c, e in zip(final, ev.expected))
    rows.append({
        "threshold": f"{threshold:.2f}" + (" (current)" if threshold == SIMILARITY_THRESHOLD else ""),
        "categorized": assigned,
        "coverage": f"{assigned / len(ev):.0%}",
        "correct": correct,
        "precision": f"{correct / assigned:.0%}" if assigned else "-",
    })
print("rules + TF-IDF fallback:")
print(pd.DataFrame(rows).to_string(index=False))
