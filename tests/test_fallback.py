"""Checks on the fuzzy-matching (TF-IDF) fallback in src/tfidf_fallback.py."""

import pandas as pd
import pytest

from taxonomy import TFIDF_EXCLUDE, normalize, rule_classify_row
from tfidf_fallback import SIMILARITY_THRESHOLD, TfidfFallback

# Minimum scores on eval/fallback_eval_set.csv (200 hand-labelled labels).
# At the time of writing: 99% correct, 67% of labels categorized. Raise these as it improves.
MIN_PRECISION = 0.95   # share of categorized labels that are correct
MIN_COVERAGE = 0.60    # share of labels that get any category


@pytest.fixture(scope="module")
def fallback():
    return TfidfFallback()


def label_text(charge_type, charge_desc):
    return (normalize(charge_type) + " " + normalize(charge_desc)).strip()


def test_fallback_score_on_hand_labelled_set(repo_root, fallback):
    ev = pd.read_csv(repo_root / "eval" / "fallback_eval_set.csv", keep_default_na=False)
    ev = ev[ev.expected != "UNSURE"]
    texts = [label_text(a, b) for a, b in zip(ev.charge_type, ev.charge_description)]
    rules = [rule_classify_row(a, b) for a, b in zip(ev.charge_type, ev.charge_description)]
    fuzzy = fallback.classify(texts)
    final = [r if r else (f[0] if f else None) for r, f in zip(rules, fuzzy)]

    assigned = [(c, e) for c, e in zip(final, ev.expected) if c is not None]
    correct = sum(c[1] in e.split(";") for c, e in assigned)
    precision, coverage = correct / len(assigned), len(assigned) / len(ev)
    wrong = [f"{t!r} -> {c[1]} (expected {e})" for t, (c, e) in
             zip([t for t, c in zip(texts, final) if c is not None], assigned) if c[1] not in e.split(";")]

    assert precision >= MIN_PRECISION, f"precision {precision:.0%} < {MIN_PRECISION:.0%}. Wrong:\n" + "\n".join(wrong)
    assert coverage >= MIN_COVERAGE, f"coverage {coverage:.0%} < {MIN_COVERAGE:.0%}"


# Labels the fallback once put in the wrong place (each is described in the comments of
# taxonomy.py / tfidf_fallback.py). They may get another category or none -- just not this one.
MUST_NOT_MATCH = [
    ("Weekend Charge", "Chargeback / Reversal"),
    ("Service Charge", "Chargeback / Reversal"),
    ("Order Lane Charge", "Chargeback / Reversal"),
    ("Terminal Charges", "Chargeback / Reversal"),
    ("Order Lane Charge", "Chargeback Fuel Surcharge"),
    ("Road transport", "Ocean Freight"),
    ("Dedicated transport", "Ocean Freight"),
    ("Early Surcharge", "Domestic Fuel Surcharge"),
    ("JFK Surcharge", "Domestic Fuel Surcharge"),
    ("DG Surcharge", "Domestic Fuel Surcharge"),
    ("Custody Fee", "Customer Service / Resolution Fee"),
]


@pytest.mark.parametrize("label, forbidden", MUST_NOT_MATCH, ids=[f"{l}-not-{f}" for l, f in MUST_NOT_MATCH])
def test_fallback_avoids_known_mistakes(fallback, label, forbidden):
    result = fallback.classify([normalize(label)])[0]
    assert result is None or result[0][1] != forbidden, f"{label!r} fuzzy-matched into {forbidden} ({result[1]:.2f})"


def test_fallback_never_returns_excluded_categories(repo_root):
    # Even with no threshold at all, TFIDF_EXCLUDE categories must never come back.
    ev = pd.read_csv(repo_root / "eval" / "fallback_eval_set.csv", keep_default_na=False)
    texts = [label_text(a, b) for a, b in zip(ev.charge_type, ev.charge_description)]
    hits = [(t, r[0]) for t, r in zip(texts, TfidfFallback(threshold=0).classify(texts)) if r and r[0] in TFIDF_EXCLUDE]
    assert not hits


def test_fallback_respects_threshold(repo_root, fallback):
    ev = pd.read_csv(repo_root / "eval" / "fallback_eval_set.csv", keep_default_na=False)
    texts = [label_text(a, b) for a, b in zip(ev.charge_type, ev.charge_description)]
    low = [(t, r[1]) for t, r in zip(texts, fallback.classify(texts)) if r and r[1] < SIMILARITY_THRESHOLD]
    assert not low
