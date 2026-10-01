"""Checks on the rule definitions in src/taxonomy.py."""

import os
import re
import subprocess
import sys

import pytest

from taxonomy import (
    CATCH_ALL, FALLBACK_RULES, FOREIGN_PATTERNS, PRIORITY_OVERRIDES, REVIEWED_LABELS,
    TAXONOMY, TFIDF_EXCLUDE, normalize, rule_classify_row,
)

ALL_SUBCATS = {(major, sub) for major, subs in TAXONOMY.items() for sub in subs}
SUB_NAMES = [sub for subs in TAXONOMY.values() for sub in subs]


# --- The rule tables are well-formed ------------------------------------------

def all_patterns():
    for major, subs in TAXONOMY.items():
        for sub, pats in subs.items():
            for p in pats:
                yield f"TAXONOMY {sub}", p
    for sub, pats in FOREIGN_PATTERNS.items():
        for p in pats:
            yield f"FOREIGN_PATTERNS {sub}", p
    for p, _ in PRIORITY_OVERRIDES:
        yield "PRIORITY_OVERRIDES", p
    for p, _ in FALLBACK_RULES:
        yield "FALLBACK_RULES", p


def test_every_pattern_compiles():
    for where, p in all_patterns():
        try:
            re.compile(p)
        except re.error as e:
            pytest.fail(f"{where}: {p!r} is not a valid pattern ({e})")


def test_patterns_are_lowercase():
    # Labels are lower-cased before matching, so an uppercase letter can never match.
    # (Escapes like \b, \s, \S are fine -- they're stripped before checking.)
    bad = [(w, p) for w, p in all_patterns() if re.sub(r"\\.", "", p) != re.sub(r"\\.", "", p).lower()]
    assert not bad, f"uppercase in patterns: {bad}"


def test_no_empty_subcategory():
    empty = [(m, s) for m, subs in TAXONOMY.items() for s, pats in subs.items() if not pats]
    assert not empty


def test_subcategory_names_are_unique():
    # FOREIGN_PATTERNS is keyed by subcategory name alone, so names must not repeat across majors.
    dupes = {s for s in SUB_NAMES if SUB_NAMES.count(s) > 1}
    assert not dupes, f"subcategory names used twice: {dupes}"


def test_foreign_patterns_use_real_subcategories():
    unknown = set(FOREIGN_PATTERNS) - set(SUB_NAMES)
    assert not unknown, f"FOREIGN_PATTERNS keys not in TAXONOMY: {unknown}"


@pytest.mark.parametrize("table_name, targets", [
    ("PRIORITY_OVERRIDES", [t for _, t in PRIORITY_OVERRIDES]),
    ("FALLBACK_RULES", [t for _, t in FALLBACK_RULES]),
    ("REVIEWED_LABELS", list(REVIEWED_LABELS.values())),
    ("TFIDF_EXCLUDE", list(TFIDF_EXCLUDE)),
])
def test_tables_point_to_real_categories(table_name, targets):
    unknown = [t for t in targets if t not in ALL_SUBCATS]
    assert not unknown, f"{table_name} points to categories not in TAXONOMY: {unknown}"


def test_reviewed_labels_are_normalized():
    # They're looked up by the normalized description, so keys must already be lowercase/stripped.
    bad = [k for k in REVIEWED_LABELS if k != normalize(k)]
    assert not bad, f"REVIEWED_LABELS keys that can never match: {bad}"


def test_catch_all_is_not_a_taxonomy_category():
    assert CATCH_ALL[0] not in TAXONOMY


# --- Basic behaviour --------------------------------------------------------

@pytest.mark.parametrize("charge_type, charge_desc", [(None, None), ("", ""), ("  ", float("nan"))])
def test_empty_label_returns_none(charge_type, charge_desc):
    assert rule_classify_row(charge_type, charge_desc) is None


def test_capital_letters_dont_matter():
    assert rule_classify_row("FUEL SURCHARGE", "FUEL SURCHARGE") == rule_classify_row("fuel surcharge", "fuel surcharge")


def test_reviewed_label_wins_over_patterns():
    for label, expected in REVIEWED_LABELS.items():
        assert rule_classify_row("", label) == expected, label


# --- The project's own example list -------------------------------------------

def test_project_multilingual_cases_pass(repo_root):
    """Runs TDE_Capstone_DB/eval/test_multilingual_rules.py (127+ labelled examples) as-is."""
    script = repo_root / "eval" / "test_multilingual_rules.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=repo_root, capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
