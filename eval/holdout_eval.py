"""Large-sample check of the TF-IDF fallback, using rule-classified labels as the answer key.

The hand-labelled set (evaluate_fallback.py) is small because only a few hundred
labels ever reach the fallback. This script borrows the ~2,000 labels the rules
DO classify: for each one it deletes, from the TF-IDF reference documents, every
taxonomy pattern that matches that label, then asks the fallback to classify it.
That recreates the fallback's real situation -- no rule knows this label -- while
the rule's answer serves as the expected category.

Two ways to hide (--hide):
  all    remove every pattern that matches the label. Worst case: the concept is
         entirely new ("fuel surcharge" with no fuel pattern left anywhere).
  first  remove only the pattern that actually fired. Closer to a typical miss --
         a variant of a known charge ("waiting time" when only "wait time" exists),
         where sibling patterns of the same subcategory remain.

Left out, because the fallback could never be right on them by construction or the
rule's answer isn't about the charge itself: labels caught by FALLBACK_RULES (the
category comes from a prefix), subcategories outside the
TF-IDF candidate pool (TFIDF_EXCLUDE), and commodity-description lines -- anything
the LTL Freight (Commodity Line) patterns match, even when an earlier pattern won
("Freight | PLT NMFC 051080 faucets ... class 70" is caught by `\bfreight\b`). There
are hundreds of those and they are goods, not fee names, so they would swamp the
result without saying anything about fee labels. Labels caught by PRIORITY_OVERRIDES are
kept: override patterns never feed the TF-IDF corpus, so those labels are unseen
by construction.

Caveat: rule-classified labels are, on average, more "keyword-like" than the ones
the rules miss, so this likely overstates precision somewhat; it measures how the
fallback behaves across thresholds more than its exact precision on real misses.

Needs SUPABASE_DB_URL (reads the distinct labels from staging.charge). Run from the
repo root:
    python eval/holdout_eval.py [--hide all|first]
"""

import argparse
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from taxonomy import (  # noqa: E402
    FALLBACK_RULES, PRIORITY_OVERRIDES, TAXONOMY, TFIDF_EXCLUDE, normalize,
)

COMMODITY_PATTERNS = TAXONOMY["Line Haul / Base Transportation"]["LTL Freight (Commodity Line)"]
from tfidf_fallback import SIMILARITY_THRESHOLD  # noqa: E402

THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--hide", choices=["all", "first"], default="all")
HIDE = parser.parse_args().hide

load_dotenv(REPO_ROOT / ".env")
url = os.environ["SUPABASE_DB_URL"]
for prefix in ("postgres://", "postgresql://"):
    if url.startswith(prefix):
        url = "postgresql+psycopg2://" + url[len(prefix):]
        break
with create_engine(url).connect() as conn:
    labels = pd.read_sql(text("""
        SELECT coalesce(charge_type, '') AS charge_type, coalesce(charge_description, '') AS charge_description,
               count(*) AS row_count
        FROM staging.charge GROUP BY 1, 2
    """), conn)
labels["text"] = [(normalize(a) + " " + normalize(b)).strip() for a, b in zip(labels.charge_type, labels.charge_description)]


def rule_with_source(t):
    """Same order as taxonomy.rule_classify_row, but also says which stage and pattern matched."""
    for pat, result in PRIORITY_OVERRIDES:
        if re.search(pat, t):
            return result, "override", None
    for major, subcats in TAXONOMY.items():
        for sub, patterns in subcats.items():
            for p in patterns:
                if re.search(p, t):
                    return (major, sub), "taxonomy", p
    if any(re.search(p, t) for p, _ in FALLBACK_RULES):
        return None, "prefix", None
    return None, "none", None


rows = []
for t, n in zip(labels.text, labels.row_count):
    if not t:
        continue
    expected, source, fired = rule_with_source(t)
    if any(re.search(p, t) for p in COMMODITY_PATTERNS):
        continue
    if source in ("override", "taxonomy") and expected not in TFIDF_EXCLUDE:
        rows.append((t, n, expected, source, fired))
cases = pd.DataFrame(rows, columns=["text", "row_count", "expected", "source", "fired"])

# Group labels by the set of patterns that match them, so each distinct "hidden"
# corpus is fitted once.
pool = [(major, sub, patterns) for major, subcats in TAXONOMY.items()
        for sub, patterns in subcats.items() if (major, sub) not in TFIDF_EXCLUDE]
if HIDE == "all":
    cases["hidden"] = [frozenset(p for _, _, pats in pool for p in pats if re.search(p, t)) for t in cases.text]
else:
    cases["hidden"] = [frozenset([f]) if f else frozenset() for f in cases.fired]

best_label, best_sim = {}, {}
for hidden, group in cases.groupby("hidden"):
    docs = [" ".join(p.replace(r"\b", "").replace(".*", " ").replace("'?", "").replace("?", "")
                     for p in pats if p not in hidden) for _, _, pats in pool]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(docs)
    sims = cosine_similarity(vec.transform(group.text), vec.transform(docs))
    for idx, row in zip(group.index, sims):
        best_label[idx] = pool[row.argmax()][:2]
        best_sim[idx] = row.max()
cases["pred"] = pd.Series(best_label)
cases["sim"] = pd.Series(best_sim)
cases["correct"] = cases.pred == cases.expected


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


print(f"hide={HIDE}: {len(cases):,} rule-classified labels used as the answer key "
      f"({(cases.source == 'override').sum()} via overrides, {(cases.source == 'taxonomy').sum()} via taxonomy patterns)\n")
out = []
for th in THRESHOLDS:
    a = cases[cases.sim >= th]
    k, n = int(a.correct.sum()), len(a)
    lo, hi = wilson(k, n)
    out.append({
        "threshold": f"{th:.2f}" + (" (current)" if abs(th - SIMILARITY_THRESHOLD) < 1e-9 else ""),
        "categorized": n,
        "coverage": f"{n / len(cases):.0%}",
        "correct": k,
        "precision": f"{k / n:.0%}" if n else "-",
        "95% CI": f"{lo:.0%}-{hi:.0%}" if n else "-",
        "row-weighted precision": f"{a.row_count[a.correct].sum() / a.row_count.sum():.0%}" if n else "-",
    })
print(pd.DataFrame(out).to_string(index=False))

print("\nPrecision within each similarity band (is a match in this band worth accepting?):")
edges = [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.01]
cases["band"] = pd.cut(cases.sim, edges, right=False)
band = cases.groupby("band", observed=True).correct.agg(["size", "sum"])
band["precision"] = (band["sum"] / band["size"]).map("{:.0%}".format)
band["95% CI"] = [f"{lo:.0%}-{hi:.0%}" for lo, hi in (wilson(k, n) for k, n in zip(band["sum"], band["size"]))]
print(band.rename(columns={"size": "labels", "sum": "correct"}).to_string())

by_major = defaultdict(list)
for exp, ok, sim in zip(cases.expected, cases.correct, cases.sim):
    if sim >= SIMILARITY_THRESHOLD:
        by_major[exp[0]].append(ok)
print(f"\nPrecision by expected major category at the current threshold ({SIMILARITY_THRESHOLD}):")
for major, oks in sorted(by_major.items(), key=lambda kv: -len(kv[1])):
    print(f"  {major:36s} {np.mean(oks):5.0%}  (n={len(oks)})")
