"""Client review overrides: the reviewed seed that outranks every rule.

seeds/charge_mapping_overrides.csv holds the client's answers to the charge-mapping review,
keyed the same way analytics.charge_mapping is: (charge_type, label) where label is the
charge description. Precedence in jobs/charge_type.py:

    override seed  >  written rules (REVIEWED_LABELS, PRIORITY_OVERRIDES, TAXONOMY)  >  TF-IDF similarity

Statuses:
    confirmed        client agreed with our mapping            -> category/subcategory from the seed
    client_assigned  client gave the category (was Other)       -> category/subcategory from the seed
    disputed         client disagreed, no answer yet            -> mapping unchanged, note attached
    excluded         client says the line is not shipping cost  -> mapping unchanged; the lines are
                     flagged is_excluded at line level (see the exclusion migration)

All of them get method = 'client_review' except `disputed`, which keeps the method it had.
"""

import pandas as pd

from taxonomy import CATCH_ALL, TAXONOMY

CLIENT_REVIEW = "client_review"
SEED_COLUMNS = ["label", "charge_type", "category", "subcategory", "status", "source", "reviewed_on", "note"]
STATUSES = {"confirmed", "client_assigned", "disputed", "excluded"}
SETS_CATEGORY = {"confirmed", "client_assigned"}   # statuses whose seed category replaces the rule result


def valid_pairs():
    pairs = {(major, sub) for major, subcats in TAXONOMY.items() for sub in subcats}
    pairs.add(CATCH_ALL)
    return pairs


def load_seed(path):
    """Read and validate the seed. Raises ValueError listing every problem found."""
    seed = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(seed.columns) != SEED_COLUMNS:
        raise ValueError(f"{path}: columns must be {SEED_COLUMNS}, got {list(seed.columns)}")
    problems = [f"unknown status {r.status!r} for {r.label!r}" for r in seed[~seed["status"].isin(STATUSES)].itertuples()]
    problems += [f"duplicate key ({r.charge_type!r}, {r.label!r})"
                 for r in seed[seed.duplicated(["charge_type", "label"], keep=False)].itertuples()]
    pairs = valid_pairs()
    for r in seed.itertuples():
        if r.status in SETS_CATEGORY and (r.category, r.subcategory) not in pairs:
            problems.append(f"{r.label!r}: ({r.category!r}, {r.subcategory!r}) is not in the taxonomy "
                            "(add the subcategory to src/taxonomy.py first -- no silent taxonomy)")
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return seed


def apply_to_labels(labels, seed):
    """Apply the seed to a frame of unique labels (jobs/charge_type.py's unique_labels).

    Needs columns 'Charge Type', 'Charge Description', 'Major Category', 'Subcategory', 'Method'.
    Adds 'review_status' and 'note'. Returns (labels, matched_keys): the seed keys that hit a
    label, so the caller can report seed rows that matched nothing.

    confirmed / client_assigned rows are fully decided here, so the caller can skip TF-IDF for
    them. disputed / excluded rows keep whatever the rules (or, later, TF-IDF) decide; call
    `finalize_after_fallback` once that is done.
    """
    labels = labels.copy()
    labels["review_status"] = None
    labels["note"] = None
    index = {(r.charge_type, r.label): r for r in seed.itertuples()}
    matched = set()
    for i, key in zip(labels.index, zip(labels["Charge Type"], labels["Charge Description"])):
        hit = index.get(key)
        if hit is None:
            continue
        matched.add(key)
        labels.at[i, "review_status"] = hit.status
        labels.at[i, "note"] = hit.note or None
        if hit.status in SETS_CATEGORY:
            labels.at[i, "Major Category"] = hit.category
            labels.at[i, "Subcategory"] = hit.subcategory
            labels.at[i, "Method"] = CLIENT_REVIEW
    return labels, matched


def finalize_after_fallback(labels):
    """`excluded` labels become client_review once their (unchanged) mapping is final."""
    labels = labels.copy()
    labels.loc[labels["review_status"] == "excluded", "Method"] = CLIENT_REVIEW
    return labels
