"""Checks on the live data in Supabase. Read-only -- see the `db` fixture in conftest.py.

Skipped automatically when SUPABASE_DB_URL isn't set in TDE_Capstone_DB/.env.
"""

from conftest import query
from taxonomy import CATCH_ALL, TAXONOMY, rule_classify_row

# Limits for the uncategorized share. At the time of writing: 0.8% of dollars, 0.3% of lines.
MAX_UNCATEGORIZED_DOLLAR_SHARE = 0.02
MAX_UNCATEGORIZED_LINE_SHARE = 0.01

VALID = {(m, s) for m, subs in TAXONOMY.items() for s in subs} | {CATCH_ALL}


def show(df, n=20):
    return "\n" + df.head(n).to_string(index=False) + (f"\n... and {len(df) - n} more" if len(df) > n else "")


def test_connection_is_read_only(db):
    assert query(db, "SHOW transaction_read_only").iloc[0, 0] == "on"


# --- Totals add up ----------------------------------------------------------

def test_every_charge_line_has_a_category(db):
    df = query(db, """
        SELECT charge_type, charge_description, count(*) AS lines
        FROM analytics.charge_categorized WHERE major_category = 'Unmapped'
        GROUP BY 1, 2 ORDER BY 3 DESC""")
    assert df.empty, "labels in staging.charge missing from charge_mapping -- run jobs/charge_type.py" + show(df)


def test_counts_and_dollars_match_raw_charges(db):
    # Per label: the line count and dollar total stored in charge_mapping equal staging.charge.
    df = query(db, """
        WITH s AS (
            SELECT coalesce(charge_type, '') AS charge_type, coalesce(charge_description, '') AS charge_description,
                   count(*) AS lines, coalesce(sum(charge_value), 0) AS dollars
            FROM staging.charge GROUP BY 1, 2)
        SELECT s.charge_type, s.charge_description, s.lines, m.row_count, s.dollars, m.total_value
        FROM s JOIN analytics.charge_mapping m USING (charge_type, charge_description)
        WHERE s.lines <> m.row_count OR abs(s.dollars - coalesce(m.total_value, 0)) > 0.01""")
    assert df.empty, "stale counts in charge_mapping -- run jobs/charge_type.py to refresh" + show(df)


def test_grand_totals_match(db):
    df = query(db, """
        SELECT (SELECT count(*) FROM staging.charge) AS raw_lines,
               (SELECT round(coalesce(sum(charge_value), 0), 2) FROM staging.charge) AS raw_dollars,
               (SELECT sum(row_count) FROM analytics.charge_mapping) AS mapped_lines,
               (SELECT round(coalesce(sum(total_value), 0), 2) FROM analytics.charge_mapping) AS mapped_dollars""")
    r = df.iloc[0]
    assert r.raw_lines == r.mapped_lines, f"lines: raw {r.raw_lines:,} vs mapping {r.mapped_lines:,}"
    assert abs(r.raw_dollars - r.mapped_dollars) <= 1, f"dollars: raw {r.raw_dollars:,} vs mapping {r.mapped_dollars:,}"


# --- Stored categories are valid --------------------------------------------

def test_stored_categories_exist_in_taxonomy(db):
    df = query(db, "SELECT DISTINCT major_category, subcategory FROM analytics.charge_mapping")
    bad = df[[(m, s) not in VALID for m, s in zip(df.major_category, df.subcategory)]]
    assert bad.empty, "categories in the database that taxonomy.py doesn't have" + show(bad)


def test_method_matches_category(db):
    df = query(db, """
        SELECT charge_type, charge_description, major_category, method, similarity FROM analytics.charge_mapping
        WHERE method NOT IN ('rule', 'tfidf_fallback', 'none')
           OR (method = 'none') <> (major_category = :catch_all)
           OR (method = 'tfidf_fallback' AND similarity IS NULL)""", catch_all=CATCH_ALL[0])
    assert df.empty, show(df)


def test_uncategorized_share_is_small(db):
    r = query(db, """
        SELECT sum(abs(total_value)) FILTER (WHERE method = 'none') / sum(abs(total_value)) AS dollar_share,
               sum(row_count) FILTER (WHERE method = 'none')::float / sum(row_count) AS line_share
        FROM analytics.charge_mapping""").fillna(0).iloc[0]
    assert r.dollar_share <= MAX_UNCATEGORIZED_DOLLAR_SHARE, f"uncategorized dollars {r.dollar_share:.1%} -- see analytics.charge_review"
    assert r.line_share <= MAX_UNCATEGORIZED_LINE_SHARE, f"uncategorized lines {r.line_share:.1%} -- see analytics.charge_review"


# --- Database is up to date with the current rules --------------------------
# jobs/charge_type.py only categorizes NEW labels; existing rows keep their old category.
# So after a rule change these fail until someone runs `python jobs/charge_type.py --rebuild`
# (try it with --schema sandbox_yourname first).

def test_rule_rows_still_match_current_rules(db):
    df = query(db, """SELECT charge_type, charge_description, major_category, subcategory
                      FROM analytics.charge_mapping WHERE method = 'rule'""")
    df["now"] = [rule_classify_row(t, d) for t, d in zip(df.charge_type, df.charge_description)]
    changed = df[[n != (m, s) for n, m, s in zip(df.now, df.major_category, df.subcategory)]]
    changed = changed.assign(now=[n[1] if n else None for n in changed.now])
    assert changed.empty, f"{len(changed)} labels would get a different category today" + show(
        changed[["charge_type", "charge_description", "subcategory", "now"]])


def test_no_label_left_behind_by_new_rules(db):
    # Labels stored as fallback/uncategorized that a rule now catches.
    df = query(db, """SELECT charge_type, charge_description, method, subcategory
                      FROM analytics.charge_mapping WHERE method <> 'rule'""")
    df["now"] = [rule_classify_row(t, d) for t, d in zip(df.charge_type, df.charge_description)]
    caught = df[df.now.notna()].assign(now=lambda d: [n[1] for n in d.now])
    assert caught.empty, f"{len(caught)} labels a rule would now categorize" + show(
        caught[["charge_type", "charge_description", "method", "subcategory", "now"]])
