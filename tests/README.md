# Charge Categorization Tests

These tests check that shipping charges are sorted into the right categories, and that the categorized data in Supabase is complete and up to date.

## What each file does

**`test_rules.py`: are the sorting rules written correctly?**
The project sorts charges using a list of keyword rules (e.g. "fuel" → Fuel Surcharge). This file checks the rules have no typos and point to real categories. It also runs the project's own list of 127 example charges (English, Dutch, Spanish) to make sure each one still lands in the right category.

**`test_fallback.py`: is the similarity matching accurate?**
When no rule fits, the project compares the charge's wording with each category's keywords and picks the closest one. A charge only gets a category if the match is close enough; otherwise it stays uncategorized for someone to review. This file checks that these matches are correct at least 95% of the time on 200 hand-checked charges. It also checks that old mistakes don't come back, such as "Weekend Charge" being matched to Chargeback.

**`test_database.py`: does the data in Supabase add up?**
This file looks at the real data. It checks that:
- every charge has a category,
- no charges or dollars went missing,
- uncategorized charges stay under 2% of dollars,
- the stored categories match what today's rules would give.

**`conftest.py`: setup, for all tests.**
Tells the tests where the project is and opens the database connection.
## Why `test_database.py` matters

Nothing else in the project does this check. If new invoices come in and nobody runs the categorization job, the dashboard quietly shows those charges as "Other / Uncategorized", with no warning. This file catches that.

**These tests now run automatically inside `jobs/build_dashboard.py`.** If any check fails, the dashboard is not built, so its numbers are always complete and current.

## How to run

The normal workflow is two commands, from the `TDE_Capstone_DB` folder:

```
python3 jobs/charge_type.py
python3 jobs/build_dashboard.py
```

`build_dashboard.py` runs all 40 checks first. `40 passed` means the dashboard is built. If a check fails, it stops with "Checks failed - dashboard not built", and the messages above name the charges involved and what to do.

Sandbox previews (`--schema sandbox_yourname`) skip the checks, because the checks always look at the shared `analytics` tables.

To run the checks on their own, for example while editing category rules:

```
python3 -m pytest tests/
```

Without the `.env` file, the database tests are skipped, not failed.

## If a database test fails

- **"missing from charge_mapping" or "stale counts":** new data came in. Run `python jobs/charge_type.py`.
- **"would get a different category today":** the rules were changed, but the stored data still has the old categories. Rebuild it, trying your own sandbox first:

  ```
  python jobs/charge_type.py --rebuild --schema sandbox_yourname
  ```

  Tell the team before rebuilding the shared `analytics` tables.
- After fixing, run `python3 jobs/build_dashboard.py` again. It reruns the checks and builds the dashboard once they pass.
