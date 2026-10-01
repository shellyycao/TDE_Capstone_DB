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

**Run these tests before `jobs/build_dashboard.py`.** That way you know the numbers on the dashboard are complete and current.

## How to run

In the terminal, from the folder that contains `tests/`:

```
python3 -m pytest tests/ -p no:cacheprovider
```

`40 passed` means everything is fine. A failure names the charges involved and what to do.

Without the `.env` file, the database tests are skipped, not failed.

## If a database test fails

- **"missing from charge_mapping" or "stale counts":** new data came in. Run `python jobs/charge_type.py`.
- **"would get a different category today":** the rules were changed, but the stored data still has the old categories. Rebuild it, trying your own sandbox first:

  ```
  python jobs/charge_type.py --rebuild --schema sandbox_yourname
  ```

  Tell the team before rebuilding the shared `analytics` tables.

