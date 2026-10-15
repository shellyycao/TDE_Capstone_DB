"""Shared setup for the tests in this folder (pytest loads this file automatically).

Finds the TDE_Capstone_DB project, makes its src/ importable, and provides a
read-only database connection for the tests that need one.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


# Don't leave __pycache__ folders inside the project when importing its code.
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def db():
    """One connection for the whole run, inside a READ ONLY transaction.

    The database itself refuses any INSERT / UPDATE / DELETE in this transaction, so
    these tests cannot change data even by mistake. Skips the database tests when
    SUPABASE_DB_URL isn't set (in TDE_Capstone_DB/.env).
    """
    load_dotenv(REPO_ROOT / ".env")
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        pytest.skip("SUPABASE_DB_URL not set -- database tests skipped")
    url = "postgresql+psycopg2://" + url.split("://", 1)[1]
    with create_engine(url).connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        yield conn
        conn.rollback()


def query(conn, sql, **params):
    return pd.read_sql(text(sql), conn, params=params)
