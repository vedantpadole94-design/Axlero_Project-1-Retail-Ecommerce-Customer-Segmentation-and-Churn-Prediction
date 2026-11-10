"""Extract user-level summary from Postgres and save interim dataset.

This module parses modular SQL queries from `src/data/sql_queries.sql`,
executes analytical aggregations against the PostgreSQL warehouse, and persists
intermediate analytical artifacts into compressed Parquet files for downstream
feature engineering and modeling.

Pipeline Context:
    PostgreSQL table (axlero_retail.transactions)
      └──> make_dataset.py [THIS MODULE]
             ├──> data/interim/users_raw.parquet (Customer aggregations)
             ├──> data/interim/monthly_revenue.parquet (Revenue trend)
             └──> data/interim/top_countries.parquet (Geographic distribution)
                    └──> features.py (RFM calculation)

Typical usage:
    python src/data/make_dataset.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/axlero_retail"
)
INTERIM_DIR = "data/interim"


def run_query(engine: Engine, sql: str) -> pd.DataFrame:
    """Execute a parameterized or raw SQL query against PostgreSQL and return a DataFrame.

    Args:
        engine: Active SQLAlchemy Engine instance.
        sql: Plaintext SQL query statement.

    Returns:
        pd.DataFrame: Tabular query results.
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def parse_queries(sql_text: str) -> dict[str, str]:
    """Parse a multi-query SQL script into a dictionary of named queries.

    Design Rationale:
        Using standard '-- name: <query_name>' block comments allows storing
        multiple production SQL statements inside a single version-controlled `.sql`
        file, keeping SQL maintainable and syntax-highlighted in IDEs while allowing
        Python extraction scripts to invoke queries individually.

    Args:
        sql_text: Full plaintext content of a `.sql` file containing '-- name:' headers.

    Returns:
        dict[str, str]: Dictionary mapping query names to their respective SQL bodies.

    Example:
        >>> script = "-- name: test_query\\nSELECT 1;"
        >>> parse_queries(script)
        {'test_query': 'SELECT 1;'}
    """
    # Split on the standardized '-- name:' query boundary
    parts = sql_text.split("-- name:")
    queries: dict[str, str] = {}
    for part in parts[1:]:
        lines = part.strip().split("\n", 1)
        name = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        queries[name] = body
    return queries


def main() -> None:
    """Execute database extraction queries and export interim Parquet datasets."""
    print("=" * 60)
    print("Axlero Project 1 - User-Level Dataset Extraction")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)

    sql_path = "src/data/sql_queries.sql"
    if not os.path.exists(sql_path):
        print(f"ERROR: {sql_path} not found")
        sys.exit(1)

    with open(sql_path, "r", encoding="utf-8") as f:
        raw = f.read()

    queries = parse_queries(raw)
    print(f"\nFound queries: {list(queries.keys())}")

    if "user_level_summary" not in queries:
        print("ERROR: user_level_summary query missing from SQL file")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE 3 DISTINCT ANALYTICAL QUERIES
    # ─────────────────────────────────────────────────────────────────────────
    # Query 1: user_level_summary
    # Aggregates 541K rows to 4,338 unique customers: computes total spend, order count,
    # first/last purchase timestamps, product diversity, and country distribution.
    user_df = run_query(engine, queries["user_level_summary"])
    print(f"\nuser_level_summary: {user_df.shape[0]:,} customers, "
          f"{user_df.shape[1]} columns")

    # Query 2: monthly_revenue_trend
    # Monthly time-series tracking gross transaction revenue and order volumes over time.
    monthly_df = run_query(engine, queries["monthly_revenue_trend"])
    print(f"monthly_revenue_trend: {monthly_df.shape[0]} months")

    # Query 3: top_countries
    # Country-level market penetration metrics (revenue, orders, average spend).
    countries_df = run_query(engine, queries["top_countries"])
    print(f"top_countries: {countries_df.shape[0]} countries")

    os.makedirs(INTERIM_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PARQUET EXPORT RATIONALE
    # ─────────────────────────────────────────────────────────────────────────
    # Parquet provides columnar storage with snappy compression, preserving
    # explicit datatypes (datetime, int64, float64) without string serialization loss,
    # and offers up to 5-10x faster read performance than unindexed CSVs.
    user_df.to_parquet(f"{INTERIM_DIR}/users_raw.parquet", index=False)
    monthly_df.to_parquet(f"{INTERIM_DIR}/monthly_revenue.parquet", index=False)
    countries_df.to_parquet(f"{INTERIM_DIR}/top_countries.parquet", index=False)

    print(f"\nSaved to {INTERIM_DIR}/")
    print(f"   - users_raw.parquet")
    print(f"   - monthly_revenue.parquet")
    print(f"   - top_countries.parquet")

    print(f"\nUser summary stats:")
    print(user_df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
