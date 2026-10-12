"""Extract user-level summary from Postgres and save interim dataset."""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/axlero_retail"
)
INTERIM_DIR = "data/interim"


def run_query(engine, sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def parse_queries(sql_text: str) -> dict:
    parts = sql_text.split("-- name:")
    queries = {}
    for part in parts[1:]:
        lines = part.strip().split("\n", 1)
        name = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        queries[name] = body
    return queries


def main():
    print("=" * 60)
    print("Axlero Project 1 - User-Level Dataset Extraction")
    print("=" * 60)

    engine = create_engine(DATABASE_URL)

    sql_path = "src/data/sql_queries.sql"
    if not os.path.exists(sql_path):
        print(f"ERROR: {sql_path} not found")
        sys.exit(1)

    with open(sql_path, "r") as f:
        raw = f.read()

    queries = parse_queries(raw)
    print(f"\nFound queries: {list(queries.keys())}")

    if "user_level_summary" not in queries:
        print("ERROR: user_level_summary query missing from SQL file")
        sys.exit(1)

    user_df = run_query(engine, queries["user_level_summary"])
    print(f"\nuser_level_summary: {user_df.shape[0]:,} customers, "
          f"{user_df.shape[1]} columns")

    monthly_df = run_query(engine, queries["monthly_revenue_trend"])
    print(f"monthly_revenue_trend: {monthly_df.shape[0]} months")

    countries_df = run_query(engine, queries["top_countries"])
    print(f"top_countries: {countries_df.shape[0]} countries")

    os.makedirs(INTERIM_DIR, exist_ok=True)

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
