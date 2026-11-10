"""Load raw retail transaction data into a PostgreSQL relational warehouse.

This module reads raw retail transaction records (Excel or CSV), standardizes
column naming to a clean relational schema, verifies database connectivity,
and bulk loads the transactions into PostgreSQL using memory-efficient chunking.

Pipeline Context:
    Raw files (data/raw/online_retail.xlsx or transactions.csv)
      └──> load_db.py [THIS MODULE]
             └──> PostgreSQL table: axlero_retail.transactions
                    └──> make_dataset.py (SQL aggregation pipeline)

Typical usage:
    python src/data/load_db.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Explicitly use psycopg2 driver so SQLAlchemy 2.x avoids looking for psycopg3
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/axlero_retail"
)
RAW_DIR = "data/raw"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column headers into lowercase, stripped snake_case format.

    Args:
        df: Input DataFrame with raw, unnormalized column headers.

    Returns:
        pd.DataFrame: DataFrame with sanitized column names (stripped, lowercased, spaces to underscores).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=[" Customer ID ", "Invoice No"])
        >>> normalize_columns(df).columns.tolist()
        ['customer_id', 'invoice_no']
    """
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_raw_file() -> pd.DataFrame:
    """Search for and ingest the raw transactions file from the raw data directory.

    Encoding Rationale:
        The UCI Online Retail dataset originates from a UK-based e-commerce merchant (2010-2011).
        Product descriptions frequently contain British pound (£) currency symbols and accented
        European characters. Using 'ISO-8859-1' (Latin-1) prevents UnicodeDecodeError exceptions
        that occur under strict UTF-8 decoders.

    Returns:
        pd.DataFrame: Ingested raw transaction table.

    Raises:
        FileNotFoundError: If none of the candidate file formats are found in data/raw/.
    """
    candidates = [
        "online_retail.xlsx",
        "Online Retail.xlsx",
        "transactions.csv",
    ]
    for name in candidates:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            print(f"Reading {path}...")
            if name.endswith(".xlsx"):
                return pd.read_excel(path, engine="openpyxl")
            # Use ISO-8859-1 to handle £ symbols and European characters in product descriptions
            return pd.read_csv(path, encoding="ISO-8859-1")

    raise FileNotFoundError(
        f"No raw file found in {RAW_DIR}. "
        "Place 'online_retail.xlsx' or 'transactions.csv' there."
    )


def main() -> None:
    """Execute raw file ingestion, normalization, and bulk database loading."""
    print("=" * 60)
    print("Axlero Project 1 - Raw Data Loader")
    print("=" * 60)

    # Ingest and normalize headers
    df = load_raw_file()
    df = normalize_columns(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Rows: {len(df):,}")

    # Column rename mapping:
    # Resolves variations between raw Excel sheets and CSV exports to match
    # the target relational database schema in PostgreSQL.
    rename_map = {
        "invoiceno": "invoice_no",
        "invoice": "invoice_no",
        "stockcode": "stock_code",
        "customerid": "customer_id",
        "invoicedate": "invoice_date",
        "unitprice": "unit_price",
        "description": "description",
        "quantity": "quantity",
        "country": "country",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    print(f"\nNormalized columns: {list(df.columns)}")

    # Verify PostgreSQL connectivity before attempting upload
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"\n✅ Connected to PostgreSQL: {DATABASE_URL}")
    except SQLAlchemyError as e:
        print(f"\n❌ Database connection failed: {e}")
        print("   Run: brew services start postgresql@16")
        print("   Then: createdb axlero_retail")
        sys.exit(1)

    # Batch insert configuration:
    #   - chunksize=50_000: Limits memory footprint to ~50K rows at a time, preventing OOM on 541K rows
    #   - method='multi': Issues bulk INSERT ... VALUES (...), (...) statements, providing a ~10x speedup
    #     over traditional single-row SQL inserts.
    print("\nUploading to table 'transactions' (chunked)...")
    df.to_sql(
        "transactions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50_000,
        method="multi",
    )

    # Confirm loaded record count
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()

    print(f"\n✅ Loaded {count:,} rows into axlero_retail.transactions")
    print(f"   Retention check: {count:,} / {len(df):,} = {count/len(df)*100:.1f}%")


if __name__ == "__main__":
    main()
