"""Load raw retail transaction data into PostgreSQL."""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Use psycopg2 explicitly so SQLAlchemy 2.x doesn't look for psycopg3
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/axlero_retail"
)
RAW_DIR = "data/raw"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_raw_file() -> pd.DataFrame:
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
            return pd.read_csv(path, encoding="ISO-8859-1")
    raise FileNotFoundError(
        f"No raw file found in {RAW_DIR}. "
        "Place 'online_retail.xlsx' or 'transactions.csv' there."
    )


def main():
    print("=" * 60)
    print("Axlero Project 1 - Raw Data Loader")
    print("=" * 60)

    df = load_raw_file()
    df = normalize_columns(df)

    print(f"\nColumns: {list(df.columns)}")
    print(f"Rows: {len(df):,}")

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

    print("\nUploading to table 'transactions' (chunked)...")
    df.to_sql(
        "transactions",
        engine,
        if_exists="replace",
        index=False,
        chunksize=50_000,
        method="multi",
    )

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()

    print(f"\n✅ Loaded {count:,} rows into axlero_retail.transactions")
    print(f"   Retention check: {count:,} / {len(df):,} = {count/len(df)*100:.1f}%")


if __name__ == "__main__":
    main()
