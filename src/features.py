"""Build customer-level RFM features from transaction or summary data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "interim" / "users_raw.parquet"
OUTPUT_PATH = ROOT / "data" / "interim" / "rfm.parquet"


def compute_rfm(
    df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute recency, purchase frequency, and monetary value per customer.

    Accepts either line-item transactions (invoice_date, invoice_no, quantity,
    unit_price) or the customer summary written by ``make_dataset.py``.
    """
    if "customer_id" not in df.columns:
        raise ValueError("Input data must contain a customer_id column.")

    if {"invoice_date", "invoice_no", "quantity", "unit_price"}.issubset(df.columns):
        transactions = df.copy()
        transactions["invoice_date"] = pd.to_datetime(
            transactions["invoice_date"], errors="coerce"
        )
        transactions["quantity"] = pd.to_numeric(
            transactions["quantity"], errors="coerce"
        )
        transactions["unit_price"] = pd.to_numeric(
            transactions["unit_price"], errors="coerce"
        )
        transactions = transactions.dropna(
            subset=["customer_id", "invoice_date", "invoice_no", "quantity", "unit_price"]
        )
        transactions = transactions.loc[
            (transactions["quantity"] > 0) & (transactions["unit_price"] > 0)
        ].copy()
        transactions["line_revenue"] = (
            transactions["quantity"] * transactions["unit_price"]
        )
        grouped = transactions.groupby("customer_id", as_index=False).agg(
            last_purchase=("invoice_date", "max"),
            frequency=("invoice_no", "nunique"),
            monetary=("line_revenue", "sum"),
        )
    elif {"last_purchase", "total_orders", "total_revenue"}.issubset(df.columns):
        grouped = df.loc[
            df["customer_id"].notna(),
            ["customer_id", "last_purchase", "total_orders", "total_revenue"],
        ].copy()
        grouped["last_purchase"] = pd.to_datetime(
            grouped["last_purchase"], errors="coerce"
        )
        grouped["frequency"] = pd.to_numeric(
            grouped["total_orders"], errors="coerce"
        )
        grouped["monetary"] = pd.to_numeric(
            grouped["total_revenue"], errors="coerce"
        )
        grouped = grouped.dropna(
            subset=["last_purchase", "frequency", "monetary"]
        )
        grouped = grouped.drop(columns=["total_orders", "total_revenue"])
    else:
        raise ValueError(
            "Input must have either invoice_date/invoice_no/quantity/unit_price "
            "or last_purchase/total_orders/total_revenue columns."
        )

    if grouped.empty:
        raise ValueError("No valid customer transactions remain after cleaning.")

    latest_purchase = grouped["last_purchase"].max()
    snapshot = (
        pd.Timestamp(snapshot_date)
        if snapshot_date is not None
        else latest_purchase.normalize() + pd.Timedelta(days=1)
    )
    if snapshot.tz is not None:
        snapshot = snapshot.tz_localize(None)
    if snapshot < latest_purchase.normalize():
        raise ValueError("snapshot_date cannot be earlier than the latest purchase.")

    grouped["recency"] = (
        snapshot.normalize() - grouped["last_purchase"].dt.normalize()
    ).dt.days
    rfm = grouped[["customer_id", "recency", "frequency", "monetary"]].copy()
    rfm["frequency"] = rfm["frequency"].astype("int64")
    rfm["monetary"] = rfm["monetary"].astype(float)
    rfm = rfm.sort_values("customer_id").reset_index(drop=True)
    return rfm, snapshot.normalize()


def log1p_transform(
    df: pd.DataFrame,
    columns: tuple[str, ...] = ("recency", "frequency", "monetary"),
) -> pd.DataFrame:
    """Return a copy with selected non-negative columns transformed by log1p."""
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Cannot log-transform missing columns: {missing}")
    transformed = df.copy()
    for column in columns:
        values = pd.to_numeric(transformed[column], errors="raise")
        if (values < 0).any():
            raise ValueError(f"Column {column!r} contains negative values.")
        transformed[column] = np.log1p(values)
    return transformed


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Customer dataset not found: {INPUT_PATH}. Run the data extraction first."
        )
    source = pd.read_parquet(INPUT_PATH)
    rfm, snapshot = compute_rfm(source)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_parquet(OUTPUT_PATH, index=False)
    retained = len(rfm) / len(source) if len(source) else 0.0
    print(
        f"✅ RFM features saved to {OUTPUT_PATH.relative_to(ROOT)} "
        f"({len(rfm):,}/{len(source):,} customers; "
        f"{retained:.1%} retention; snapshot {snapshot.date()})"
    )


if __name__ == "__main__":
    main()
