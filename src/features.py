"""Build customer-level RFM features from transaction or summary data.

This module computes core Recency, Frequency, and Monetary (RFM) behavioral
metrics for e-commerce customers, forming the analytical backbone for both
unsupervised customer clustering (K-Means) and supervised churn modeling.

Pipeline Context:
    Raw transactions (PostgreSQL / CSV)
      └──> make_dataset.py (summary aggregations)
             └──> features.py (RFM extraction & log scaling) [THIS MODULE]
                    ├──> segmentation.py (K-Means clustering)
                    └──> model.py (churn classification)

Typical usage:
    from src.features import compute_rfm, log1p_transform
    rfm_df, snapshot_date = compute_rfm(source_df)
    rfm_transformed = log1p_transform(rfm_df)
"""

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
    unit_price) or pre-aggregated customer summaries written by ``make_dataset.py``.

    Business Definition of RFM Metrics:
        - Recency (R): Number of days elapsed between a customer's latest order
          and the analysis snapshot date. Lower recency indicates active customers.
        - Frequency (F): Total count of distinct completed orders/invoices placed.
          Higher frequency indicates brand engagement and repeat purchasing.
        - Monetary (M): Total cumulative revenue generated across all completed orders.
          Identifies high-value VIP accounts driving outsized revenue.

    Args:
        df: Input DataFrame containing either raw transaction lines or
            pre-aggregated customer summary records.
        snapshot_date: Reference date for recency calculation. If None, defaults
            to exactly one day after the latest observed transaction date.

    Returns:
        A tuple of:
            - pd.DataFrame: Clean customer RFM table with columns:
                ['customer_id', 'recency', 'frequency', 'monetary'].
            - pd.Timestamp: Normalized snapshot timestamp used as reference.

    Raises:
        ValueError: If 'customer_id' is missing or required columns for both
            raw and summary schemas cannot be found.
        ValueError: If cleaning eliminates all valid rows.
        ValueError: If 'snapshot_date' is set earlier than the latest transaction.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "customer_id": [101, 102],
        ...     "last_purchase": ["2023-01-01", "2023-01-05"],
        ...     "total_orders": [2, 5],
        ...     "total_revenue": [150.0, 420.0],
        ... })
        >>> rfm, snapshot = compute_rfm(data, snapshot_date="2023-01-06")
        >>> rfm[["customer_id", "recency", "frequency", "monetary"]].to_dict(orient="records")
        [{'customer_id': 101, 'recency': 5, 'frequency': 2, 'monetary': 150.0},
         {'customer_id': 102, 'recency': 1, 'frequency': 5, 'monetary': 420.0}]
    """
    if "customer_id" not in df.columns:
        raise ValueError("Input data must contain a customer_id column.")

    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 1: Raw line-item transaction schema
    # ─────────────────────────────────────────────────────────────────────────
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
        # Drop corrupt or missing transaction lines
        transactions = transactions.dropna(
            subset=["customer_id", "invoice_date", "invoice_no", "quantity", "unit_price"]
        )
        # Filter out returns (negative quantity) and price adjustments/vouchers
        transactions = transactions.loc[
            (transactions["quantity"] > 0) & (transactions["unit_price"] > 0)
        ].copy()
        # Compute individual item line revenue
        transactions["line_revenue"] = (
            transactions["quantity"] * transactions["unit_price"]
        )
        # Group into customer-level RFM metrics
        grouped = transactions.groupby("customer_id", as_index=False).agg(
            last_purchase=("invoice_date", "max"),
            frequency=("invoice_no", "nunique"),
            monetary=("line_revenue", "sum"),
        )
    # ─────────────────────────────────────────────────────────────────────────
    # BRANCH 2: Pre-aggregated customer summary schema (from make_dataset.py)
    # ─────────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────────
    # SNAPSHOT DATE & RECENCY CALCULATION
    # ─────────────────────────────────────────────────────────────────────────
    latest_purchase = grouped["last_purchase"].max()

    # Default snapshot is 1 day after the latest purchase in the entire dataset.
    # This ensures that even the most recent purchaser has recency >= 1 day,
    # preventing recency = 0 which would cause division by zero or log(0) issues.
    snapshot = (
        pd.Timestamp(snapshot_date)
        if snapshot_date is not None
        else latest_purchase.normalize() + pd.Timedelta(days=1)
    )
    if snapshot.tz is not None:
        snapshot = snapshot.tz_localize(None)
    if snapshot < latest_purchase.normalize():
        raise ValueError("snapshot_date cannot be earlier than the latest purchase.")

    # Calculate days elapsed since last purchase
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
    """Return a copy with selected non-negative columns transformed by log1p.

    Why log1p instead of standard log:
        - Customer monetary spend and frequency distributions are heavily right-skewed
          with long power-law tails (a few VIP customers spend orders of magnitude more).
        - np.log1p(x) calculates ln(1 + x). When monetary spend or frequency is zero or small,
          log1p maps cleanly to ln(1) = 0 without producing -inf errors.
        - Log transformation stabilizes variance and compresses outliers, which is
          mandatory before feeding metrics into distance-based algorithms like K-Means.

    Args:
        df: Input DataFrame containing numeric feature columns.
        columns: Tuple of column names to transform. Defaults to
            ('recency', 'frequency', 'monetary').

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns transformed.

    Raises:
        ValueError: If any requested column is missing from the DataFrame.
        ValueError: If any column contains negative values (logarithm undefined).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"recency": [0, 9], "frequency": [1, 10]})
        >>> logged = log1p_transform(df, columns=("recency", "frequency"))
        >>> round(logged.loc[0, "recency"], 4)
        0.0
    """
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise ValueError(f"Cannot log-transform missing columns: {missing}")
    transformed = df.copy()
    for column in columns:
        values = pd.to_numeric(transformed[column], errors="raise")
        if (values < 0).any():
            raise ValueError(f"Column {column!r} contains negative values.")
        # log1p(x) = log(1 + x), safe for zero values
        transformed[column] = np.log1p(values)
    return transformed


# Alias for backward compatibility and alternative naming conventions
add_log_transform = log1p_transform


def main() -> None:
    """Execute the end-to-end RFM feature generation pipeline."""
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
