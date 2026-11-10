"""Prepare flat Business Intelligence exports for Streamlit, Power BI, and Tableau.

This module consolidates customer attributes, RFM segment classifications,
and machine learning churn risk predictions into denormalized, lightweight CSV
data products stored in `data/processed/`.

Pipeline Context:
    model.py (Scored probabilities) & segmentation.py (Segment assignments)
      └──> dashboard_data.py [THIS MODULE]
             ├──> data/processed/bi_export.csv (Master 4,338 customer flat file)
             ├──> data/processed/segment_summary.csv (Aggregated segment profiles)
             ├──> data/processed/segment_actions.csv (Strategic playbook rules)
             └──> data/processed/high_risk_customers.csv (Prioritized churn list)
                    └──> app.py (Streamlit Web Dashboard)

Typical usage:
    from src.dashboard_data import generate_exports
    exported_files = generate_exports()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CUSTOMERS_PATH = ROOT / "data" / "interim" / "users_raw.parquet"
SEGMENTS_PATH = PROCESSED_DIR / "rfm_segments.parquet"
SCORED_PATH = PROCESSED_DIR / "customers_scored.parquet"
ACTION_COLUMNS = ["segment", "marketing_recommendation"]

# Canonical customer lifecycle sequence
SEGMENT_ORDER = [
    "Champions",
    "Promising",
    "At Risk (High Value)",
    "Hibernating",
]

# Actionable marketing playbooks associated with each customer segment
SEGMENT_ACTIONS = {
    "Champions": "Reward loyalty with early access, VIP service, and referral incentives.",
    "Promising": "Use a low-friction second-purchase offer and personalized product recommendations.",
    "At Risk (High Value)": "Trigger personal win-back outreach with a tailored incentive before lapse.",
    "Hibernating": "Use low-cost reactivation campaigns; suppress or reduce spend if unresponsive.",
}


def generate_exports() -> dict[str, Path]:
    """Build four BI-ready CSV files from the generated parquet datasets.

    Data Processing Steps:
        1. Validates presence of intermediate Parquet outputs from upstream pipelines.
        2. Merges user features, RFM clusters, and churn risk scores into a unified schema.
        3. Computes customer-level marketing playbooks.
        4. Calculates segment-level executive KPIs (headcount, revenue share, churn rate).
        5. Filters high-risk accounts (churn probability > 70%) for marketing action.

    Returns:
        dict[str, Path]: Dictionary mapping export identifiers to generated file paths.

    Raises:
        FileNotFoundError: If any of the required parquet source files are missing.
        RuntimeError: If inner joins result in customer attrition (must retain 100% of accounts).

    Example:
        >>> files = generate_exports()
        >>> 'bi_export' in files
        True
    """
    missing = [
        path
        for path in [CUSTOMERS_PATH, SEGMENTS_PATH, SCORED_PATH]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required pipeline outputs are missing: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        )

    customers = pd.read_parquet(CUSTOMERS_PATH)
    segments = pd.read_parquet(SEGMENTS_PATH)
    scored = pd.read_parquet(SCORED_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    # MERGE INTO MASTER DENORMALIZED BI EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    # Merge customer features with segment cluster labels
    bi_export = (
        customers.merge(
            segments[["customer_id", "recency", "frequency", "monetary", "segment"]],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            scored[
                ["customer_id", "churn", "churn_probability", "churn_predicted"]
            ],
            on="customer_id",
            how="inner",
            validate="one_to_one",
        )
    )

    # Integrity verification: Verify 100% customer retention across all joins
    if len(bi_export) != len(customers):
        raise RuntimeError(
            f"BI export retained {len(bi_export):,} of {len(customers):,} customers."
        )

    # Attach marketing recommendations mapped to segment
    bi_export["marketing_recommendation"] = bi_export["segment"].map(SEGMENT_ACTIONS)

    # ─────────────────────────────────────────────────────────────────────────
    # AGGREGATE SEGMENT SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    # Compute profile averages, total revenue, and historical vs predicted churn rates
    summary = (
        bi_export.groupby("segment", as_index=False)
        .agg(
            count=("customer_id", "nunique"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            revenue=("monetary", "sum"),
            churn_rate=("churn", "mean"),
            predicted_churn_rate=("churn_predicted", "mean"),
        )
    )
    summary["customer_percent"] = summary["count"] / len(bi_export) * 100
    summary = summary[
        [
            "segment",
            "count",
            "customer_percent",
            "avg_recency",
            "avg_frequency",
            "avg_monetary",
            "revenue",
            "churn_rate",
            "predicted_churn_rate",
        ]
    ].set_index("segment").reindex(SEGMENT_ORDER).reset_index()

    # Create standalone action lookup table
    actions = pd.DataFrame(
        {
            "segment": SEGMENT_ORDER,
            "marketing_recommendation": [
                SEGMENT_ACTIONS[segment] for segment in SEGMENT_ORDER
            ],
        }
    )

    # Filter critical high-risk accounts (> 70% model churn probability)
    # 70% probability cutoff prioritizes accounts with imminent lapse risk
    high_risk = bi_export.loc[
        bi_export["churn_probability"] > 0.7
    ].sort_values("churn_probability", ascending=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "bi_export": PROCESSED_DIR / "bi_export.csv",
        "segment_summary": PROCESSED_DIR / "segment_summary.csv",
        "segment_actions": PROCESSED_DIR / "segment_actions.csv",
        "high_risk_customers": PROCESSED_DIR / "high_risk_customers.csv",
    }

    # Persist all 4 clean CSV deliverables
    bi_export.to_csv(files["bi_export"], index=False)
    summary.to_csv(files["segment_summary"], index=False)
    actions.to_csv(files["segment_actions"], index=False)
    high_risk.to_csv(files["high_risk_customers"], index=False)
    return files


def main() -> None:
    """Execute the BI dataset generation pipeline and display file status."""
    for name, path in generate_exports().items():
        print(f"✅ {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
