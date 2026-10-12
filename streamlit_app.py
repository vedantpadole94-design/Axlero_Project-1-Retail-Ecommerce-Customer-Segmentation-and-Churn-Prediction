"""Streamlit dashboard for customer segments, revenue, and churn risk."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
BI_EXPORT_PATH = PROCESSED_DIR / "bi_export.csv"
SUMMARY_PATH = PROCESSED_DIR / "segment_summary.csv"
HIGH_RISK_PATH = PROCESSED_DIR / "high_risk_customers.csv"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="Axlero Customer Insights", layout="wide")
    st.title("Axlero | Retail Customer Insights")
    st.caption("RFM customer segments, revenue contribution, and churn risk.")
    missing = [
        path.name
        for path in [BI_EXPORT_PATH, SUMMARY_PATH, HIGH_RISK_PATH]
        if not path.is_file()
    ]
    if missing:
        st.error(
            "Dashboard data is not ready. Run `python -m src.dashboard_data` "
            f"after training the model. Missing: {', '.join(missing)}"
        )
        st.stop()

    bi_export = load_csv(BI_EXPORT_PATH)
    summary = load_csv(SUMMARY_PATH).fillna(
        {"segment": "Unknown", "count": 0, "revenue": 0, "churn_rate": 0}
    )
    high_risk = load_csv(HIGH_RISK_PATH)
    total_customers = int(bi_export["customer_id"].nunique())
    total_revenue = float(bi_export["total_revenue"].sum())
    high_risk_count = int(len(high_risk))
    predicted_churn = float(bi_export["churn_predicted"].mean())

    metric_columns = st.columns(4)
    metric_columns[0].metric("Customers", f"{total_customers:,}")
    metric_columns[1].metric("Revenue", f"£{total_revenue:,.0f}")
    metric_columns[2].metric("Predicted churn", f"{predicted_churn:.1%}")
    metric_columns[3].metric("High-risk customers", f"{high_risk_count:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Customer segments")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.pie(
            summary["count"],
            labels=summary["segment"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.42},
        )
        axis.set_title("Customer mix")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with right:
        st.subheader("Revenue by segment")
        revenue = summary.sort_values("revenue", ascending=False)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.bar(revenue["segment"], revenue["revenue"], color="#3976a8")
        axis.set_ylabel("Revenue")
        axis.tick_params(axis="x", rotation=25)
        axis.set_title("Historical revenue contribution")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    st.subheader("Segment profile and recommended actions")
    profile_columns = [
        name
        for name in [
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
        if name in summary.columns
    ]
    st.dataframe(summary[profile_columns], use_container_width=True, hide_index=True)

    st.subheader("Customers with churn probability above 70%")
    if high_risk.empty:
        st.info("No customers currently exceed the 70% risk threshold.")
    else:
        visible_columns = [
            column
            for column in [
                "customer_id",
                "primary_country",
                "segment",
                "recency",
                "total_revenue",
                "churn_probability",
            ]
            if column in high_risk.columns
        ]
        st.dataframe(
            high_risk[visible_columns].head(100),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing up to 100 of {high_risk_count:,} high-risk customers.")


if __name__ == "__main__":
    main()
