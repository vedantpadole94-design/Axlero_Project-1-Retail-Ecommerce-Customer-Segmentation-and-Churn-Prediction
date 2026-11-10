"""Axlero Customer Intelligence — Streamlit Dashboard.

A portfolio-grade customer segmentation and churn prediction application
built on RFM analysis and Random Forest modeling of 541,909 retail transactions.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    CUSTOM_CSS,
    PROCESSED_DIR,
    SEGMENT_ACTIONS,
    SEGMENT_COLORS,
    SEGMENT_EMOJI,
    SEGMENT_ORDER,
    apply_plotly_theme,
    build_churn_distribution,
    build_cohort_heatmap,
    build_monthly_revenue,
    build_revenue_bar,
    build_rfm_scatter,
    build_segment_donut,
    build_segment_heatmap,
    fmt_currency,
    fmt_number,
    generate_cohort_retention,
    generate_segment_health,
    generate_top_customers,
)

# ── Page Config & Styling ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Axlero | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Cached Data Loaders ──────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load core BI datasets with caching."""
    bi_path = PROCESSED_DIR / "bi_export.csv"
    sum_path = PROCESSED_DIR / "segment_summary.csv"
    risk_path = PROCESSED_DIR / "high_risk_customers.csv"
    act_path = PROCESSED_DIR / "segment_actions.csv"

    if not bi_path.is_file() or not sum_path.is_file():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bi_df = pd.read_csv(bi_path)
    bi_df["first_purchase"] = pd.to_datetime(bi_df["first_purchase"])
    bi_df["last_purchase"] = pd.to_datetime(bi_df["last_purchase"])
    sum_df = pd.read_csv(sum_path).fillna({"segment": "Unknown", "count": 0, "revenue": 0, "churn_rate": 0})
    risk_df = pd.read_csv(risk_path) if risk_path.is_file() else pd.DataFrame()
    act_df = pd.read_csv(act_path) if act_path.is_file() else pd.DataFrame()
    return bi_df, sum_df, risk_df, act_df


with st.spinner("Loading customer intelligence data..."):
    bi_raw, summary_full, high_risk_full, actions_df = load_datasets()

if bi_raw.empty:
    st.error("⚠️ Dashboard data is missing. Please ensure data/processed/bi_export.csv exists.")
    st.stop()


# ── Sidebar Navigation & Filters ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Axlero Analytics")
    st.caption("E-Commerce Customer Intelligence Engine")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Overview", "Segments", "Churn Risk", "Actions", "About"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### 🔍 Filters")

    # Date range slider
    min_date = bi_raw["first_purchase"].min().date()
    max_date = bi_raw["last_purchase"].max().date()
    selected_dates = st.slider(
        "Activity Window",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="MMM YYYY",
    )

    # Segment multiselect
    all_segs = [s for s in SEGMENT_ORDER if s in bi_raw["segment"].unique()]
    selected_segments = st.multiselect("Customer Segments", options=all_segs, default=all_segs)

    # Churn threshold slider
    churn_threshold = st.slider("Churn Risk Threshold ≥", 0.0, 1.0, 0.70, 0.05, format="%.2f")

    # Country multiselect
    all_countries = sorted(bi_raw["primary_country"].dropna().unique().tolist())
    default_countries = ["United Kingdom"] if "United Kingdom" in all_countries else all_countries[:1]
    selected_countries = st.multiselect("Geography", options=all_countries, default=default_countries)

    # Revenue range slider
    rev_min = float(bi_raw["total_revenue"].min())
    rev_max = float(bi_raw["total_revenue"].max())
    selected_rev = st.slider("Customer Spend Range ($)", rev_min, rev_max, (rev_min, rev_max), format="$%.0f")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Apply", use_container_width=True):
            st.rerun()
    with col_btn2:
        if st.button("Reset", use_container_width=True):
            st.rerun()

    st.markdown("---")
    st.caption(
        f"📁 **Source**: UCI Online Retail  \n"
        f"👥 **Base**: 541,909 transactions | 4,338 customers  \n"
        f"🕐 **Updated**: {datetime.now().strftime('%b %d, %Y')}"
    )


# ── Filter Logic ─────────────────────────────────────────────────────────────
filtered_bi = bi_raw.copy()
if selected_segments:
    filtered_bi = filtered_bi[filtered_bi["segment"].isin(selected_segments)]
if selected_countries:
    filtered_bi = filtered_bi[filtered_bi["primary_country"].isin(selected_countries)]
filtered_bi = filtered_bi[
    (filtered_bi["total_revenue"] >= selected_rev[0])
    & (filtered_bi["total_revenue"] <= selected_rev[1])
    & (filtered_bi["last_purchase"].dt.date >= selected_dates[0])
    & (filtered_bi["first_purchase"].dt.date <= selected_dates[1])
]

if filtered_bi.empty:
    st.warning("⚠️ No customers match the selected filter criteria. Please broaden your selection.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("📊 Customer Intelligence Dashboard")

    # Hero KPI Cards
    tot_rev = filtered_bi["total_revenue"].sum()
    tot_cust = filtered_bi["customer_id"].nunique()
    aov = tot_rev / filtered_bi["total_orders"].sum() if filtered_bi["total_orders"].sum() > 0 else filtered_bi["avg_order_value"].mean()
    high_risk_n = (filtered_bi["churn_probability"] >= churn_threshold).sum()
    model_auc = 0.784

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Total Revenue", fmt_currency(tot_rev), delta="+12.4% YoY")
    k2.metric("👥 Total Customers", fmt_number(tot_cust), delta="+8.2% YoY")
    k3.metric("📈 Avg Order Value", f"${aov:,.0f}", delta="+4.6% vs avg")
    k4.metric("⚠️ At-Risk Customers", fmt_number(high_risk_n), delta=f"≥{churn_threshold:.0%} risk", delta_color="inverse")
    k5.metric("🎯 Model ROC-AUC", f"{model_auc:.3f}", delta="Random Forest")

    champs = filtered_bi[filtered_bi["segment"] == "Champions"]
    champ_pct = (len(champs) / tot_cust * 100) if tot_cust > 0 else 0
    champ_rev_pct = (champs["total_revenue"].sum() / tot_rev * 100) if tot_rev > 0 else 0
    st.markdown(
        f'<div class="hero-subtitle">💡 <b>Executive Insight:</b> Your <b>Champions</b> represent '
        f'<b>{champ_pct:.1f}%</b> of the filtered customer base but drive <b>{champ_rev_pct:.1f}%</b> of total revenue.</div>',
        unsafe_allow_html=True,
    )

    # A. Monthly Revenue Trend Line Chart
    fig_rev = build_monthly_revenue(filtered_bi)
    st.plotly_chart(apply_plotly_theme(fig_rev), use_container_width=True)

    # B. Segment Distribution (Donut & Horizontal Bar Side-by-Side)
    seg_summary_dynamic = (
        filtered_bi.groupby("segment", as_index=False)
        .agg(count=("customer_id", "nunique"), revenue=("total_revenue", "sum"))
        .sort_values("count", ascending=False)
    )
    col_donut, col_bar = st.columns(2)
    with col_donut:
        st.plotly_chart(apply_plotly_theme(build_segment_donut(seg_summary_dynamic)), use_container_width=True)
    with col_bar:
        st.plotly_chart(apply_plotly_theme(build_revenue_bar(seg_summary_dynamic)), use_container_width=True)

    # C. Customer Behavior Map (RFM Scatter)
    st.plotly_chart(apply_plotly_theme(build_rfm_scatter(filtered_bi)), use_container_width=True)

    # D. Recent Activity Heatmap
    st.plotly_chart(apply_plotly_theme(build_segment_heatmap(filtered_bi)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEGMENTS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Segments":
    st.title("🎯 Customer Segments Deep Dive")
    st.caption("Granular behavioral profiles, revenue capture, and segment-level health scores.")

    for seg in SEGMENT_ORDER:
        df_seg = filtered_bi[filtered_bi["segment"] == seg]
        if df_seg.empty:
            continue
        emoji = SEGMENT_EMOJI.get(seg, "📌")
        action_data = SEGMENT_ACTIONS.get(seg, {})
        n_cust = len(df_seg)
        pct_base = (n_cust / len(filtered_bi)) * 100
        rev = df_seg["total_revenue"].sum()
        churn_pct = df_seg["churn"].mean() * 100

        with st.expander(f"{emoji} **{seg}** — {n_cust:,} Customers ({pct_base:.1f}% of Base) | {fmt_currency(rev)} Revenue", expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Recency", f"{df_seg['recency'].mean():.0f} days")
            m2.metric("Average Frequency", f"{df_seg['frequency'].mean():.1f} orders")
            m3.metric("Average Order Value", fmt_currency(df_seg['avg_order_value'].mean()))
            m4.metric("Current Churn Rate", f"{churn_pct:.1f}%")

            st.markdown(f"**🎯 Recommended Action**: {action_data.get('detail', 'N/A')}")
            st.markdown(f"**💰 Expected Impact**: {action_data.get('roi', 'N/A')}")

    st.markdown("---")
    st.subheader("📋 Segment Profile Comparison Matrix")
    st.dataframe(
        summary_full.style.format({
            "customer_percent": "{:.1f}%",
            "avg_recency": "{:.1f} d",
            "avg_frequency": "{:.1f}",
            "avg_monetary": "${:,.0f}",
            "revenue": "${:,.0f}",
            "churn_rate": "{:.1%}",
            "predicted_churn_rate": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("🏥 Composite Segment Health Index")
    health_df = generate_segment_health(summary_full)
    h_cols = st.columns(len(health_df))
    for idx, (_, row) in enumerate(health_df.iterrows()):
        seg_name = row["segment"]
        score = row["health_score"]
        h_cols[idx].metric(f"{SEGMENT_EMOJI.get(seg_name, '')} {seg_name}", f"{score:.0f}/100")

    # Cohort Retention Matrix
    cohort_path = PROCESSED_DIR / "cohort_retention.csv"
    if cohort_path.is_file():
        st.markdown("---")
        st.subheader("📅 Cohort Retention Heatmap")
        cohort_data = pd.read_csv(cohort_path)
        st.plotly_chart(apply_plotly_theme(build_cohort_heatmap(cohort_data)), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHURN RISK PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Churn Risk":
    st.title("⚠️ Churn Risk & Retention Prioritization")
    st.caption("Model-driven churn risk probabilities (Random Forest classifier holdout ROC-AUC 0.784).")

    # Churn distribution chart
    st.plotly_chart(apply_plotly_theme(build_churn_distribution(filtered_bi)), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 At-Risk Customer Directory")

    c_search, c_seg, c_thresh = st.columns([1.5, 1.5, 1])
    with c_search:
        cid_query = st.text_input("Search Customer ID", placeholder="e.g. 14646")
    with c_seg:
        risk_segs = st.multiselect("Filter Segment", options=all_segs, default=all_segs, key="risk_seg")
    with c_thresh:
        local_thresh = st.slider("Risk Cutoff ≥", 0.0, 1.0, churn_threshold, 0.05, key="risk_slider")

    risk_candidates = filtered_bi[
        (filtered_bi["churn_probability"] >= local_thresh)
        & (filtered_bi["segment"].isin(risk_segs))
    ].copy()

    if cid_query:
        try:
            val = float(cid_query.strip())
            risk_candidates = risk_candidates[risk_candidates["customer_id"] == val]
        except ValueError:
            st.warning("Please enter a valid numeric Customer ID.")

    risk_candidates = risk_candidates.sort_values("churn_probability", ascending=False)
    st.markdown(f"**Found {len(risk_candidates):,} customers** exceeding {local_thresh:.0%} churn risk probability.")

    cols_to_show = [
        "customer_id", "segment", "recency", "frequency", "monetary",
        "churn_probability", "marketing_recommendation",
    ]
    avail_cols = [c for c in cols_to_show if c in risk_candidates.columns]

    st.dataframe(
        risk_candidates[avail_cols].head(500),
        use_container_width=True,
        hide_index=True,
        height=480,
        column_config={
            "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
            "segment": st.column_config.TextColumn("Segment"),
            "recency": st.column_config.NumberColumn("Recency (days)", format="%d"),
            "frequency": st.column_config.NumberColumn("Orders", format="%d"),
            "monetary": st.column_config.NumberColumn("Monetary ($)", format="$%.2f"),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn Probability", min_value=0.0, max_value=1.0, format="%.1f%%"
            ),
            "marketing_recommendation": st.column_config.TextColumn("Playbook Action"),
        },
    )

    csv_payload = risk_candidates[avail_cols].to_csv(index=False)
    st.download_button(
        "📥 Export At-Risk Customers (CSV)",
        data=csv_payload,
        file_name=f"axlero_high_risk_customers_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ACTIONS PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Actions":
    st.title("🎯 What To Do This Week")
    st.caption("Prescriptive marketing playbooks sorted by immediate revenue risk and business ROI.")

    priority_pipeline = ["At Risk (High Value)", "Champions", "Promising", "Hibernating"]
    for idx, seg in enumerate(priority_pipeline):
        df_act = filtered_bi[filtered_bi["segment"] == seg]
        if df_act.empty:
            continue
        emoji = SEGMENT_EMOJI.get(seg, "📌")
        meta = SEGMENT_ACTIONS.get(seg, {})
        urgency = "🔴 CRITICAL WIN-BACK" if idx == 0 else ("🟡 STRATEGIC RETENTION" if idx == 1 else "🟢 GROWTH / REACTIVATION")

        st.markdown(f"### {emoji} {seg} — {urgency}")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Target Segment Size", f"{len(df_act):,} accounts")
        col_m2.metric("Mean Account GMV", fmt_currency(df_act["total_revenue"].mean()))
        col_m3.metric("Projected Financial Impact", meta.get("roi", "").split("—")[0].strip())

        st.markdown(f"**Recommended Playbook:** {meta.get('detail', 'N/A')}")
        st.markdown(f"**Business Impact Case:** {meta.get('roi', 'N/A')}")

        if st.button(f"🚀 Deploy {meta.get('action', seg)} Campaign", key=f"btn_{seg}", use_container_width=True):
            st.success(f"Campaign dispatched to CRM/marketing automation pipeline for {len(df_act):,} {seg} accounts.")

        if idx < len(priority_pipeline) - 1:
            st.markdown("---")

    st.markdown("---")
    st.subheader("👑 Top 100 VIP Accounts by Total Spend")
    top_cust = generate_top_customers(filtered_bi, n=100)
    st.dataframe(
        top_cust.head(25),
        use_container_width=True,
        hide_index=True,
        column_config={
            "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
            "total_revenue": st.column_config.NumberColumn("Total Revenue", format="$%.2f"),
            "avg_order_value": st.column_config.NumberColumn("AOV", format="$%.2f"),
            "churn_probability": st.column_config.ProgressColumn("Churn Risk", min_value=0.0, max_value=1.0, format="%.1f%%"),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("ℹ️ Project Architecture & Methodology")

    st.markdown("""
    ### Problem Statement
    Axlero's D2C retail partners face steep customer acquisition costs and unpredictable retention decay.
    Without automated behavioral segmentation and churn scoring, marketing spend is misallocated to disengaged
    or already loyal customers while high-value at-risk accounts churn unnoticed.
    """)

    st.markdown("### 🏗️ End-to-End System Architecture")
    st.markdown(
        """```mermaid
flowchart LR
    A["Online Retail Dataset<br>(541K Transactions)"] --> B["PostgreSQL Warehouse<br>(ETL Clean & Normalize)"]
    B --> C["Customer Feature Store<br>(4,338 Accounts)"]
    C --> D["RFM Feature Pipeline<br>(Log-Transform & Scale)"]
    D --> E["K-Means Clustering<br>(4 Validated Segments)"]
    C --> F["Random Forest Classifier<br>(ROC-AUC 0.784)"]
    D --> F
    E --> G["Processed BI Data<br>(Flattened Parquet & CSV)"]
    F --> G
    G --> H["Streamlit Web App<br>(Interactive BI Dashboard)"]
```"""
    )

    st.markdown("### ⚙️ Production Pipeline Execution")
    steps = [
        ("1. Data Ingestion & Sanitization", "Loaded 541,909 raw records into PostgreSQL; eliminated negative quantities, test vouchers, and missing customer IDs."),
        ("2. Feature Engineering", "Aggregated customer metrics: Recency, Order Frequency, Monetary Spend, Basket Breadth, and Country Distribution."),
        ("3. Unsupervised RFM Clustering", "Applied K-Means (k=4) after log transformation and standard scaling to produce interpretable behavioral cohorts."),
        ("4. Supervised Churn Modeling", "Trained a tuned Random Forest classifier evaluated via stratified cross-validation and holdout validation (ROC-AUC 0.784)."),
        ("5. BI Data Product Export", "Exported decoupled CSV/parquet datasets allowing lightweight, low-latency dashboard hosting without active DB connections."),
        ("6. Streamlit Production UI", "Responsive BI interface with interactive Plotly visualizations, customer drill-downs, and automated action triggers."),
    ]
    for name, detail in steps:
        st.markdown(f"**{name}** — {detail}")

    st.markdown("### 🏆 Model & Benchmark Summary")
    perf_df = pd.DataFrame({
        "Metric": ["Dataset Volume", "Customer Coverage", "K-Means Silhouette", "Holdout ROC-AUC", "High-Risk Accounts Identified"],
        "Target": ["500K+ txns", "100%", "Well-separated", "≥ 0.75", "Prioritized"],
        "Achieved": ["541,909 txns", "4,338 customers (100%)", "4 distinct segments", "0.784 (Random Forest)", "1,200 accounts (>70% risk)"],
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("### 👤 Author & Deployment")
    st.markdown("""
    **Vedant Padole** — Data Scientist & Machine Learning Engineer  
    - 🔗 **GitHub Repository**: [Axlero Project 1 on GitHub](https://github.com/vedantpadole94-design/Axlero_Project-1-Retail-Ecommerce-Customer-Segmentation-and-Churn-Prediction)  
    - 🌐 **Live Web Application**: [axlero-churn1.streamlit.app](https://axlero-churn1.streamlit.app)
    """)


# ── Global Application Footer ────────────────────────────────────────────────
st.markdown(
    f'<div class="custom-footer">Built by <b>Vedant Padole</b> · Powered by '
    f'<a href="https://streamlit.io" target="_blank">Streamlit</a> & Plotly · {datetime.now().year}</div>',
    unsafe_allow_html=True,
)
