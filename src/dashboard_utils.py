"""Dashboard utility helpers for the Axlero Streamlit app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Constants ────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")

SEGMENT_ORDER = ["Champions", "Promising", "At Risk (High Value)", "Hibernating"]

SEGMENT_COLORS = {
    "Champions": "#10B981",
    "Promising": "#3B82F6",
    "At Risk (High Value)": "#F59E0B",
    "Hibernating": "#EF4444",
}

SEGMENT_EMOJI = {
    "Champions": "🏆",
    "Promising": "🌱",
    "At Risk (High Value)": "⚠️",
    "Hibernating": "😴",
}

SEGMENT_ACTIONS = {
    "Champions": {
        "action": "VIP Loyalty & Referral",
        "detail": "Reward top accounts with early access, VIP service tiers, and referral incentives to protect their loyalty.",
        "roi": "Protect ~$5.79M in revenue — this cohort drives 64% of total GMV.",
    },
    "Promising": {
        "action": "Second-Purchase Nurture",
        "detail": "Trigger automated onboarding flows with a personalized second-purchase offer within 30 days.",
        "roi": "Convert 15% more to repeat buyers — upside of ~$69K per cohort.",
    },
    "At Risk (High Value)": {
        "action": "High-Touch Win-Back",
        "detail": "Assign account managers and trigger a personalized discount before lapse.",
        "roi": "Retain 10% of ~$2.11M at-risk revenue — ~$211K protected per quarter.",
    },
    "Hibernating": {
        "action": "Low-Cost Reactivation",
        "detail": "Use minimal-cost seasonal reactivation; suppress from paid ads if unresponsive.",
        "roi": "Reduce ad waste — reallocate ~$40K per quarter to higher-ROI segments.",
    },
}

# ── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main {
    background-color: #F9FAFB;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid #E5E7EB;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
[data-testid="stMetricLabel"] {
    color: #6B7280;
    font-weight: 500;
}
[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 700;
}
h1 {
    color: #111827;
    font-weight: 700;
    letter-spacing: -0.02em;
}
h2, h3 {
    color: #374151;
    font-weight: 600;
}
.hero-subtitle {
    background: linear-gradient(90deg, #EEF2FF 0%, #F0F9FF 100%);
    padding: 16px 20px;
    border-radius: 10px;
    color: #1F2937;
    border-left: 4px solid #6366F1;
    margin: 16px 0;
    font-size: 15px;
}
.custom-footer {
    text-align: center;
    padding: 24px 0;
    color: #9CA3AF;
    font-size: 13px;
    border-top: 1px solid #E5E7EB;
    margin-top: 40px;
}
.stExpander {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background: white;
}
</style>
"""

# ── Formatters ───────────────────────────────────────────────────────────────
def fmt_currency(value: float) -> str:
    """Format number as compact currency."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def fmt_number(value: float) -> str:
    """Format large numbers with K/M suffixes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


# ── Plotly Theme ─────────────────────────────────────────────────────────────
def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent professional styling to a Plotly figure."""
    fig.update_layout(
        font_family="Inter, sans-serif",
        font_color="#374151",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#E5E7EB", borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F3F4F6", zeroline=False)
    return fig


# ── Chart Builders ───────────────────────────────────────────────────────────
def build_monthly_revenue(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly revenue + active customers."""
    if "first_purchase" not in df.columns or df.empty:
        return go.Figure()

    monthly = (
        df.assign(month=df["first_purchase"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(revenue=("total_revenue", "sum"), active_customers=("customer_id", "nunique"))
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["revenue"],
        mode="lines+markers", name="Revenue",
        line=dict(color="#6366F1", width=3),
        fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=monthly["month"], y=monthly["active_customers"],
        mode="lines+markers", name="Active Customers",
        line=dict(color="#10B981", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>%{x|%b %Y}</b><br>Active: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="📈 Monthly Revenue & Active Customers",
        yaxis=dict(title="Revenue ($)", tickformat="$,.0f"),
        yaxis2=dict(title="Active Customers", overlaying="y", side="right"),
        hovermode="x unified",
    )
    return fig


def build_segment_donut(seg_summary: pd.DataFrame) -> go.Figure:
    """Donut chart of customer count per segment."""
    colors = [SEGMENT_COLORS.get(s, "#9CA3AF") for s in seg_summary["segment"]]
    fig = px.pie(
        seg_summary, values="count", names="segment",
        hole=0.6, color="segment",
        color_discrete_map=SEGMENT_COLORS,
        title="👥 Customer Distribution",
    )
    fig.update_traces(
        textposition="outside", textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} customers<br>%{percent}<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def build_revenue_bar(seg_summary: pd.DataFrame) -> go.Figure:
    """Horizontal bar of revenue per segment."""
    sorted_df = seg_summary.sort_values("revenue", ascending=True)
    colors = [SEGMENT_COLORS.get(s, "#9CA3AF") for s in sorted_df["segment"]]
    fig = go.Figure(go.Bar(
        x=sorted_df["revenue"], y=sorted_df["segment"],
        orientation="h", marker_color=colors,
        text=sorted_df["revenue"].apply(fmt_currency), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="💰 Revenue by Segment",
        xaxis=dict(title="Revenue ($)", tickformat="$,.0f"),
        yaxis=dict(title=""),
    )
    return fig


def build_rfm_scatter(df: pd.DataFrame) -> go.Figure:
    """3D scatter: Recency vs Frequency vs Monetary, colored by segment."""
    sample = df.sample(min(1500, len(df)), random_state=42) if len(df) > 1500 else df
    fig = px.scatter_3d(
        sample, x="recency", y="frequency", z="monetary",
        color="segment", color_discrete_map=SEGMENT_COLORS,
        hover_data={"customer_id": True, "primary_country": True},
        title="🗺️ Customer Behavior Map (RFM Space)",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        scene=dict(
            xaxis_title="Recency (days)",
            yaxis_title="Frequency (orders)",
            zaxis_title="Monetary ($)",
        ),
        height=600,
    )
    return fig


def build_segment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of segment activity by month."""
    if "first_purchase" not in df.columns or df.empty:
        return go.Figure()

    hm = (
        df.assign(month=df["first_purchase"].dt.to_period("M").dt.strftime("%Y-%m"))
        .groupby(["segment", "month"], as_index=False)
        .agg(customers=("customer_id", "nunique"))
    )

    pivot = hm.pivot(index="segment", columns="month", values="customers").fillna(0)
    pivot = pivot.reindex([s for s in SEGMENT_ORDER if s in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="Purples",
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:,.0f} customers<extra></extra>",
        colorbar=dict(title="Customers"),
    ))
    fig.update_layout(
        title="📅 Segment Activity by Month",
        xaxis=dict(title="Month"),
        yaxis=dict(title=""),
        height=400,
    )
    return fig


def build_churn_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of churn probability distribution."""
    fig = px.histogram(
        df, x="churn_probability", color="segment",
        color_discrete_map=SEGMENT_COLORS,
        nbins=30, title="⚠️ Churn Probability Distribution",
        labels={"churn_probability": "Predicted Churn Probability"},
        opacity=0.85,
    )
    fig.update_layout(barmode="overlay", yaxis_title="Customer Count")
    return fig


def build_cohort_heatmap(cohort_df: pd.DataFrame) -> go.Figure:
    """Retention heatmap from cohort_retention.csv."""
    if cohort_df.empty:
        return go.Figure()

    cohort_df = cohort_df.set_index("cohort_month")
    z = cohort_df.values * 100

    fig = go.Figure(go.Heatmap(
        z=z,
        x=cohort_df.columns.tolist(),
        y=cohort_df.index.tolist(),
        colorscale="Blues",
        hovertemplate="Cohort: %{y}<br>%{x}<br>Retention: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Retention %"),
    ))
    fig.update_layout(
        title="📊 Cohort Retention Matrix",
        xaxis=dict(title="Months Since First Purchase"),
        yaxis=dict(title="Signup Cohort", autorange="reversed"),
        height=500,
    )
    return fig


# ── Data Generators ──────────────────────────────────────────────────────────
def generate_segment_health(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a 0-100 composite health score per segment."""
    if summary_df.empty:
        return pd.DataFrame(columns=["segment", "health_score"])

    df = summary_df.copy()
    rev_norm = df["revenue"] / df["revenue"].max() if df["revenue"].max() > 0 else 0
    ret_norm = 1 - df["churn_rate"]
    score = (rev_norm * 0.5 + ret_norm * 0.5) * 100

    return pd.DataFrame({"segment": df["segment"], "health_score": score.round(1)})


def generate_top_customers(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Top N customers by total revenue."""
    cols = ["customer_id", "segment", "recency", "frequency", "monetary", "total_revenue",
            "avg_order_value", "churn_probability", "primary_country"]
    avail = [c for c in cols if c in df.columns]
    return df.nlargest(n, "total_revenue")[avail].reset_index(drop=True)


def generate_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cohort retention matrix from bi_export data."""
    if "first_purchase" not in df.columns or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["cohort"] = df["first_purchase"].dt.to_period("M").astype(str)
    df["last"] = df["last_purchase"].dt.to_period("M")
    df["cohort_period"] = df["first_purchase"].dt.to_period("M")
    df["months_since"] = (df["last"] - df["cohort_period"]).apply(lambda x: x.n if hasattr(x, "n") else 0)

    cohort_size = df.groupby("cohort")["customer_id"].nunique()
    retention = df.groupby(["cohort", "months_since"])["customer_id"].nunique().unstack(fill_value=0)
    retention_pct = retention.div(cohort_size, axis=0)

    return retention_pct.reset_index().rename(columns={"cohort": "cohort_month"})
