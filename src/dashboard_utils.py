"""Utility functions, chart builders, and styling for the Axlero Streamlit dashboard."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Constants ────────────────────────────────────────────────────────────────
SEGMENT_ORDER = ["Champions", "Promising", "At Risk (High Value)", "Hibernating"]

SEGMENT_COLORS = {
    "Champions": "#6366F1",        # indigo
    "Promising": "#10B981",        # emerald
    "At Risk (High Value)": "#F59E0B",  # amber
    "Hibernating": "#EF4444",      # red
}

SEGMENT_EMOJI = {
    "Champions": "🏆",
    "Promising": "🌱",
    "At Risk (High Value)": "⚠️",
    "Hibernating": "💤",
}

SEGMENT_ACTIONS = {
    "Champions": {
        "action": "Reward & Retain",
        "detail": "Reward loyalty with early VIP access, priority support, and exclusive referral incentives.",
        "roi": "High — Champions generate 64% of total revenue; a 5% uplift yields ~$370K.",
    },
    "Promising": {
        "action": "Nurture & Convert",
        "detail": "Deliver targeted second-purchase discounts and personalized cross-sell recommendations.",
        "roi": "Medium — Converting 20% into repeat multi-buyers adds ~$92K in margin.",
    },
    "At Risk (High Value)": {
        "action": "Win-Back Outreach",
        "detail": "Trigger automated high-touch win-back emails with time-sensitive incentives before complete lapse.",
        "roi": "High — Retaining just 10% of At Risk accounts preserves ~$211K in recurring GMV.",
    },
    "Hibernating": {
        "action": "Low-Cost Reactivation",
        "detail": "Deploy automated re-engagement drip sequences; suppress inactive ad spend to cut CAC waste.",
        "roi": "Low — Focus spend on high-LTV tiers; suppress persistent non-responders.",
    },
}

# ── CSS Styling ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1300px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(16,185,129,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99,102,241,0.15);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.82rem;
        color: #4F46E5 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }

    /* Expander card */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        background: rgba(255, 255, 255, 0.6);
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(99,102,241,0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        padding: 4px 0;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stDownloadButton button, .stButton button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stDownloadButton button {
        background-color: #6366F1 !important;
        color: white !important;
        border: none !important;
    }
    .stDownloadButton button:hover {
        background-color: #4F46E5 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.25);
    }

    /* Header styling */
    .hero-subtitle {
        background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(16,185,129,0.08) 100%);
        border-left: 4px solid #6366F1;
        padding: 12px 20px;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0 1.8rem 0;
        font-size: 1.02rem;
        color: #334155;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #94A3B8;
        font-size: 0.85rem;
        border-top: 1px solid rgba(148,163,184,0.2);
        margin-top: 3rem;
    }
    .custom-footer a {
        color: #6366F1;
        text-decoration: none;
    }
    #MainMenu, footer {visibility: hidden;}
</style>
"""

# ── Plotly Base Config ───────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font_family="DM Sans, Inter, sans-serif",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=25, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#0F172A",
        font_color="#FFFFFF",
        font_size=12,
        font_family="DM Sans, Inter, sans-serif",
    ),
)


def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def fmt_currency(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    return f"${value:,.0f}"


def fmt_number(value: float) -> str:
    return f"{value:,.0f}"


# ── Chart Builders ───────────────────────────────────────────────────────────
def build_segment_donut(summary: pd.DataFrame) -> go.Figure:
    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in summary["segment"]]
    fig = go.Figure(
        go.Pie(
            labels=summary["segment"],
            values=summary["count"],
            hole=0.6,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Customer Segment Distribution", font_size=17, x=0.5),
        showlegend=False,
        height=380,
    )
    return fig


def build_revenue_bar(summary: pd.DataFrame) -> go.Figure:
    df = summary.sort_values("revenue", ascending=True)
    colors = [SEGMENT_COLORS.get(s, "#94A3B8") for s in df["segment"]]
    fig = go.Figure(
        go.Bar(
            y=df["segment"],
            x=df["revenue"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            text=[fmt_currency(v) for v in df["revenue"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Total Revenue: $%{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Revenue Contribution by Segment", font_size=17, x=0.5),
        xaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
        yaxis_title="",
        height=380,
        showlegend=False,
    )
    return fig


def build_rfm_scatter(bi_export: pd.DataFrame) -> go.Figure:
    sample = bi_export if len(bi_export) <= 2000 else bi_export.sample(2000, random_state=42)
    fig = px.scatter(
        sample,
        x="recency",
        y="monetary",
        size="frequency",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data={"customer_id": True, "recency": True, "frequency": True, "monetary": ":.0f"},
        opacity=0.75,
        size_max=32,
        category_orders={"segment": SEGMENT_ORDER},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Customer Behavior Map (Recency vs Monetary, Size=Frequency)", font_size=17, x=0.5),
        xaxis_title="Recency (Days Since Last Purchase)",
        yaxis_title="Monetary Value ($)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_churn_distribution(bi_export: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        bi_export,
        x="churn_probability",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        nbins=40,
        barmode="overlay",
        opacity=0.7,
        category_orders={"segment": SEGMENT_ORDER},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Model Churn Probability Distribution", font_size=17, x=0.5),
        xaxis_title="Predicted Churn Probability",
        yaxis_title="Customer Count",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_monthly_revenue(bi_export: pd.DataFrame) -> go.Figure:
    df = bi_export.copy()
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["month"] = df["last_purchase"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("month")
        .agg(revenue=("total_revenue", "sum"), active_customers=("customer_id", "nunique"))
        .reset_index()
        .sort_values("month")
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["revenue"],
            name="Revenue ($)",
            line=dict(color="#6366F1", width=3),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["active_customers"],
            name="Active Customers",
            yaxis="y2",
            line=dict(color="#10B981", width=2.5, dash="dot"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Active Customers: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Monthly Revenue & Active Customers", font_size=17, x=0.5),
        yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="rgba(148,163,184,0.15)"),
        yaxis2=dict(title="Active Customers", overlaying="y", side="right", showgrid=False),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_segment_heatmap(bi_export: pd.DataFrame) -> go.Figure:
    df = bi_export.copy()
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["month"] = df["last_purchase"].dt.to_period("M").astype(str)

    pivot = (
        df.groupby(["segment", "month"])
        .agg(count=("customer_id", "nunique"))
        .reset_index()
        .pivot(index="segment", columns="month", values="count")
        .fillna(0)
    )
    ordered = [s for s in SEGMENT_ORDER if s in pivot.index]
    pivot = pivot.reindex(ordered)

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Tealgrn",
            hovertemplate="<b>%{y}</b><br>Month: %{x}<br>Active: %{z:,} customers<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Segment Activity by Month (Customer Engagement Heatmap)", font_size=17, x=0.5),
        xaxis_title="Month",
        yaxis_title="",
        height=340,
    )
    return fig


def build_cohort_heatmap(cohort_df: pd.DataFrame) -> go.Figure:
    if cohort_df.empty:
        return go.Figure()
    cohorts = cohort_df["cohort_month"].tolist()
    matrix = cohort_df.drop(columns=["cohort_month"])
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=[c.replace("month_", "M+") for c in matrix.columns],
            y=cohorts,
            colorscale="Purples",
            zmin=0,
            zmax=100,
            text=matrix.values,
            texttemplate="%{text:.0f}%",
            hovertemplate="Cohort: %{y}<br>%{x}: %{z:.1f}% retained<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text="Customer Cohort Retention Rate (%)", font_size=17, x=0.5),
        xaxis_title="Months Since First Order",
        yaxis_title="Cohort Month",
        height=380,
    )
    return fig


# ── Data Generators ──────────────────────────────────────────────────────────
def generate_ltv_estimates(bi_export: pd.DataFrame) -> pd.DataFrame:
    df = bi_export.copy()
    lifespan = {"Champions": 3.0, "Promising": 1.5, "At Risk (High Value)": 1.0, "Hibernating": 0.5}
    df["lifespan_factor"] = df["segment"].map(lifespan).fillna(1.0)
    df["ltv"] = (df["avg_order_value"] * df["frequency"] * df["lifespan_factor"]).round(2)
    return df[["customer_id", "segment", "avg_order_value", "frequency", "lifespan_factor", "ltv"]].sort_values("ltv", ascending=False)


def generate_cohort_retention(bi_export: pd.DataFrame) -> pd.DataFrame:
    df = bi_export.copy()
    df["first_purchase"] = pd.to_datetime(df["first_purchase"])
    df["last_purchase"] = pd.to_datetime(df["last_purchase"])
    df["cohort_month"] = df["first_purchase"].dt.to_period("M")
    df["activity_month"] = df["last_purchase"].dt.to_period("M")
    df["months_since_join"] = (df["activity_month"] - df["cohort_month"]).apply(lambda x: x.n if hasattr(x, "n") else 0)

    retention = df.groupby(["cohort_month", "months_since_join"])["customer_id"].nunique().reset_index()
    retention.columns = ["cohort_month", "months_since_join", "customers"]
    retention["cohort_month"] = retention["cohort_month"].astype(str)

    pivot = retention.pivot(index="cohort_month", columns="months_since_join", values="customers").fillna(0)
    for col in pivot.columns:
        if col != 0:
            pivot[col] = (pivot[col] / pivot[0] * 100).round(1)
    pivot[0] = 100.0
    pivot.columns = [f"month_{int(c)}" for c in pivot.columns]
    return pivot.reset_index()


def generate_segment_health(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    if df.empty:
        return df
    max_churn = df["churn_rate"].max() if df["churn_rate"].max() > 0 else 1.0
    max_rev = df["revenue"].max() if df["revenue"].max() > 0 else 1.0
    max_rec = df["avg_recency"].max() if df["avg_recency"].max() > 0 else 1.0

    churn_score = 1.0 - (df["churn_rate"] / max_churn)
    rev_score = df["revenue"] / max_rev
    rec_score = 1.0 - (df["avg_recency"] / max_rec)

    df["health_score"] = ((churn_score * 0.40 + rev_score * 0.35 + rec_score * 0.25) * 100).round(1)
    return df[["segment", "count", "revenue", "churn_rate", "avg_recency", "health_score"]]


def generate_top_customers(bi_export: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    cols = ["customer_id", "primary_country", "segment", "total_orders", "total_revenue", "avg_order_value", "recency", "frequency", "monetary", "churn_probability"]
    avail = [c for c in cols if c in bi_export.columns]
    return bi_export.nlargest(n, "monetary")[avail]
