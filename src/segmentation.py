"""RFM clustering helpers and report visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if __package__:
    from .features import log1p_transform
else:
    from features import log1p_transform

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "reports" / "figures"
SEGMENTS_PATH = ROOT / "data" / "processed" / "rfm_segments.parquet"
RFM_COLUMNS = ["recency", "frequency", "monetary"]
SEGMENT_ACTIONS = {
    "Champions": "Reward loyalty with early access, VIP service, and referral incentives.",
    "Promising": "Use a low-friction second-purchase offer and personalized product recommendations.",
    "At Risk (High Value)": "Trigger personal win-back outreach with a tailored incentive before lapse.",
    "Hibernating": "Use low-cost reactivation campaigns; suppress or reduce spend if unresponsive.",
}


def prepare_rfm(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Log-transform and standardize RFM inputs, returning the fitted scaler."""
    missing = sorted(set(["customer_id", *RFM_COLUMNS]).difference(rfm.columns))
    if missing:
        raise ValueError(f"RFM data is missing required columns: {missing}")
    clean = rfm.dropna(subset=["customer_id", *RFM_COLUMNS]).copy()
    if clean.empty:
        raise ValueError("No complete RFM rows are available for clustering.")
    logged = log1p_transform(clean, tuple(RFM_COLUMNS))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(logged[RFM_COLUMNS])
    result = pd.DataFrame(scaled, columns=RFM_COLUMNS, index=clean.index)
    return result, scaler


def evaluate_cluster_counts(
    scaled_rfm: pd.DataFrame,
    k_values: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate within-cluster inertia and silhouette scores for candidate k."""
    if len(scaled_rfm) < 3:
        raise ValueError("At least three customers are required for clustering.")
    rows = []
    for k in k_values:
        if k >= len(scaled_rfm):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled_rfm)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_rfm, labels),
            }
        )
    if not rows:
        raise ValueError("No valid k values are available for clustering.")
    return pd.DataFrame(rows)


def save_optimal_k_plot(scores: pd.DataFrame, path: Path | None = None) -> Path:
    """Save elbow and silhouette diagnostics for candidate cluster counts."""
    output = path or FIGURES_DIR / "optimal_k.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set(title="Elbow method", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[1].plot(scores["k"], scores["silhouette"], marker="o", color="darkorange")
    axes[1].set(
        title="Silhouette score",
        xlabel="Number of clusters (k)",
        ylabel="Mean silhouette score",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output


def assign_business_labels(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster means to four distinct RFM business labels."""
    means = profile[RFM_COLUMNS].astype(float)
    spread = means.std(ddof=0).replace(0, 1)
    z = (means - means.mean()) / spread
    remaining = set(profile.index.astype(int))

    champion_scores = -z["recency"] + z["frequency"] + z["monetary"]
    champion = int(champion_scores.loc[list(remaining)].idxmax())
    remaining.remove(champion)

    lower_value = remaining.intersection(
        set(
            means.index[
                (means["frequency"] <= means["frequency"].median())
                & (means["monetary"] <= means["monetary"].median())
            ].astype(int)
        )
    )
    hibernating_candidates = lower_value or remaining
    hibernating = int(z["recency"].loc[list(hibernating_candidates)].idxmax())
    remaining.remove(hibernating)

    risk_scores = z["recency"] + z["frequency"] + z["monetary"]
    at_risk = int(risk_scores.loc[list(remaining)].idxmax())
    remaining.remove(at_risk)

    promising = remaining.pop()
    return {
        champion: "Champions",
        promising: "Promising",
        at_risk: "At Risk (High Value)",
        hibernating: "Hibernating",
    }


def fit_rfm_segments(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans]:
    """Fit KMeans to log-scaled RFM and produce profiles and business labels."""
    scaled, scaler = prepare_rfm(rfm)
    if len(scaled) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} customers; received {len(scaled)}."
        )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = model.fit_predict(scaled)
    segments = rfm.loc[scaled.index, ["customer_id", *RFM_COLUMNS]].copy()
    segments["cluster_id"] = cluster_ids
    profile = (
        segments.groupby("cluster_id")
        .agg(
            count=("customer_id", "nunique"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .sort_index()
    )
    cluster_to_label = assign_business_labels(profile)
    segments["segment"] = segments["cluster_id"].map(cluster_to_label)
    profile["segment"] = profile.index.map(cluster_to_label)
    profile["marketing_recommendation"] = profile["segment"].map(SEGMENT_ACTIONS)
    segments["marketing_recommendation"] = segments["segment"].map(SEGMENT_ACTIONS)
    if segments["segment"].nunique() != n_clusters:
        raise RuntimeError("Business labels did not map to all fitted clusters.")
    return segments, profile.reset_index(), scaler, model


def save_segment_figures(segments: pd.DataFrame) -> tuple[Path, Path]:
    """Save a 3D RFM scatterplot and cluster-level RFM bar chart."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scatter_path = FIGURES_DIR / "rfm_segments_3d.png"
    bar_path = FIGURES_DIR / "rfm_segment_profiles.png"

    fig = plt.figure(figsize=(10, 7))
    axis = fig.add_subplot(111, projection="3d")
    for segment, group in segments.groupby("segment", sort=True):
        axis.scatter(
            group["recency"],
            group["frequency"],
            group["monetary"],
            label=segment,
            alpha=0.65,
            s=18,
        )
    axis.set(xlabel="Recency (days)", ylabel="Frequency (orders)", zlabel="Monetary")
    axis.set_title("Customer RFM segments")
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    profile = (
        segments.groupby("segment")[RFM_COLUMNS]
        .mean()
        .reindex(["Champions", "Promising", "At Risk (High Value)", "Hibernating"])
    )
    axis = profile.plot(kind="bar", subplots=True, layout=(1, 3), figsize=(13, 4), legend=False)
    figure = axis[0, 0].get_figure()
    figure.suptitle("Mean RFM by business segment")
    figure.tight_layout()
    figure.savefig(bar_path, dpi=100, bbox_inches="tight")
    plt.close(figure)
    return scatter_path, bar_path


def main() -> None:
    rfm_path = ROOT / "data" / "interim" / "rfm.parquet"
    if not rfm_path.is_file():
        raise FileNotFoundError(
            f"RFM data not found at {rfm_path}. Run python src/features.py first."
        )
    rfm = pd.read_parquet(rfm_path)
    scaled, _ = prepare_rfm(rfm)
    scores = evaluate_cluster_counts(scaled)
    plot_path = save_optimal_k_plot(scores)
    segments, profile, _, _ = fit_rfm_segments(rfm, n_clusters=4)
    SEGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(SEGMENTS_PATH, index=False)
    scatter_path, bar_path = save_segment_figures(segments)
    print(f"✅ Candidate-k diagnostics saved to {plot_path.relative_to(ROOT)}")
    print(f"✅ Four customer segments saved to {SEGMENTS_PATH.relative_to(ROOT)}")
    print(profile[["segment", "count", *RFM_COLUMNS]].to_string(index=False))
    print(
        f"✅ Segment figures saved to {scatter_path.relative_to(ROOT)} and "
        f"{bar_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
