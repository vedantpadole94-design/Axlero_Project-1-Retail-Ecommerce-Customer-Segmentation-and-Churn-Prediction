"""Customer RFM behavioral segmentation using unsupervised K-Means clustering.

This module prepares scaled RFM features, evaluates optimal cluster counts
(Elbow method and Silhouette analysis), partitions customers into four behavioral
cohorts using K-Means, and maps cluster centroids into actionable marketing personas:
Champions, Promising, At Risk (High Value), and Hibernating.

Pipeline Context:
    features.py (RFM extraction & log scaling)
      └──> segmentation.py [THIS MODULE]
             ├──> data/processed/rfm_segments.parquet (Segment assignments)
             ├──> reports/figures/optimal_k.png (Elbow/Silhouette diagnostics)
             └──> reports/figures/rfm_segments_3d.png (3D cluster map)
                    └──> model.py & app.py (Dashboard presentation)

Typical usage:
    from src.segmentation import fit_rfm_segments, prepare_rfm
    segments_df, profile_df, scaler, kmeans_model = fit_rfm_segments(rfm_df, n_clusters=4)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use non-interactive Agg backend for headless diagram rendering
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

# Strategic marketing playbooks aligned with customer lifecycle stage
SEGMENT_ACTIONS = {
    "Champions": "Reward loyalty with early access, VIP service, and referral incentives.",
    "Promising": "Use a low-friction second-purchase offer and personalized product recommendations.",
    "At Risk (High Value)": "Trigger personal win-back outreach with a tailored incentive before lapse.",
    "Hibernating": "Use low-cost reactivation campaigns; suppress or reduce spend if unresponsive.",
}


def prepare_rfm(rfm: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Log-transform and standardize RFM inputs, returning scaled data and fitted scaler.

    Why StandardScaler before K-Means:
        K-Means computes Euclidean distances between customer feature vectors.
        Monetary spend spans \$0 to \$100,000+, whereas order frequency spans 1 to 50.
        Without standardization (mean=0, unit variance), monetary spend would dominate
        the distance metric by a factor of 1,000x, rendering order frequency and recency
        statistically irrelevant in cluster formation.

    Args:
        rfm: Customer DataFrame containing 'customer_id', 'recency', 'frequency', and 'monetary'.

    Returns:
        tuple[pd.DataFrame, StandardScaler]:
            - pd.DataFrame: Standardized numeric DataFrame ready for K-Means.
            - StandardScaler: Fitted scaler object for inference or inversion.

    Raises:
        ValueError: If required RFM columns are missing.
        ValueError: If input DataFrame has no rows after removing nulls.
    """
    missing = sorted(set(["customer_id", *RFM_COLUMNS]).difference(rfm.columns))
    if missing:
        raise ValueError(f"RFM data is missing required columns: {missing}")

    clean = rfm.dropna(subset=["customer_id", *RFM_COLUMNS]).copy()
    if clean.empty:
        raise ValueError("No complete RFM rows are available for clustering.")

    # Apply log1p transform first to compress heavy power-law tails
    logged = log1p_transform(clean, tuple(RFM_COLUMNS))

    # Standardize to zero mean and unit variance for distance-based clustering
    scaler = StandardScaler()
    scaled = scaler.fit_transform(logged[RFM_COLUMNS])
    result = pd.DataFrame(scaled, columns=RFM_COLUMNS, index=clean.index)
    return result, scaler


def evaluate_cluster_counts(
    scaled_rfm: pd.DataFrame,
    k_values: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Calculate within-cluster inertia and silhouette scores across candidate k values.

    Args:
        scaled_rfm: Standardized RFM feature DataFrame.
        k_values: Iterable range of candidate cluster counts. Defaults to range(2, 11).
        random_state: Random seed for deterministic KMeans initialization.

    Returns:
        pd.DataFrame: Table recording 'k', 'inertia', and 'silhouette' scores.

    Raises:
        ValueError: If fewer than three customer records are supplied.
        ValueError: If no valid k evaluations could be computed.
    """
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
    """Render and save Elbow and Silhouette diagnostic charts to disk.

    Args:
        scores: DataFrame output from evaluate_cluster_counts.
        path: Optional file path for the figure. Defaults to reports/figures/optimal_k.png.

    Returns:
        Path: Absolute file path where the plot was saved.
    """
    output = path or FIGURES_DIR / "optimal_k.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Elbow Method: Plots reduction in within-cluster sum-of-squares (inertia)
    axes[0].plot(scores["k"], scores["inertia"], marker="o")
    axes[0].set(title="Elbow method (Inertia)", xlabel="Number of clusters (k)", ylabel="Inertia")

    # 2. Silhouette Analysis: Evaluates cluster separation and cohesion
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
    """Map raw numerical cluster IDs to 4 standardized business personas using z-scores.

    Business Persona Derivation Logic:
      1. Champions (🏆): Highest combined score of low recency, high frequency, and high monetary.
         Formula: z_score = -z(recency) + z(frequency) + z(monetary).
      2. Hibernating (😴): Lower spenders with the longest period of inactivity (maximum recency).
         Formula: Highest z(recency) among below-median frequency/monetary clusters.
      3. At Risk - High Value (⚠️): Customers with high historic spend and frequency but elevated recency.
         Formula: z_score = +z(recency) + z(frequency) + z(monetary) among remaining clusters.
      4. Promising (🌱): Recent buyers with moderate spend and frequency representing growth upside.

    Args:
        profile: DataFrame of cluster centroids indexed by cluster_id.

    Returns:
        dict[int, str]: Mapping from cluster integer IDs to business persona names.
    """
    means = profile[RFM_COLUMNS].astype(float)
    # Compute z-scores across cluster means to normalize relative standing
    spread = means.std(ddof=0).replace(0, 1)
    z = (means - means.mean()) / spread
    remaining = set(profile.index.astype(int))

    # Identify Champions: Low recency (negative z) + high frequency + high spend
    champion_scores = -z["recency"] + z["frequency"] + z["monetary"]
    champion = int(champion_scores.loc[list(remaining)].idxmax())
    remaining.remove(champion)

    # Identify Hibernating: Longest inactive days among low-value clusters
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

    # Identify At Risk (High Value): High historic spend and frequency, but high recency (dormant)
    risk_scores = z["recency"] + z["frequency"] + z["monetary"]
    at_risk = int(risk_scores.loc[list(remaining)].idxmax())
    remaining.remove(at_risk)

    # Remaining cluster represents Promising new/growing accounts
    promising = remaining.pop()

    return {
        champion: "Champions",
        promising: "Promising",
        at_risk: "At Risk (High Value)",
        hibernating: "Hibernating",
    }


# Standard alias
label_segments = assign_business_labels


def fit_rfm_segments(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, KMeans]:
    """Fit KMeans to log-standardized RFM data, assign persona labels, and produce profiles.

    Why k=4:
        Empirical evaluation of the Elbow curve and Silhouette coefficients reveals an
        inflection point at k=4. In addition, 4 cohorts correspond directly to the
        4 fundamental operational lifecycle states in retail CRM: VIPs, New Growth,
        Lapsing Accounts, and Churned/Dormant Accounts.

    Args:
        rfm: Clean customer RFM DataFrame.
        n_clusters: Number of clusters to partition. Defaults to 4.
        random_state: Deterministic random seed for KMeans centroid seeding.

    Returns:
        tuple containing:
            - pd.DataFrame: Customer table annotated with cluster_id, segment, and recommendation.
            - pd.DataFrame: Aggregate profile summary table for each segment.
            - StandardScaler: Fitted scaler used during clustering.
            - KMeans: Fitted scikit-learn KMeans estimator.

    Raises:
        ValueError: If fewer customer rows exist than the requested n_clusters.
        RuntimeError: If cluster IDs do not cleanly map to all distinct business labels.
    """
    scaled, scaler = prepare_rfm(rfm)
    if len(scaled) < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} customers; received {len(scaled)}."
        )

    # Fit K-Means on log-standardized features
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = model.fit_predict(scaled)

    segments = rfm.loc[scaled.index, ["customer_id", *RFM_COLUMNS]].copy()
    segments["cluster_id"] = cluster_ids

    # Compute unscaled average profile metrics for business interpretation
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


# Standard alias
fit_kmeans = fit_rfm_segments


def save_segment_figures(segments: pd.DataFrame) -> tuple[Path, Path]:
    """Render and save 3D RFM scatter and segment profile bar charts.

    Args:
        segments: Customer DataFrame with 'recency', 'frequency', 'monetary', and 'segment'.

    Returns:
        tuple[Path, Path]: Paths to the saved 3D scatter and profile bar chart images.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scatter_path = FIGURES_DIR / "rfm_segments_3d.png"
    bar_path = FIGURES_DIR / "rfm_segment_profiles.png"

    # Render 3D cluster separation plot
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

    # Render bar charts of average RFM metrics by segment
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
    """Execute customer segmentation pipeline and output diagnostic figures."""
    rfm_path = ROOT / "data" / "interim" / "rfm.parquet"
    if not rfm_path.is_file():
        raise FileNotFoundError(
            f"RFM data not found at {rfm_path}. Run python src/features.py first."
        )

    rfm = pd.read_parquet(rfm_path)
    scaled, _ = prepare_rfm(rfm)

    # Calculate optimal-k diagnostics (Elbow & Silhouette)
    scores = evaluate_cluster_counts(scaled)
    plot_path = save_optimal_k_plot(scores)

    # Segment customers into k=4 behavioral cohorts
    segments, profile, _, _ = fit_rfm_segments(rfm, n_clusters=4)
    SEGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(SEGMENTS_PATH, index=False)

    # Generate visual artifacts
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
