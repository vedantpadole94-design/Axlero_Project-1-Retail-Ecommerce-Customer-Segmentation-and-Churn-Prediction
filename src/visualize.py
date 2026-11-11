"""Visualization helpers and diagnostic plotting routines for customer analytics.

This module provides reusable static plotting functions using Matplotlib and Seaborn
for model diagnostics, feature importance rankings, confusion matrices, and RFM distributions.

Pipeline Context:
    model.py & segmentation.py
      └──> visualize.py [THIS MODULE]
             └──> reports/figures/ (PNG diagnostic artifacts)

Typical usage:
    from src.visualize import plot_roc_curve, plot_feature_importances
    fig = plot_roc_curve(y_test, probabilities, model_name="Random Forest")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Set Agg backend for server-side / headless rendering without active display
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve


def plot_roc_curve(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    model_name: str = "Classifier",
    output_path: Path | None = None,
) -> plt.Figure:
    """Render and optionally save an empirical Receiver Operating Characteristic (ROC) curve.

    Business Interpretation:
        The ROC curve displays the diagnostic trade-off between True Positive Rate (Sensitivity)
        and False Positive Rate (1 - Specificity) across every probability threshold.
        An Area Under the Curve (AUC) > 0.75 indicates strong discriminative separation.

    Args:
        y_true: Ground truth binary outcomes (0 or 1).
        y_prob: Predicted continuous probabilities of churn.
        model_name: Display label for the model in the legend.
        output_path: Optional file path to persist the rendered PNG.

    Returns:
        plt.Figure: Rendered matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name}", color="#4F46E5", lw=2)
    # 45-degree diagonal represents random baseline performance (AUC = 0.50)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9CA3AF")
    ax.set(
        xlabel="False Positive Rate (1 - Specificity)",
        ylabel="True Positive Rate (Recall / Sensitivity)",
        title=f"ROC Diagnostic Curve — {model_name}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig


def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    labels: tuple[str, str] = ("Retained", "Churned"),
    output_path: Path | None = None,
) -> plt.Figure:
    """Render a styled confusion matrix heatmap showing classification distributions.

    Business Interpretation:
        Highlights True Positives (correctly identified churners), False Positives
        (retained accounts mistakenly flagged), and False Negatives (unpredicted churners).

    Args:
        y_true: Ground truth binary targets.
        y_pred: Model binary predictions.
        labels: String tuple for class display names. Defaults to ('Retained', 'Churned').
        output_path: Optional file path to persist the image.

    Returns:
        plt.Figure: Rendered matplotlib figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set(
        xlabel="Predicted Classification",
        ylabel="Actual Ground Truth",
        title="Confusion Matrix",
    )
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig


def plot_feature_importances(
    feature_names: list[str],
    importances: list[float] | np.ndarray,
    top_n: int = 10,
    output_path: Path | None = None,
) -> plt.Figure:
    """Render a horizontal bar chart ranking the top predictive drivers of customer churn.

    Business Interpretation:
        Identifies which behavioral features (e.g. order frequency, monetary spend,
        product diversity) contribute most significantly to Random Forest tree splits,
        guiding targeted marketing interventions.

    Args:
        feature_names: List of model predictor column names.
        importances: Array of feature importance values (e.g., Gini impurity decrease).
        top_n: Maximum number of top features to display. Defaults to 10.
        output_path: Optional path to save the generated figure.

    Returns:
        plt.Figure: Rendered matplotlib figure object.
    """
    df_imp = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_imp["feature"], df_imp["importance"], color="#3B82F6", edgecolor="#1D4ED8")
    ax.set(
        xlabel="Relative Importance (Gini / Split Contribution)",
        ylabel="Feature Name",
        title=f"Top {top_n} Churn Predictors",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig


def plot_rfm_distributions(
    rfm_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot distribution histograms for Recency, Frequency, and Monetary metrics.

    Args:
        rfm_df: Customer DataFrame containing 'recency', 'frequency', and 'monetary'.
        output_path: Optional path to persist the figure.

    Returns:
        plt.Figure: Matplotlib figure with 3 distribution subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(rfm_df["recency"], bins=30, color="#6366F1", edgecolor="black", alpha=0.7)
    axes[0].set(title="Recency (days)", xlabel="Days Since Last Purchase", ylabel="Customers")

    axes[1].hist(rfm_df["frequency"], bins=30, color="#10B981", edgecolor="black", alpha=0.7)
    axes[1].set(title="Frequency (orders)", xlabel="Order Count", ylabel="Customers")

    axes[2].hist(rfm_df["monetary"], bins=30, color="#F59E0B", edgecolor="black", alpha=0.7)
    axes[2].set(title="Monetary Spend ($)", xlabel="Total Spend", ylabel="Customers")

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)

    return fig
