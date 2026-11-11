"""Evaluation utilities and diagnostic metrics for customer churn models.

This module provides benchmarking routines, lift tables, and classification
metrics (ROC-AUC, Precision, Recall, F1, Accuracy) to validate that churn models
meet production standards (e.g. holdout ROC-AUC >= 0.75).

Pipeline Context:
    model.py (Model training and predictions)
      └──> evaluate.py [THIS MODULE]
             └──> reports/model_metrics.md (Governance documentation)

Typical usage:
    from src.evaluate import compute_classification_metrics, calculate_lift_table
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive binary classification evaluation metrics.

    Business Relevance:
        - ROC-AUC: Measures rank-ordering ability across all possible thresholds.
        - Precision: Indicates the proportion of identified at-risk customers who truly churn.
          High precision prevents wasted marketing incentives on retained customers.
        - Recall: Indicates the proportion of actual churners successfully captured.
          High recall minimizes surprise churn of high-value accounts.

    Args:
        y_true: Ground truth binary labels (0 = retained, 1 = churned).
        y_pred: Hard binary predictions (0 or 1).
        y_prob: Continuous predicted probabilities for the positive class (churn=1).
            If provided, 'roc_auc' is included in the output.

    Returns:
        dict[str, float]: Performance metric dictionary with keys:
            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc' (optional)].

    Example:
        >>> import numpy as np
        >>> y_t = np.array([0, 1, 0, 1])
        >>> y_p = np.array([0, 1, 0, 0])
        >>> y_pr = np.array([0.1, 0.9, 0.2, 0.4])
        >>> m = compute_classification_metrics(y_t, y_p, y_pr)
        >>> m['accuracy']
        0.75
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))

    return metrics


def calculate_lift_table(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Calculate cumulative gains and lift table across probability deciles.

    Why Lift Matters:
        In targeted marketing campaigns with fixed budgets, marketing teams cannot
        contact 100% of the customer base. Lift demonstrates how much more effective
        targeting the top deciles is compared to random outreach.

    Args:
        y_true: Ground truth binary labels (1 = churn, 0 = active).
        y_prob: Predicted churn probabilities.
        n_bins: Number of equal-frequency quantiles/deciles. Defaults to 10.

    Returns:
        pd.DataFrame: Lift table with columns:
            ['decile', 'total_count', 'churn_count', 'churn_rate', 'lift', 'cumulative_gain'].
    """
    data = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    # Sort descending by predicted risk probability
    data = data.sort_values("y_prob", ascending=False).reset_index(drop=True)
    # Assign equal-frequency deciles
    data["decile"] = pd.qcut(data.index, q=n_bins, labels=False) + 1

    overall_rate = data["y_true"].mean()
    summary = (
        data.groupby("decile")
        .agg(
            total_count=("y_true", "count"),
            churn_count=("y_true", "sum"),
        )
        .reset_index()
    )
    summary["churn_rate"] = summary["churn_count"] / summary["total_count"]
    # Lift = decile churn rate relative to average population churn rate
    summary["lift"] = summary["churn_rate"] / overall_rate if overall_rate > 0 else 0
    # Cumulative percentage of total churners captured
    summary["cumulative_gain"] = (
        summary["churn_count"].cumsum() / summary["churn_count"].sum() * 100
    )
    return summary


def evaluate_model_performance(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained model pipeline against unseen holdout test data.

    Args:
        model: Trained classifier exposing ``predict`` and ``predict_proba``.
        X_test: Feature matrix for testing.
        y_test: Ground truth test labels.

    Returns:
        dict[str, float]: Evaluation metrics dictionary.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    return compute_classification_metrics(y_test, predictions, probabilities)
