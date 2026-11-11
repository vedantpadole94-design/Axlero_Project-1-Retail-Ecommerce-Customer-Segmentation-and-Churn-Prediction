"""Train and evaluate customer churn classification models.

This module builds predictive churn models for e-commerce customers using
behavioral and transactional indicators. It handles label generation,
data leakage prevention, class imbalance resolution via SMOTE, cross-validated
hyperparameter tuning, and probability scoring for business intelligence dashboards.

Pipeline Context:
    features.py (RFM extraction) & make_dataset.py (Customer summary)
      └──> model.py [THIS MODULE]
             ├──> models/churn_model.joblib (Tuned model bundle)
             ├──> data/processed/customers_scored.parquet (Scored customer base)
             └──> reports/model_metrics.md & reports/figures/ (Audit reporting)

Typical usage:
    from src.model import load_and_label, build_features, run_training
    best_model, metrics, scored_df = run_training()
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

# Set non-interactive Agg backend before importing pyplot to allow headless rendering
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from .features import compute_rfm
else:
    from features import compute_rfm

ROOT = Path(__file__).resolve().parents[1]
RAW_CUSTOMERS_PATH = ROOT / "data" / "interim" / "users_raw.parquet"
RFM_PATH = ROOT / "data" / "interim" / "rfm.parquet"
SEGMENTS_PATH = ROOT / "data" / "processed" / "rfm_segments.parquet"
SCORED_PATH = ROOT / "data" / "processed" / "customers_scored.parquet"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_metrics.md"
FIGURES_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42

# Business rule: In high-repeat retail/e-commerce, customers with no purchase
# activity in the past 60 days are operationally categorized as lapsed/churned.
CHURN_THRESHOLD_DAYS = 60

BASE_FEATURES = [
    "frequency",
    "monetary",
    "tenure_days",
    "avg_order_value",
    "distinct_products",
    "distinct_countries",
    "total_quantity",
    "avg_days_between_orders",
    "monetary_per_order",
    "product_diversity",
]
OPTIONAL_FEATURES = ["order_regularity"]


def load_and_label(
    customers: pd.DataFrame | None = None,
    rfm: pd.DataFrame | None = None,
    churn_threshold_days: int = CHURN_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """Join customer-level predictors to the recency-based churn target.

    Business Rationale:
        Customer churn is defined operationally by purchase inactivity. If a customer
        has not placed an order within ``churn_threshold_days`` (default: 60 days),
        they are labeled as churned (1). Otherwise, they are active (0).

    Args:
        customers: DataFrame of customer profiles and summary aggregates. If None,
            loads from `data/interim/users_raw.parquet`.
        rfm: DataFrame of customer RFM values. If None, loads from `rfm.parquet`
            or computes dynamically from customer data.
        churn_threshold_days: Inactivity threshold in days beyond which a customer
            is marked as churned. Defaults to 60 days.

    Returns:
        pd.DataFrame: Merged DataFrame containing customer attributes and a binary
        'churn' label column (1 for churned, 0 for retained).

    Raises:
        FileNotFoundError: If default customer input file does not exist.
        ValueError: If 'customer_id' is missing from either customer or RFM data.
        ValueError: If churn labels contain only a single class (stratified split impossible).

    Example:
        >>> import pandas as pd
        >>> cust = pd.DataFrame({"customer_id": [1, 2], "tenure_days": [100, 20]})
        >>> rfm_df = pd.DataFrame({
        ...     "customer_id": [1, 2],
        ...     "recency": [80, 15],
        ...     "frequency": [2, 5],
        ...     "monetary": [100.0, 500.0]
        ... })
        >>> labeled = load_and_label(cust, rfm_df, churn_threshold_days=60)
        >>> labeled[["customer_id", "churn"]].to_dict(orient="records")
        [{'customer_id': 1, 'churn': 1}, {'customer_id': 2, 'churn': 0}]
    """
    if customers is None:
        if not RAW_CUSTOMERS_PATH.is_file():
            raise FileNotFoundError(f"Customer input is missing: {RAW_CUSTOMERS_PATH}")
        customers = pd.read_parquet(RAW_CUSTOMERS_PATH)

    if rfm is None:
        if RFM_PATH.is_file():
            rfm = pd.read_parquet(RFM_PATH)
        else:
            rfm, _ = compute_rfm(customers)

    if "customer_id" not in customers or "customer_id" not in rfm:
        raise ValueError("Both customer and RFM data must include customer_id.")

    labeled = customers.merge(
        rfm[["customer_id", "recency", "frequency", "monetary"]],
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    # Business rule: Customers inactive for > 60 days are marked as churned (1)
    labeled["churn"] = (labeled["recency"] > churn_threshold_days).astype("int8")

    if labeled["churn"].nunique() != 2:
        raise ValueError(
            "Churn labels contain only one class; a stratified model split is not possible."
        )
    return labeled


def build_features(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create leakage-aware predictors, strictly excluding recency and the target.

    Anti-Leakage Safeguard:
        Because the target 'churn' is defined directly from 'recency' (> 60 days),
        including 'recency' as an input feature would introduce severe label leakage,
        artificially inflating model metrics while providing zero predictive utility.
        Recency is therefore strictly excluded from the feature matrix X.

    Args:
        labeled: Customer DataFrame including 'churn' label and behavioral columns.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Clean feature matrix X and target labels y.

    Raises:
        ValueError: If 'frequency' is missing or no usable numeric predictors exist.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "frequency": [2, 4], "monetary": [100.0, 250.0],
        ...     "tenure_days": [60, 120], "churn": [0, 1]
        ... })
        >>> X, y = build_features(df)
        >>> list(X.columns)
        ['frequency', 'monetary', 'tenure_days', 'avg_days_between_orders', 'monetary_per_order', 'product_diversity']
    """
    data = labeled.copy()
    numeric = lambda name: pd.to_numeric(data[name], errors="coerce")

    if "frequency" not in data:
        raise ValueError("Labeled data must include frequency.")

    frequency = numeric("frequency").clip(lower=1)
    monetary = numeric("monetary")
    tenure = numeric("tenure_days") if "tenure_days" in data else pd.Series(
        np.nan, index=data.index
    )
    distinct_products = (
        numeric("distinct_products")
        if "distinct_products" in data
        else pd.Series(np.nan, index=data.index)
    )

    # Derive behavioral interaction indicators
    data["avg_days_between_orders"] = tenure / (frequency - 1).clip(lower=1)
    data["monetary_per_order"] = monetary / frequency
    data["product_diversity"] = distinct_products / frequency

    # Collect all available engineered features
    available = [
        column
        for column in [*BASE_FEATURES, *OPTIONAL_FEATURES]
        if column in data.columns and data[column].notna().any()
    ]
    if not available:
        raise ValueError("No usable numeric customer predictors are available.")

    X = data[available].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data["churn"].astype("int8")
    complete = y.notna()

    return X.loc[complete].reset_index(drop=True), y.loc[complete].reset_index(drop=True)


def split_and_balance(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, SimpleImputer]:
    """Perform a stratified holdout split, impute medians, and balance training data via SMOTE.

    Methodology & Leakage Prevention:
        1. Stratified split (80% train, 20% test) preserves class ratios across splits.
        2. Imputer fits on training data ONLY; test set is transformed with train medians.
        3. SMOTE (Synthetic Minority Over-sampling Technique) is applied ONLY to training data.
           Applying SMOTE before splitting or to test data causes data leakage where synthetic
           training examples contaminate the evaluation holdout.

    Args:
        X: Feature matrix of numeric customer attributes.
        y: Binary target series (0 = retained, 1 = churned).

    Returns:
        tuple containing:
            - pd.DataFrame: Balanced training features (X_train).
            - pd.DataFrame: Holdout test features (X_test).
            - pd.Series: Balanced training targets (y_train).
            - pd.Series: Holdout test targets (y_test).
            - SimpleImputer: Fitted imputer object for inference.

    Raises:
        ValueError: If training minority class has fewer than 2 examples for SMOTE.
    """
    # 80/20 train/test split stratified by churn to guarantee proportional representation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Learn median values from training partition only to avoid data leakage
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    minority_count = int(y_train.value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two minority-class training examples.")

    # Synthesize minority class samples using SMOTE on training partition only
    sampler = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=min(5, minority_count - 1),
    )
    X_balanced, y_balanced = sampler.fit_resample(X_train_imputed, y_train)

    return (
        pd.DataFrame(X_balanced, columns=X.columns),
        X_test_imputed.reset_index(drop=True),
        pd.Series(y_balanced, name=y.name),
        y_test.reset_index(drop=True),
        imputer,
    )


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit a standardized, class-weighted logistic regression baseline classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary target series.

    Returns:
        sklearn Pipeline: Fitted pipeline with StandardScaler and LogisticRegression.
    """
    # StandardScaler ensures features on different scales (spend vs order count) contribute fairly
    # class_weight='balanced' penalizes misclassification of the churn minority class
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )
    return model.fit(X_train, y_train)


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune a Random Forest classifier for ROC-AUC via 3-fold cross-validation.

    Why ROC-AUC over Accuracy:
        In customer churn prediction, classes are imbalanced. A naive model predicting
        'no churn' for all customers achieves high accuracy while failing 100% of the
        business objective. ROC-AUC evaluates discrimination across all probability
        thresholds, ensuring optimal rank-ordering of high-risk accounts.

    Why GridSearchCV on Hyperparameters:
        - `n_estimators`: [200, 400] tests stability against overfitting.
        - `max_depth`: [6, 10, None] constrains tree complexity to avoid memorization.
        - `cv=3`: 3-fold stratified cross-validation balances validation rigor and compute time.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary target series.

    Returns:
        GridSearchCV: Fitted grid search object with the best estimator accessible
        via ``best_search.best_estimator_``.
    """
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        param_grid={
            "n_estimators": [200, 400],
            "max_depth": [6, 10, None],
        },
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        refit=True,
    )
    return search.fit(X_train, y_train)


def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    name: str,
) -> dict[str, float]:
    """Evaluate holdout test metrics and render an audit ROC curve figure.

    Args:
        model: Trained classifier implementing ``predict`` and ``predict_proba``.
        X_test: Unseen test feature matrix.
        y_test: True test binary targets.
        name: Human-readable model identifier (e.g. 'Random Forest').

    Returns:
        dict[str, float]: Performance dictionary containing 'roc_auc', 'precision',
        'recall', 'f1', and 'accuracy'.
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)

    # Plot and persist diagnostic ROC curve
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{name} (AUC = {metrics['roc_auc']:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"ROC curve — {name}",
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"roc_{name.lower().replace(' ', '_')}.png", dpi=100)
    plt.close(fig)

    print(
        f"{name}: ROC-AUC={metrics['roc_auc']:.3f}, "
        f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )

    if metrics["roc_auc"] < 0.75:
        warnings.warn(
            f"{name} ROC-AUC {metrics['roc_auc']:.3f} is below the 0.75 target. "
            "See reports/model_metrics.md for limitations and next steps.",
            RuntimeWarning,
            stacklevel=2,
        )
    return metrics


def save_model(
    model: object,
    feature_names: list[str],
    churn_threshold_days: int,
    selected_model: str,
    imputer: SimpleImputer,
    output_path: Path = MODEL_PATH,
) -> None:
    """Serialize the trained model, imputer, and feature metadata to disk.

    Args:
        model: Fitted scikit-learn model object.
        feature_names: List of column names expected by the model.
        churn_threshold_days: Business recency threshold in days.
        selected_model: Name of the winning model architecture.
        imputer: Fitted SimpleImputer for handling missing runtime inputs.
        output_path: Target path for the joblib bundle. Defaults to models/churn_model.joblib.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "feature_names": feature_names,
            "churn_threshold_days": churn_threshold_days,
            "selected_model": selected_model,
            "imputer": imputer,
        },
        output_path,
    )


def _write_metrics_report(
    results: dict[str, dict[str, float]],
    best_name: str,
    best_auc: float,
    retention: float,
    feature_names: list[str],
    best_params: dict[str, object] | None,
) -> None:
    """Generate Markdown summary documentation for governance and model review."""
    lines = [
        "# Churn Model Metrics",
        "",
        f"- Customer retention after feature preparation: {retention:.1%}",
        f"- Churn label: recency greater than {CHURN_THRESHOLD_DAYS} days.",
        "- Recency is intentionally excluded from predictors because it defines the target.",
        f"- Selected model: {best_name}",
        f"- Selected holdout ROC-AUC: {best_auc:.3f}",
        f"- ROC-AUC target (0.75): {'MET' if best_auc >= 0.75 else 'NOT MET'}",
        f"- Predictors: {', '.join(feature_names)}",
        "",
        "## Held-out metrics",
        "",
        "| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['roc_auc']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['accuracy']:.3f} |"
        )
    if best_params:
        lines.extend(["", f"Random forest best parameters: `{best_params}`"])
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "The current churn outcome is a recency threshold at a single snapshot, not "
            "observed future churn. As a result, these results measure how well past "
            "purchase aggregates distinguish currently lapsed customers; they do not "
            "establish prospective 30-day churn performance. Recency is excluded from "
            "predictors to avoid direct target leakage. A time-based holdout and "
            "future-window outcome should be used before operational deployment.",
            "",
        ]
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_training() -> tuple[object, dict[str, dict[str, float]], pd.DataFrame]:
    """Execute end-to-end model training, validation, selection, scoring, and persistence.

    Returns:
        tuple containing:
            - object: Best trained model instance.
            - dict: Comparative holdout metrics for all candidates.
            - pd.DataFrame: Full customer cohort scored with churn probabilities.
    """
    customers = pd.read_parquet(RAW_CUSTOMERS_PATH)
    rfm = pd.read_parquet(RFM_PATH) if RFM_PATH.is_file() else None
    labeled = load_and_label(customers, rfm)
    X, y = build_features(labeled)

    if len(X) != len(customers):
        raise RuntimeError(
            f"Unexpected customer loss: {len(X):,} feature rows for "
            f"{len(customers):,} source rows."
        )

    # Split, impute, and apply SMOTE to training partition only
    X_train, X_test, y_train, y_test, scoring_imputer = split_and_balance(X, y)

    # 1. Baseline: Logistic Regression
    logistic = train_logistic(X_train, y_train)
    logistic_metrics = evaluate(logistic, X_test, y_test, "Logistic Regression")

    # 2. Candidate: Random Forest (Tuned via 3-Fold Grid Search)
    forest_search = train_random_forest(X_train, y_train)
    forest = forest_search.best_estimator_
    forest_metrics = evaluate(forest, X_test, y_test, "Random Forest")

    results = {
        "Logistic Regression": logistic_metrics,
        "Random Forest": forest_metrics,
    }

    # Model selection based strictly on holdout ROC-AUC
    candidates = {
        "Logistic Regression": (logistic, logistic_metrics["roc_auc"]),
        "Random Forest": (forest, forest_metrics["roc_auc"]),
    }
    best_name = max(candidates, key=lambda name: candidates[name][1])
    best_model, best_auc = candidates[best_name]

    # Save model artifacts using dedicated save_model helper
    save_model(
        model=best_model,
        feature_names=list(X.columns),
        churn_threshold_days=CHURN_THRESHOLD_DAYS,
        selected_model=best_name,
        imputer=scoring_imputer,
        output_path=MODEL_PATH,
    )

    # Batch score entire customer database for BI dashboard consumption
    X_all = pd.DataFrame(
        scoring_imputer.transform(X),
        columns=X.columns,
        index=X.index,
    )
    scored = labeled[["customer_id", "recency", "churn"]].copy()
    if "segment" in labeled:
        scored["segment"] = labeled["segment"].values

    probabilities = best_model.predict_proba(X_all)[:, 1]
    scored["churn_probability"] = probabilities
    # 0.5 decision threshold for binary classification
    scored["churn_predicted"] = (probabilities >= 0.5).astype("int8")

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_PATH, index=False)

    # Document metrics
    _write_metrics_report(
        results=results,
        best_name=best_name,
        best_auc=best_auc,
        retention=len(labeled) / len(customers),
        feature_names=list(X.columns),
        best_params=forest_search.best_params_,
    )

    print(f"✅ Model saved to {MODEL_PATH.relative_to(ROOT)} ({best_name})")
    print(f"✅ Scored customers saved to {SCORED_PATH.relative_to(ROOT)}")
    print(f"✅ Metrics report saved to {METRICS_PATH.relative_to(ROOT)}")
    return best_model, results, scored


def main() -> None:
    """CLI entrypoint to run the model training and scoring pipeline."""
    if not RAW_CUSTOMERS_PATH.is_file():
        raise FileNotFoundError(
            f"Customer data is missing: {RAW_CUSTOMERS_PATH}. "
            "Run the data extraction first."
        )
    _, results, _ = run_training()
    best_auc = max(metrics["roc_auc"] for metrics in results.values())
    if best_auc < 0.75:
        print(f"🚨 Churn ROC-AUC target NOT MET: best holdout score is {best_auc:.3f}.")
    else:
        print(f"✅ Churn ROC-AUC target met: best holdout score is {best_auc:.3f}.")


if __name__ == "__main__":
    main()
