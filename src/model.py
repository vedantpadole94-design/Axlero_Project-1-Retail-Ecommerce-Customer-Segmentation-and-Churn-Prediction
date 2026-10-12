"""Train and evaluate customer churn classifiers."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

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
    """Join customer-level predictors to the requested recency-based label."""
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
    labeled["churn"] = (labeled["recency"] > churn_threshold_days).astype("int8")
    if labeled["churn"].nunique() != 2:
        raise ValueError(
            "Churn labels contain only one class; a stratified model split is not possible."
        )
    return labeled


def build_features(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create leakage-aware predictors, excluding recency and the target itself."""
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
    data["avg_days_between_orders"] = tenure / (frequency - 1).clip(lower=1)
    data["monetary_per_order"] = monetary / frequency
    data["product_diversity"] = distinct_products / frequency

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
    """Stratify a holdout split, impute using train medians, then SMOTE train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
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
    """Fit a standardized, class-weighted logistic regression classifier."""
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
    """Tune a random forest for ROC-AUC with the requested grid and 3-fold CV."""
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
    """Print holdout metrics and save the model's ROC curve."""
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


def _write_metrics_report(
    results: dict[str, dict[str, float]],
    best_name: str,
    best_auc: float,
    retention: float,
    feature_names: list[str],
    best_params: dict[str, object] | None,
) -> None:
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
    """Run data prep, both models, selection, scoring, and artifact creation."""
    customers = pd.read_parquet(RAW_CUSTOMERS_PATH)
    rfm = pd.read_parquet(RFM_PATH) if RFM_PATH.is_file() else None
    labeled = load_and_label(customers, rfm)
    X, y = build_features(labeled)
    if len(X) != len(customers):
        raise RuntimeError(
            f"Unexpected customer loss: {len(X):,} feature rows for "
            f"{len(customers):,} source rows."
        )
    X_train, X_test, y_train, y_test, scoring_imputer = split_and_balance(X, y)

    logistic = train_logistic(X_train, y_train)
    logistic_metrics = evaluate(logistic, X_test, y_test, "Logistic Regression")
    forest_search = train_random_forest(X_train, y_train)
    forest = forest_search.best_estimator_
    forest_metrics = evaluate(forest, X_test, y_test, "Random Forest")
    results = {
        "Logistic Regression": logistic_metrics,
        "Random Forest": forest_metrics,
    }

    candidates = {
        "Logistic Regression": (logistic, logistic_metrics["roc_auc"]),
        "Random Forest": (forest, forest_metrics["roc_auc"]),
    }
    best_name = max(candidates, key=lambda name: candidates[name][1])
    best_model, best_auc = candidates[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": best_model,
            "feature_names": list(X.columns),
            "churn_threshold_days": CHURN_THRESHOLD_DAYS,
            "selected_model": best_name,
            "imputer": scoring_imputer,
        },
        MODEL_PATH,
    )

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
    scored["churn_predicted"] = (probabilities >= 0.5).astype("int8")
    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(SCORED_PATH, index=False)

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
