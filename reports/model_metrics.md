# Churn Model Metrics

- Customer retention after feature preparation: 100.0%
- Churn label: recency greater than 60 days.
- Recency is intentionally excluded from predictors because it defines the target.
- Selected model: Random Forest
- Selected holdout ROC-AUC: 0.783
- ROC-AUC target (0.75): MET
- Predictors: frequency, monetary, tenure_days, avg_order_value, distinct_products, distinct_countries, total_quantity, avg_days_between_orders, monetary_per_order, product_diversity

## Held-out metrics

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.761 | 0.622 | 0.784 | 0.694 | 0.687 |
| Random Forest | 0.783 | 0.631 | 0.817 | 0.712 | 0.700 |

Random forest best parameters: `{'max_depth': 6, 'n_estimators': 200}`

## Interpretation and limitations

The current churn outcome is a recency threshold at a single snapshot, not observed future churn. As a result, these results measure how well past purchase aggregates distinguish currently lapsed customers; they do not establish prospective 30-day churn performance. Recency is excluded from predictors to avoid direct target leakage. A time-based holdout and future-window outcome should be used before operational deployment.
