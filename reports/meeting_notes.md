# Meeting Notes — Week 1 & Week 2 Review

## Week 1 — Data Ingestion & EDA
- Loaded 541,909 transactions into PostgreSQL
- Extracted 4,338 unique customers
- Data retention: 100% (target: 95%)
- Created 4 EDA charts: revenue trend, top countries, distributions
- Notebook: notebooks/1.0-eda.ipynb

## Week 2 — RFM Segmentation & K-Means
- Computed Recency, Frequency, Monetary per customer
- Applied K-Means clustering (k=4 via Elbow + Silhouette)
- Produced 4 segments: Champions, Promising, At Risk, Hibernating
- Key insight: 16% of customers drive 64% of revenue
- Notebook: notebooks/2.0-rfm-clustering.ipynb

## Key Numbers
| Metric | Value |
|---|---|
| Raw transactions | 541,909 |
| Unique customers | 4,338 |
| Champions revenue share | 64% |
| Hibernating churn rate | 91% |
| Data retention | 100% |

## Next Steps (Week 3 & 4)
- Churn prediction model (target ROC-AUC >= 0.75)
- Live Streamlit dashboard deployment
