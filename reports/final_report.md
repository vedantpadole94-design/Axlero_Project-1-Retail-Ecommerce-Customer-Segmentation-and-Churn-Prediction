# Axlero Project 1 — Final Report

## Executive Summary

The pipeline transforms the Week 1 customer summary into RFM metrics, four
K-Means segments, churn-risk scores, and dashboard exports. It retains all
4,338 customer-summary records and produces four distinct marketing audiences.
The selected Random Forest benchmark has a stratified holdout ROC-AUC of 0.783,
above the requested 0.75 threshold.

The churn result must be interpreted narrowly: churn is defined as recency
greater than 60 days at one snapshot, not as a customer's future behavior over
the next 30 days. The model excludes recency to avoid direct leakage, but a
time-based validation set and prospective outcome are required before using
probabilities as true future churn forecasts.

## Data Quality

- Input customer summary: 4,338 rows.
- RFM output: 4,338 customers; 100.0% retained.
- RFM snapshot date: 2011-12-10 (one day after the latest observed purchase).
- Line-item calculations exclude missing customer/date/order fields, returns
  or non-positive quantities, and non-positive unit prices.
- The customer summary used for this run is already filtered to valid positive
  quantity and price transactions by its SQL extraction query.

This 100.0% retention figure is measured from the customer-level Week 1
summary to the RFM dataset. It does not claim that every raw transaction was
retained during the original SQL cleaning and aggregation.

## RFM Methodology

For each customer:

- **Recency** is the whole-day difference between the snapshot and the last
  observed purchase.
- **Frequency** is the number of distinct invoices/orders.
- **Monetary** is the sum of quantity multiplied by unit price.

R, F, and M are log1p-transformed and standardized before K-Means. Candidate
cluster counts from 2 through 10 are compared using inertia and silhouette
score. The business model is fitted with four clusters, `random_state=42`, and
`n_init=10`. The diagnostic plot is
[`figures/optimal_k.png`](./figures/optimal_k.png).

## Segment Deep-Dive

| Segment | Customers | Share | Average R | Average F | Average M | Marketing recommendation |
|---|---:|---:|---:|---:|---:|---|
| Champions | 720 | 16.6% | 12.5 days | 13.7 orders | 8,037 | Reward loyalty with early access, VIP service, and referral incentives. |
| Promising | 877 | 20.2% | 20.6 days | 2.0 orders | 527 | Encourage a second purchase with a low-friction offer and personalized product recommendations. |
| At Risk (High Value) | 1,162 | 26.8% | 69.1 days | 4.1 orders | 1,813 | Trigger personal win-back outreach with a tailored incentive before lapse. |
| Hibernating | 1,579 | 36.4% | 187.7 days | 1.3 orders | 352 | Use low-cost reactivation campaigns; suppress or reduce spend if unresponsive. |

### Champions

The most valuable and most frequent group, with the highest average historical
revenue and low recency. Protect retention through VIP service, early access,
and referral programs rather than broad discounts.

### Promising

These customers purchased recently but have low order frequency and relatively
low value to date. Use post-purchase journeys, complementary product
recommendations, and a modest second-order incentive.

### At Risk (High Value)

This group has more historical order frequency and revenue than the
hibernating audience, with mean recency just beyond the 60-day churn
definition. Prioritize personalized outreach and measure incremental recovery
against a holdout group.

### Hibernating

These customers have the longest average recency and the lowest frequency and
historical value. Start with low-cost channels and suppress costly campaigns
when engagement does not return.

## Churn Model Evaluation

The label is `recency > 60 days`; the split is stratified, 80/20, with random
state 42. SMOTE is applied only to the training split. Two models were
evaluated on the untouched holdout:

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.761 | 0.622 | 0.784 | 0.694 | 0.687 |
| Random Forest | 0.783 | 0.631 | 0.817 | 0.712 | 0.700 |

The tuned Random Forest was selected, with `max_depth=6` and
`n_estimators=200`. Recency is deliberately not used as a predictor because it
directly defines the target. Additional features include average days between
orders, monetary value per order, and product diversity per order. ROC curves
are saved in `reports/figures/`.

## Business Recommendations

1. Route Champions to loyalty, early-access, and referral journeys.
2. Trigger second-purchase flows for Promising customers, with a cost cap.
3. Prioritize the At Risk (High Value) segment for personalized retention and
   track recovered revenue with a randomized campaign holdout.
4. Use inexpensive win-back attempts for Hibernating customers, with frequency
   caps and suppression after repeated non-response.
5. Treat model scores as prioritization signals for the current proxy label,
   not calibrated future churn probabilities, until a future-window outcome is
   available.

## Limitations & Next Steps

- A recency-threshold label describes current lapse, not future 30-day churn.
- A random split may mix customers from similar time periods. Train on earlier
  purchase windows and validate on a later period.
- The customer summary lacks per-order chronology, so order regularity and
  temporal behavioral changes are not modeled.
- Revenue is historical and does not account for margin, returns, contact
  cost, or campaign uplift.
- Calibrate risk thresholds and test interventions using randomized holdout
  groups before deployment.
- Refresh the pipeline on current client data; the included retail data ends
  in 2011 and is suitable for demonstration, not present-day targeting.
