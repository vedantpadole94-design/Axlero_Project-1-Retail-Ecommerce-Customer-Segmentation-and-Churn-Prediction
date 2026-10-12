-- Axlero Project 1 — SQL Extraction Queries

-- name: user_level_summary
SELECT
    customer_id,
    MIN(invoice_date)                          AS first_purchase,
    MAX(invoice_date)                          AS last_purchase,
    EXTRACT(DAY FROM (MAX(invoice_date) - MIN(invoice_date)))::INT
                                               AS tenure_days,
    COUNT(DISTINCT invoice_no)                 AS total_orders,
    SUM(quantity * unit_price)                 AS total_revenue,
    ROUND((SUM(quantity * unit_price) / NULLIF(COUNT(DISTINCT invoice_no), 0))::numeric, 2)
                                               AS avg_order_value,
    COUNT(DISTINCT stock_code)                 AS distinct_products,
    COUNT(DISTINCT country)                    AS distinct_countries,
    SUM(quantity)                              AS total_quantity,
    MAX(country)                               AS primary_country
FROM transactions
WHERE customer_id IS NOT NULL
  AND quantity > 0
  AND unit_price > 0
GROUP BY customer_id
ORDER BY total_revenue DESC;

-- name: monthly_revenue_trend
SELECT
    DATE_TRUNC('month', invoice_date)          AS month,
    SUM(quantity * unit_price)                 AS revenue,
    COUNT(DISTINCT invoice_no)                 AS orders,
    COUNT(DISTINCT customer_id)                AS active_customers
FROM transactions
WHERE quantity > 0
  AND unit_price > 0
GROUP BY DATE_TRUNC('month', invoice_date)
ORDER BY month;

-- name: top_countries
SELECT
    country,
    COUNT(DISTINCT customer_id)                AS customers,
    SUM(quantity * unit_price)                 AS revenue
FROM transactions
WHERE quantity > 0
  AND unit_price > 0
GROUP BY country
ORDER BY revenue DESC
LIMIT 15;
