import unittest

import pandas as pd

from src.features import compute_rfm, log1p_transform


class ComputeRfmTests(unittest.TestCase):
    def test_computes_rfm_from_transaction_rows(self):
        transactions = pd.DataFrame(
            {
                "customer_id": [1, 1, 1, 2],
                "invoice_no": ["A", "A", "B", "C"],
                "invoice_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-05", "2024-01-03"]
                ),
                "quantity": [2, 1, 1, 3],
                "unit_price": [5.0, 5.0, 10.0, 2.0],
            }
        )
        result, snapshot = compute_rfm(transactions)
        customer = result.set_index("customer_id").loc[1]
        self.assertEqual(snapshot, pd.Timestamp("2024-01-06"))
        self.assertEqual(customer["recency"], 1)
        self.assertEqual(customer["frequency"], 2)
        self.assertEqual(customer["monetary"], 25.0)

    def test_computes_rfm_from_customer_summary(self):
        summary = pd.DataFrame(
            {
                "customer_id": [1, 2],
                "last_purchase": pd.to_datetime(["2024-02-01", "2024-01-30"]),
                "total_orders": [4, 1],
                "total_revenue": [80.0, 5.0],
            }
        )
        result, snapshot = compute_rfm(summary)
        self.assertEqual(snapshot, pd.Timestamp("2024-02-02"))
        self.assertEqual(result.loc[0, "frequency"], 4)
        self.assertEqual(result.loc[1, "recency"], 3)

    def test_log_transform_rejects_negative_values(self):
        with self.assertRaises(ValueError):
            log1p_transform(pd.DataFrame({"recency": [-1]}), ("recency",))


if __name__ == "__main__":
    unittest.main()
