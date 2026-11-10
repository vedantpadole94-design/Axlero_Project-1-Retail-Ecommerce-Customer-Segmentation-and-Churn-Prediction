# Code Style & Documentation Guide

This document establishes the documentation and coding standards for the **Axlero Customer Intelligence & Churn Prediction** codebase. All contributors and maintainers must adhere to these conventions.

---

## 1. Guiding Philosophy

Every module, class, and function should be **self-documenting**:
- **WHAT**: Clearly define what the routine performs.
- **WHY**: Explain the business logic, financial implications, and data science rationale behind decisions (e.g. *Why 60 days for churn? Why log1p? Why k=4?*).
- **INPUTS & OUTPUTS**: Strictly document parameter types, constraints, return structures, and possible exceptions.
- **EDGE CASES**: Document how null values, missing identifiers, zero spend, or class imbalances are mitigated.

---

## 2. Docstring Standards (PEP 257 + Google Style)

### Module-Level Docstrings
Every Python file must begin with a multi-line module docstring situated before any imports. It must include:
1. A concise one-line summary ending in a period.
2. A description of the module's role within the broader analytical pipeline.
3. An ASCII or text representation of pipeline context.
4. Typical usage examples.

```python
"""Build customer-level RFM features from transaction or summary data.

This module computes core Recency, Frequency, and Monetary (RFM) behavioral
metrics for e-commerce customers, forming the analytical backbone for both
unsupervised customer clustering (K-Means) and supervised churn modeling.

Pipeline Context:
    make_dataset.py -> features.py -> segmentation.py & model.py

Typical usage:
    from src.features import compute_rfm, log1p_transform
    rfm_df, snapshot_date = compute_rfm(source_df)
"""
```

### Function & Method Docstrings
Every function must follow the Google style docstring format containing:
- **One-line summary**: Active imperative verb ending in a period.
- **Detailed description / Business rationale**: Background context, anti-leakage considerations, or mathematical reasoning.
- **Args**: Each argument with its expected type and constraints.
- **Returns**: Structure, type, and meaning of returned values.
- **Raises**: Explicit exceptions raised under failure conditions.
- **Example**: Executable doctest-compatible usage demonstration.

```python
def compute_rfm(
    df: pd.DataFrame,
    snapshot_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Compute recency, purchase frequency, and monetary value per customer.

    Business Definition of RFM Metrics:
        - Recency (R): Days elapsed between latest order and snapshot date.
        - Frequency (F): Distinct completed order count.
        - Monetary (M): Cumulative gross revenue across all transactions.

    Args:
        df: Input DataFrame containing raw transaction lines or summary records.
        snapshot_date: Reference date for recency calculation. Defaults to max purchase + 1 day.

    Returns:
        tuple[pd.DataFrame, pd.Timestamp]:
            - pd.DataFrame: RFM metrics per customer.
            - pd.Timestamp: Reference snapshot timestamp used.

    Raises:
        ValueError: If 'customer_id' is missing or inputs are empty after cleaning.

    Example:
        >>> rfm, snapshot = compute_rfm(source_df)
        >>> print(rfm.columns.tolist())
        ['customer_id', 'recency', 'frequency', 'monetary']
    """
```

---

## 3. Inline Commenting Rules

### When to Add Inline Comments
Add inline comments (`# ...`) before non-obvious lines to clarify:
- **Business Cutoffs & Magic Numbers**: Document why a threshold is chosen (e.g. `CHURN_THRESHOLD_DAYS = 60` — *in retail, 60 days of inactivity indicates a lapsed account*).
- **Anti-Leakage Safeguards**: Document why features like recency are excluded from predictor matrices during churn modeling.
- **Statistical Transformations**: Document why `np.log1p` is preferred over `np.log` (compressing right-skewed revenue tails while safely mapping zero spend to 0).
- **Complex Aggregations**: Document multi-stage merges, cross-tabulations, and cohort retention indexing.
- **Section Headers**: Group logical phases in files exceeding 100 lines using:
  ```python
  # ─────────────────────────────────────────────────────────────────────────────
  # SECTION: <Purpose of block>
  # ─────────────────────────────────────────────────────────────────────────────
  ```

### What NOT to Comment
- Trivial assignments (`x = 1`, `count = len(df)`).
- Standard imports (`import pandas as pd`).
- Self-explanatory loops and standard library operations.

---

## 4. Code Quality & Verification Commands

Before submitting pull requests or merging changes into `main`, verify all modules:

```bash
# 1. Verify module imports
python -c "from src.features import compute_rfm; print('features OK')"
python -c "from src.model import load_and_label; print('model OK')"
python -c "from src.dashboard_utils import CUSTOM_CSS; print('dashboard OK')"
python -c "from src.segmentation import fit_rfm_segments; print('segmentation OK')"

# 2. Execute unit tests
python -m unittest discover tests/

# 3. Test Streamlit application locally
streamlit run app.py
```
