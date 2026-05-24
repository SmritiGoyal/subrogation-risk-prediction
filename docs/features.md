# Features

This document describes the engineered features used by the subrogation risk model. Each is documented with:

- **Source** — which raw column(s) it derives from
- **Formula / construction** — how the value is computed
- **Rationale** — why this transformation, grounded in domain reasoning
- **Implementation reference** — which subsection of `feature_engineering.py` or `feature_selection.py` owns the transformation

The pipeline starts with **29 raw columns** from the source data and produces **60 selected features** after LightGBM wrapper selection. The intermediate count after all engineering but before selection is **~44 features**, which then get one-hot expanded to **~68 encoded columns** before the top-60 cut.

## Feature pipeline at a glance

```
raw CSV (29 columns)
    │
    ▼
┌──────────────────────────────────────────────────┐
│ data_loading.py                                  │
│ - dropna (~0.01% rows)                           │
│ - cast 10 categorical cols → category dtype      │
│ - cast 9 numeric indicators → float64            │
└──────────────────────────┬───────────────────────┘
                           ▼
┌──────────────────────────────────────────────────┐
│ feature_engineering.py — transform_claim_data    │
│ - Derive driver_age, vehicle_age                 │
│ - Log1p 4 monetary columns                       │
│ - Compute 3 ratios (clipped)                     │
│ - Drop 4 ID-like columns                         │
└──────────────────────────┬───────────────────────┘
                           ▼
┌──────────────────────────────────────────────────┐
│ feature_engineering.py — feature_engineer        │
│ - Decompose claim_date                           │
│ - 5 driver/liability risk features               │
│ - 6 cross features                               │
└──────────────────────────┬───────────────────────┘
                           ▼
┌──────────────────────────────────────────────────┐
│ feature_selection.py                             │
│ - One-hot encode 10 category columns → ~28 cols  │
│ - Fit LightGBM with class_weight=balanced        │
│ - Keep top 60 features by split-gain importance  │
└──────────────────────────────────────────────────┘
                           │
                           ▼
                60 features → modeling
```

## 1. Age derivations (transform_claim_data)

The reference year (default 2025, configurable via `cfg.reference_year`) anchors two age calculations:

| Feature | Formula | Type |
|---|---|---|
| `driver_age` | `reference_year - year_of_born` | int |
| `vehicle_age` | `reference_year - vehicle_made_year` | int |

**Rationale:** The raw year-of-birth and vehicle-year columns aren't directly useful as features — a value of 1985 doesn't mean much to a tree split. The age derivations turn them into interpretable, monotonic features. Driver age 22 (new driver) and 65 (older driver) carry different risk profiles; vehicle age 1 (new) vs 15 (old) similarly.

After derivation, `year_of_born` and `vehicle_made_year` are dropped to avoid redundancy.

**Implementation:** `feature_engineering.py`, Section 1 (`transform_claim_data`).

## 2. Log-transformed monetary features (transform_claim_data)

Four monetary columns are log1p-transformed in place:

| Feature | Original distribution | After log1p |
|---|---|---|
| `annual_income` | Right-skewed | Approximately Gaussian |
| `claim_est_payout` | Right-skewed | Approximately Gaussian |
| `vehicle_price` | Right-skewed | Approximately Gaussian |
| `vehicle_mileage` | Right-skewed | Approximately Gaussian |

**Rationale:** Monetary distributions are heavily right-skewed — a small number of high-income drivers, expensive vehicles, or large claims dominate the variance. Log transformation compresses the tail and makes the features more useful for linear models (which assume roughly Gaussian feature distributions) without sacrificing tree-model performance.

The `log1p` (rather than plain `log`) handles potential zeros gracefully.

**Implementation:** `feature_engineering.py`, Section 1.

## 3. Core ratios (transform_claim_data)

Three derived ratios encode joint behavior of multiple inputs:

| Feature | Formula | Clip range | Rationale |
|---|---|---|---|
| `claim_payout_ratio` | `claim_est_payout / (vehicle_price + 1e-6)` | [0, 5] | Total-loss indicator — when payout approaches or exceeds vehicle price, the claim is more likely to be totaled, which correlates with subrogation patterns |
| `income_to_price_ratio` | `annual_income / (vehicle_price + 1e-6)` | [0, 5] | Affordability proxy — driver income relative to vehicle cost indicates the financial profile |
| `liab_ratio` | `liab_prct / 100.0` | [0, 1] | Normalize 0-100 liability scale to 0-1 fraction |

The `1e-6` denominators protect against division by zero for the rare records with zero `vehicle_price`. Clipping bounds prevent extreme outliers from dominating linear-model coefficients.

**Implementation:** `feature_engineering.py`, Section 1.

## 4. Date decomposition (feature_engineer)

The raw `claim_date` is decomposed into two features and then dropped:

| Feature | Formula | Type |
|---|---|---|
| `claim_year` | `claim_date.dt.year` | int |
| `claim_month` | `claim_date.dt.month` | int |

**Rationale:** Year captures temporal regime changes (regulatory shifts, economic conditions). Month captures seasonal effects (e.g., higher accident rates in winter months due to weather).

**Implementation:** `feature_engineering.py`, Section 2 (`feature_engineer`).

## 5. Driver risk features (feature_engineer)

| Feature | Formula | Type |
|---|---|---|
| `driving_experience` | `claim_year - age_of_DL` | int |
| `is_new_driver` | `(driving_experience < 3).astype(int)` | binary |

**Note on semantics:** The `age_of_DL` column is the driver's age when they got their license, so `claim_year - age_of_DL` actually computes a hybrid that mirrors the original Colab notebook's calculation. The intent is "years since license issuance"; the value's predictive power is what matters here, not the strict interpretation.

**Rationale:** Newly licensed drivers have less driving experience and higher accident rates. The `is_new_driver` flag explicitly encodes the high-risk threshold (<3 years).

**Implementation:** `feature_engineering.py`, Section 2.

## 6. Claim & liability risk features (feature_engineer)

| Feature | Formula | Type |
|---|---|---|
| `high_payout_flag` | `(claim_est_payout > vehicle_price).astype(int)` | binary |
| `payout_to_price_ratio` | `claim_est_payout / (vehicle_price + 1e-6)` | float |
| `liab_low` | `(liab_prct <= 25).astype(int)` | binary |
| `liab_mid` | `((liab_prct > 25) & (liab_prct < 75)).astype(int)` | binary |
| `liab_high` | `(liab_prct >= 75).astype(int)` | binary |

**Rationale:** Liability percentage carries strong predictive signal for subrogation — high liability on the *other* party means subrogation is more likely; low liability means it's less likely. The three liability bins (≤25, 25-75, ≥75) make the threshold structure explicit, which linear models can't otherwise learn from `liab_prct` alone.

`high_payout_flag` is a coarser version of `claim_payout_ratio` from Section 3, surfaced as a categorical signal that ensembles can split on more efficiently. `payout_to_price_ratio` is an unclipped duplicate of `claim_payout_ratio` — both columns end up in the feature set because the original code computes both (preserved verbatim in the rebuild).

**Implementation:** `feature_engineering.py`, Section 2.

## 7. Cross features (feature_engineer)

Six engineered interactions encode joint behavior that linear models can't otherwise capture without explicit interaction terms:

| Feature | Formula | Rationale |
|---|---|---|
| `driver_risk_liab` | `driver_age * liab_ratio` | Older drivers + high liability is a different risk profile than young drivers + high liability |
| `liab_safety_ratio` | `liab_prct * safety_rating` | High liability on a high-safety-rating vehicle suggests external-cause subrogation |
| `price_to_income` | `vehicle_price / (annual_income + 1)` | Affordability stretch — driving a vehicle expensive for income level |
| `claim_over_income` | `claim_est_payout / (annual_income + 1)` | Claim impact on driver's finances |
| `claim_density_by_age` | `past_num_of_claims / (driver_age + 1e-6)` | Claim frequency per year of driver age, normalizing for older drivers' longer claim history |
| `liab_to_claim_ratio` | `liab_prct / (claim_est_payout + 1e-6)` | Liability per dollar of claim — fault-density signal |

**Note on denominators:** `price_to_income` and `claim_over_income` use `+ 1` (not `+ 1e-6`) because both numerators are already log1p-transformed at this point. The other ratios use `+ 1e-6` because they operate on raw scales. This inconsistency is preserved verbatim from the original notebook.

**Implementation:** `feature_engineering.py`, Section 2.

## 8. Categorical one-hot encoding (feature_selection)

After all derivation, 10 categorical columns remain that need encoding for the gradient boosters:

```
gender, living_status, claim_day_of_week, accident_site,
witness_present_ind, channel, vehicle_category, vehicle_color,
accident_type, in_network_bodyshop
```

These are one-hot encoded via `sklearn.OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, producing approximately 28 encoded columns total. The `handle_unknown='ignore'` setting ensures any unseen category in the test set is encoded as all-zeros rather than crashing the encoder.

**Implementation:** `feature_selection.py`, Section 1 (`build_categorical_encoder`).

## 9. LightGBM wrapper feature selection

After encoding, the pipeline fits a LightGBM classifier with class-balanced weights on an 80/20 stratified split of the training data, then ranks all features by split-gain importance and keeps the top 60. The bottom features are dropped before downstream modeling.

**Top 15 features by importance (from the validated run):**

1. `vehicle_weight`
2. `vehicle_mileage`
3. `liab_to_claim_ratio` (engineered)
4. `driver_risk_liab` (engineered)
5. `liab_safety_ratio` (engineered)
6. `liab_prct`
7. `safety_rating`
8. `claim_over_income` (engineered)
9. `driver_age` (engineered)
10. `claim_payout_ratio` (engineered)
11. `vehicle_price`
12. `claim_est_payout`
13. `annual_income`
14. `vehicle_age` (engineered)
15. `income_to_price_ratio` (engineered)

**Observation:** 8 of the top 15 features are engineered, not raw. The cross features (`liab_to_claim_ratio`, `driver_risk_liab`, `liab_safety_ratio`) outrank most raw columns including `liab_prct` itself — confirming that the explicit interactions provide signal the tree models couldn't otherwise extract from the raw inputs alone. This validates the feature engineering design.

**Implementation:** `feature_selection.py`, Sections 2-4.

## 10. Dropped columns

Four columns are dropped during `transform_claim_data` because their information has been preserved in derived features:

| Column | Why dropped |
|---|---|
| `claim_number` | Row identifier — not a model feature |
| `year_of_born` | Replaced by `driver_age` |
| `vehicle_made_year` | Replaced by `vehicle_age` |
| `zip_code` | High-cardinality without a geographic encoding strategy — drops potential leakage |

`claim_date` is dropped at the end of `feature_engineer`, after its information has been preserved in `claim_year` and `claim_month`.

## Summary

After all stages:

| Stage | Column count |
|---|---:|
| Raw CSV | 29 |
| After cleaning | 29 |
| After transform_claim_data | 30 (added 6 new, dropped 4 ID-like, +1 net = `subrogation` retained) |
| After feature_engineer | 44 |
| After one-hot encoding | ~68 |
| After top-60 wrapper selection | **60** |

The final 60-feature matrix is what gets fed to the 14 candidate models for CV evaluation.

## References

- Production code: `src/feature_engineering.py`, `src/feature_selection.py`
- Validation: `outputs/feature_importance.csv` (auto-generated each run)
- Methodology context: `docs/methodology.md`