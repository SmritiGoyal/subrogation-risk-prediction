"""
feature_engineering.py
======================
Steps 2-3: Transform raw claim data into the modeling feature set.

Two concerns, kept in separate functions to mirror the original
pipeline structure:

    1. transform_claim_data — log-scale monetary variables, derive age
                              metrics, compute the three core ratios
                              (claim_payout_ratio, income_to_price_ratio,
                              liab_ratio), and drop ID-like columns.
                              Idempotent on already-transformed data.

    2. feature_engineer     — decompose claim_date, add domain-driven
                              risk metrics (is_new_driver,
                              high_payout_flag, liability bins), and
                              compute 6 cross features that capture
                              joint behavior the linear models can't
                              learn alone.

The two functions are sequential — `transform_claim_data` is run first,
then `feature_engineer` consumes its output. Both operate on a copy and
do not mutate their input.

The reference year (used for `driver_age` and `vehicle_age`) is taken
from `PipelineConfig.reference_year` (default 2025). The original Colab
notebook hardcoded 2025 inline; lifting this to config makes the
pipeline forward-deployable to future years without code edits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# COLUMN GROUPS
# =====================================================================

LOG_TRANSFORM_COLUMNS: tuple[str, ...] = (
    "annual_income", "claim_est_payout", "vehicle_price", "vehicle_mileage",
)
"""Skewed monetary columns transformed in-place via log1p."""

ID_LIKE_COLUMNS_TO_DROP: tuple[str, ...] = (
    "claim_number", "year_of_born", "vehicle_made_year", "zip_code",
)
"""Columns dropped after their derived counterparts are computed.

`claim_number` is the row identifier (not a feature). `year_of_born`
and `vehicle_made_year` are dropped because they are replaced by
`driver_age` and `vehicle_age`. `zip_code` is dropped because it
introduces high-cardinality leakage potential without a corresponding
geographic encoding strategy.
"""


# =====================================================================
# SECTION 1: TRANSFORM CLAIM DATA (STEP 2)
# =====================================================================

def transform_claim_data(
    df: pd.DataFrame,
    reference_year: int = 2025,
) -> pd.DataFrame:
    """Add log-transforms and ratio-based derived features.

    Stage 1 of the feature pipeline. Runs five transformations in
    order:

        1. Derive driver_age and vehicle_age from year_of_born and
           vehicle_made_year using ``reference_year`` as the anchor.
        2. Log1p-transform the four skewed monetary columns.
        3. Compute three core ratios: claim_payout_ratio,
           income_to_price_ratio, liab_ratio.
        4. Clip each ratio to reasonable bounds (5x for payout/income,
           1.0 for liability fraction).
        5. Drop ID-like columns whose information has been preserved
           in derived features.

    Args:
        df: Cleaned DataFrame from `data_loading.clean_claim_data()`.
        reference_year: Year used for age derivations. Original
            competition used 2025. Lifted from hardcode to parameter
            for forward-deployment to future inference years.

    Returns:
        DataFrame with the original columns replaced by their
        log-transformed / derived versions, plus the three new ratios.

    Notes:
        Operates on a copy. The 1e-6 denominator add in each ratio
        protects against division by zero for the (rare) records with
        zero vehicle_price; the result is a very large number that
        then gets clipped to the [0, 5] or [0, 1] range.
    """
    df = df.copy()

    # Derived age metrics
    df["driver_age"] = reference_year - df["year_of_born"]
    df["vehicle_age"] = reference_year - df["vehicle_made_year"]

    # Log-transform skewed monetary variables
    for col in LOG_TRANSFORM_COLUMNS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Derived ratios
    df["claim_payout_ratio"] = df["claim_est_payout"] / (df["vehicle_price"] + 1e-6)
    df["income_to_price_ratio"] = df["annual_income"] / (df["vehicle_price"] + 1e-6)
    df["liab_ratio"] = df["liab_prct"] / 100.0

    # Clip to prevent extreme outliers
    df["claim_payout_ratio"] = df["claim_payout_ratio"].clip(0, 5)
    df["income_to_price_ratio"] = df["income_to_price_ratio"].clip(0, 5)
    df["liab_ratio"] = df["liab_ratio"].clip(0, 1)

    # Drop redundant or ID columns
    df = df.drop(
        columns=[c for c in ID_LIKE_COLUMNS_TO_DROP if c in df.columns],
        errors="ignore",
    )

    return df


# =====================================================================
# SECTION 2: FEATURE ENGINEERING (STEP 3)
# =====================================================================

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-driven features capturing risk and payout behavior.

    Stage 2 of the feature pipeline. Runs four feature families:

        1. Datetime decomposition — extract claim_year, claim_month
           from claim_date (if present), then drop claim_date itself.
        2. Driver risk metrics — driving_experience, is_new_driver.
        3. Liability bins — liab_low / liab_mid / liab_high binary
           flags around the conventional 25%/75% liability cutoffs.
        4. Cross features — six engineered interactions encoding joint
           behavior (driver_risk_liab, liab_safety_ratio,
           price_to_income, claim_over_income, claim_density_by_age,
           liab_to_claim_ratio) that linear models can't otherwise
           capture without explicit interaction terms.

    Args:
        df: DataFrame from `transform_claim_data()`.

    Returns:
        DataFrame with all original columns plus the engineered
        features. `claim_date` is dropped (its information lives on
        in claim_year and claim_month).

    Notes:
        Operates on a copy. The 1e-6 denominator add protects against
        division by zero for the small set of records with zero
        denominator.

        ``driving_experience = claim_year - age_of_DL`` is interpreted
        as "years since driver's license issuance" — `age_of_DL` is
        already an age, so this computes "the year_of_issue" then
        subtracts it from `claim_year`. The semantics here mirror the
        original Colab notebook verbatim.
    """
    df = df.copy()

    # Datetime decomposition
    if "claim_date" in df.columns:
        df["claim_year"] = df["claim_date"].dt.year
        df["claim_month"] = df["claim_date"].dt.month

    # Driver & risk metrics
    df["driving_experience"] = df["claim_year"] - df["age_of_DL"]
    df["is_new_driver"] = (df["driving_experience"] < 3).astype(int)

    # Claim & liability metrics
    df["high_payout_flag"] = (df["claim_est_payout"] > df["vehicle_price"]).astype(int)
    df["payout_to_price_ratio"] = df["claim_est_payout"] / (df["vehicle_price"] + 1e-6)
    df["liab_low"] = (df["liab_prct"] <= 25).astype(int)
    df["liab_mid"] = ((df["liab_prct"] > 25) & (df["liab_prct"] < 75)).astype(int)
    df["liab_high"] = (df["liab_prct"] >= 75).astype(int)

    # Cross features
    df["driver_risk_liab"] = df["driver_age"] * df["liab_ratio"]
    df["liab_safety_ratio"] = df["liab_prct"] * df["safety_rating"]
    df["price_to_income"] = df["vehicle_price"] / (df["annual_income"] + 1)
    df["claim_over_income"] = df["claim_est_payout"] / (df["annual_income"] + 1)
    df["claim_density_by_age"] = df["past_num_of_claims"] / (df["driver_age"] + 1e-6)
    df["liab_to_claim_ratio"] = df["liab_prct"] / (df["claim_est_payout"] + 1e-6)

    # Drop irrelevant columns
    df = df.drop(columns=["claim_date"], errors="ignore")

    return df


# =====================================================================
# ORCHESTRATOR
# =====================================================================

def run_feature_engineering(
    train_cleaned: pd.DataFrame,
    test_cleaned: pd.DataFrame,
    reference_year: int = 2025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Steps 2-3 end-to-end on the cleaned training and test sets.

    Args:
        train_cleaned: Cleaned training DataFrame from data_loading.
        test_cleaned: Cleaned test DataFrame from data_loading.
        reference_year: Year used for age derivations. See
            `transform_claim_data()` for details.

    Returns:
        Tuple ``(train_feature, test_feature)`` with all engineered
        features applied identically to both sets.
    """
    logger.info("Step 2: transform_claim_data — log scales, ratios, age derivations")
    train_transformed = transform_claim_data(train_cleaned, reference_year)
    test_transformed = transform_claim_data(test_cleaned, reference_year)
    logger.info(
        "  After transform — Train: %s | Test: %s",
        train_transformed.shape, test_transformed.shape,
    )

    logger.info("Step 3: feature_engineer — driver/liab/cross features")
    train_feature = feature_engineer(train_transformed)
    test_feature = feature_engineer(test_transformed)
    logger.info(
        "  After feature engineering — Train: %s | Test: %s",
        train_feature.shape, test_feature.shape,
    )

    return train_feature, test_feature