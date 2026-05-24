"""
data_loading.py
===============
Step 1: Load and clean the raw competition CSVs.

Handles two concerns separated cleanly:

    1. Loading        — read training and test CSVs from disk.
    2. Cleaning       — normalize datatypes, parse dates, drop missing
                        rows, and identify the row ID column.

The cleaning is intentionally minimal because the source data has only
~0.01% missing values overall (2 missing entries per feature). Heavier
imputation isn't justified at this missingness rate — a simple dropna
preserves nearly the entire dataset.

Categorical columns are cast to pandas `category` dtype for memory
efficiency and to enable native category handling in tree models
downstream. Numeric indicator columns are cast to float64 so the
downstream sklearn pipelines accept them without dtype warnings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# =====================================================================
# COLUMN GROUPS
# =====================================================================

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "gender", "living_status", "claim_day_of_week", "accident_site",
    "witness_present_ind", "channel", "vehicle_category",
    "vehicle_color", "accident_type", "in_network_bodyshop",
)
"""Columns cast to pandas `category` dtype during cleaning."""

NUMERIC_INDICATOR_COLUMNS: tuple[str, ...] = (
    "year_of_born", "email_or_tel_available", "high_education_ind",
    "address_change_ind", "past_num_of_claims", "policy_report_filed_ind",
    "age_of_DL", "vehicle_made_year", "subrogation",
)
"""Columns cast to float64 during cleaning."""


# =====================================================================
# SECTION 1: LOAD RAW CSVS
# =====================================================================

def load_raw_data(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and test CSVs from disk.

    Args:
        train_path: Path to the training CSV.
        test_path: Path to the test CSV.

    Returns:
        Tuple ``(train_df, test_df)`` with the raw, untransformed rows.
    """
    logger.info("Loading training data from %s", train_path)
    train = pd.read_csv(train_path)
    logger.info("  Train shape: %s", train.shape)

    logger.info("Loading test data from %s", test_path)
    test = pd.read_csv(test_path)
    logger.info("  Test shape:  %s", test.shape)

    return train, test


def identify_id_column(df: pd.DataFrame) -> str:
    """Return the row identifier column name.

    Looks for ``claim_number`` first (the original competition column),
    falls back to ``claim_nbr`` if that's not present. Raises if neither
    is found.

    Args:
        df: A DataFrame from `load_raw_data()`.

    Returns:
        The identifier column name as a string.

    Raises:
        KeyError: If neither candidate column is present.
    """
    if "claim_number" in df.columns:
        return "claim_number"
    if "claim_nbr" in df.columns:
        return "claim_nbr"
    raise KeyError(
        "Could not find row identifier — expected 'claim_number' or 'claim_nbr'"
    )


def log_missingness(df: pd.DataFrame, top_n: int = 30) -> None:
    """Log the top ``top_n`` columns by missingness rate.

    Args:
        df: DataFrame to inspect.
        top_n: Number of columns to log. Defaults to 30.
    """
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    if missing.max() == 0:
        logger.info("No missing values detected across %d columns inspected", top_n)
        return

    logger.info("Top %d columns by missingness:", top_n)
    for col, rate in missing.items():
        if rate > 0:
            logger.info("  %-30s  %.2f%% missing", col, 100 * rate)


# =====================================================================
# SECTION 2: CLEAN DATATYPES + DATES
# =====================================================================

def clean_claim_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize core datatypes before transformation.

    Steps:
        1. Drop rows with any null value (~0.01% of the data).
        2. Parse `claim_date` to datetime, coercing invalid values to NaT.
        3. Cast `CATEGORICAL_COLUMNS` to pandas `category` dtype.
        4. Cast `NUMERIC_INDICATOR_COLUMNS` to float64.

    Args:
        df: Raw DataFrame from `load_raw_data()`.

    Returns:
        Cleaned DataFrame with normalized dtypes and reset index.

    Notes:
        Operates on a copy — input DataFrame is not mutated.

        The `dropna` is aggressive (any null in any column drops the row),
        which is appropriate here because the source data has minimal
        missingness. For datasets with heavier missingness this would
        need to be replaced with column-specific imputation.
    """
    df = df.copy()

    # Drop missing values (for this dataset, minimal NA — ~0.01%)
    df = df.dropna().reset_index(drop=True)

    # Convert date columns
    if "claim_date" in df.columns:
        df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

    # Convert categorical features
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Convert numeric indicators to float64
    for col in NUMERIC_INDICATOR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("float64")

    return df


# =====================================================================
# ORCHESTRATOR
# =====================================================================

def run_data_loading(
    train_path: Path,
    test_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run the full data-loading pipeline end-to-end.

    Loads both CSVs, logs missingness diagnostics on the training set,
    cleans both DataFrames, and identifies the row ID column.

    Args:
        train_path: Path to the training CSV.
        test_path: Path to the test CSV.

    Returns:
        Tuple ``(train_cleaned, test_cleaned, id_col)`` ready for
        feature engineering.
    """
    train, test = load_raw_data(train_path, test_path)

    log_missingness(train)

    id_col = identify_id_column(train)
    logger.info("Identifier column: %s", id_col)

    train_cleaned = clean_claim_data(train)
    test_cleaned = clean_claim_data(test)
    logger.info(
        "After cleaning — Train: %s | Test: %s",
        train_cleaned.shape, test_cleaned.shape,
    )

    return train_cleaned, test_cleaned, id_col