"""
feature_selection.py
====================
Step 4: Nonlinear feature selection via LightGBM wrapper.

The pipeline uses LightGBM's split-gain feature importance as a wrapper
selector. The procedure:

    1. One-hot encode all categorical columns into a dense matrix.
    2. Hold out 20% of the training data (stratified by target) to give
       LightGBM a validation set during fitting.
    3. Fit LightGBM on the remaining 80% with class-balanced weights.
    4. Rank all (now numeric) features by ``feature_importances_``.
    5. Keep the top N features by importance, drop the rest.

The 80/20 split is internal to this stage — it is *not* the final
modeling split. The downstream cross-validation in `modeling.py` uses
StratifiedKFold on the full feature-selected matrix. The split here
exists only to give the importance-fitting LightGBM a held-out
validation signal (which it doesn't actually use because no early
stopping is configured in the original, but the pattern is preserved
for fidelity).

The fitted ``ColumnTransformer`` (which holds the OneHotEncoder's
learned vocabulary) is returned so it can be reapplied to the test set
later — applying the same encoding to test without re-fitting is what
makes this leakage-safe.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: BUILD THE ONE-HOT ENCODER COLUMN TRANSFORMER
# =====================================================================

def build_categorical_encoder(X: pd.DataFrame) -> ColumnTransformer:
    """Build a fitted ColumnTransformer that one-hot encodes categoricals.

    Identifies all columns of pandas ``category`` dtype, configures a
    OneHotEncoder with ``handle_unknown='ignore'`` (so unseen categories
    in test data are encoded as all-zeros), and lets numeric columns
    pass through unchanged via ``remainder='passthrough'``.

    The transformer is returned unfitted — the caller fits it.

    Args:
        X: Training feature DataFrame (target already removed).

    Returns:
        Unfitted ColumnTransformer ready for ``fit_transform()``.
    """
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    logger.info("  Identified %d categorical columns for one-hot encoding", len(cat_cols))

    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        remainder="passthrough",
    )


# =====================================================================
# SECTION 2: FIT LIGHTGBM FOR IMPORTANCE SCORING
# =====================================================================

def fit_importance_lgb(
    X_enc: Any,
    y: pd.Series,
    *,
    lgb_params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
) -> LGBMClassifier:
    """Fit a LightGBM classifier whose feature importances drive selection.

    Splits the encoded matrix into 80/20 train/val (stratified by
    target), then fits LightGBM with the configured hyperparameters.
    The validation split is preserved for fidelity to the original
    notebook even though no early stopping is configured.

    Args:
        X_enc: Encoded feature matrix from the ColumnTransformer.
        y: Target Series (already int-cast).
        lgb_params: Hyperparameter dict (from PipelineConfig.fs_lgb_params).
        test_size: Holdout fraction for the internal split. Default 0.2.
        random_state: Seed for the split. Default 42.

    Returns:
        Fitted LGBMClassifier with ``feature_importances_`` populated.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X_enc, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    logger.info(
        "  Importance fit split — Train: %s | Val: %s",
        X_train.shape, X_val.shape,
    )

    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_train, y_train)
    return lgb


# =====================================================================
# SECTION 3: RANK FEATURES BY IMPORTANCE
# =====================================================================

def rank_features_by_importance(
    lgb: LGBMClassifier,
    feature_names: Any,
) -> pd.Series:
    """Rank all features by LightGBM split-gain importance, descending.

    Args:
        lgb: Fitted LGBMClassifier from `fit_importance_lgb()`.
        feature_names: Feature name array from the fitted ColumnTransformer.

    Returns:
        Series indexed by feature name, sorted descending by importance.
    """
    importances = pd.Series(
        lgb.feature_importances_, index=feature_names,
    ).sort_values(ascending=False)

    logger.info("Top 15 most important features:")
    for name, score in importances.head(15).items():
        logger.info("  %-50s  %s", name, score)

    return importances


# =====================================================================
# SECTION 4: APPLY SELECTION TO TRAIN + TEST
# =====================================================================

def apply_feature_selection(
    X_enc: Any,
    test_feature: pd.DataFrame,
    transformer: ColumnTransformer,
    feature_names: Any,
    y: pd.Series,
    top_feats: pd.Index,
    target_col: str = "subrogation",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce train + test matrices to the top-N selected features.

    The training matrix is wrapped back into a DataFrame using the
    transformer's feature names, then sliced to keep only the selected
    columns. The same transformer (already fitted on training) is
    applied to the test set, and the test matrix is sliced identically.

    Args:
        X_enc: Encoded training matrix from `transformer.fit_transform()`.
        test_feature: Raw test feature DataFrame (post-engineering, pre-encoding).
        transformer: Fitted ColumnTransformer from training.
        feature_names: Feature name array from transformer.
        y: Training target Series (re-attached to the reduced train DataFrame).
        top_feats: pd.Index of the selected feature names.
        target_col: Name of the target column to attach to train. Default 'subrogation'.

    Returns:
        Tuple ``(train_feature_reduced, test_feature_reduced)``. The
        train DataFrame has the target column appended; the test
        DataFrame is features-only.
    """
    # Reduce X to top features only
    X_reduced = pd.DataFrame(X_enc, columns=feature_names)[top_feats]
    train_feature_reduced = pd.concat(
        [X_reduced, y.reset_index(drop=True).rename(target_col)],
        axis=1,
    )

    # Apply same transformation to test data
    X_test_enc = transformer.transform(test_feature)
    test_feature_reduced = pd.DataFrame(X_test_enc, columns=feature_names)[top_feats]

    return train_feature_reduced, test_feature_reduced


# =====================================================================
# ORCHESTRATOR
# =====================================================================

def run_feature_selection(
    train_feature: pd.DataFrame,
    test_feature: pd.DataFrame,
    *,
    target_col: str,
    top_n: int,
    lgb_params: dict,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run Step 4 end-to-end: encode, fit importance LightGBM, select top-N.

    Args:
        train_feature: Training DataFrame with target column present.
        test_feature: Test DataFrame with target column absent.
        target_col: Name of the target column in train_feature.
        top_n: Number of features to keep by importance (default 60).
        lgb_params: Hyperparameters for the importance-fitting LightGBM.
        random_state: Seed for the internal 80/20 split.

    Returns:
        Tuple ``(train_feature_reduced, test_feature_reduced, importances)``.
        importances is the full Series of feature importances for the
        feature-importance.csv output deliverable.
    """
    logger.info("Step 4: LightGBM wrapper feature selection (top %d)", top_n)

    # Split target from features on training set
    y = train_feature[target_col].astype(int)
    X = train_feature.drop(columns=[target_col])

    # Build + fit encoder
    transformer = build_categorical_encoder(X)
    X_enc = transformer.fit_transform(X)
    feature_names = transformer.get_feature_names_out()
    logger.info("  Encoded feature matrix shape: %s", X_enc.shape)

    # Fit LightGBM and rank
    lgb = fit_importance_lgb(
        X_enc, y, lgb_params=lgb_params, random_state=random_state,
    )
    importances = rank_features_by_importance(lgb, feature_names)
    top_feats = importances.head(top_n).index

    # Apply selection to train and test
    train_feature_reduced, test_feature_reduced = apply_feature_selection(
        X_enc, test_feature, transformer, feature_names, y, top_feats,
        target_col=target_col,
    )

    logger.info(
        "  Selected %d features. Train: %s | Test: %s",
        len(top_feats), train_feature_reduced.shape, test_feature_reduced.shape,
    )

    return train_feature_reduced, test_feature_reduced, importances