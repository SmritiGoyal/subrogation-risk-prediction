"""
modeling.py
===========
Step 5: Model construction, CV strategy selection, and CV evaluation.

This module owns three concerns:

    1. Model construction   — the 7 base models (LR, DT, RF, XGB, LGB,
                              CatBoost, KNN) plus 7 stacked/blended
                              ensembles, all instantiated from
                              PipelineConfig hyperparameters.
    2. CV strategy selector — picks the appropriate sklearn CV splitter
                              based on cfg.cv_mode (strat5, strat10,
                              repeated, group, timeseries).
    3. CV evaluation loop   — per-fold fit + predict + F1 threshold
                              tuning + ROC/AUC recording, aggregated
                              across folds.

The CV evaluation produces the headline comparison table — the F1 0.60
result that the README reports as the validated out-of-sample number.

The SMOTE-balanced high-F1 optimization for top-3 models is in
`optimization.py` (Step 6), not here. This module's CV loop uses
`use_oversampling=False` by default to keep the comparison clean across
all 14 models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: BASE MODELS
# =====================================================================

def build_base_models(
    *,
    lr_params: dict,
    dt_params: dict,
    rf_params: dict,
    xgb_params: dict,
    lgb_params: dict,
    cat_params: dict,
    knn_params: dict,
) -> dict[str, Any]:
    """Instantiate the 7 base models with their hyperparameter dicts.

    KNN is wrapped in a make_pipeline with StandardScaler(with_mean=False)
    because distance-based models need scaled features. The with_mean=False
    setting ensures the scaler works on sparse-like data without
    centering.

    Args:
        lr_params: LogisticRegression hyperparameters.
        dt_params: DecisionTreeClassifier hyperparameters.
        rf_params: RandomForestClassifier hyperparameters.
        xgb_params: XGBClassifier hyperparameters.
        lgb_params: LGBMClassifier hyperparameters.
        cat_params: CatBoostClassifier hyperparameters.
        knn_params: KNeighborsClassifier hyperparameters.

    Returns:
        Dict mapping model name (str) to instantiated estimator.
        Order matches the original notebook.
    """
    return {
        "LogisticRegression": LogisticRegression(**lr_params),
        "DecisionTree": DecisionTreeClassifier(**dt_params),
        "RandomForest": RandomForestClassifier(**rf_params),
        "XGBoost": XGBClassifier(**xgb_params),
        "LightGBM": LGBMClassifier(**lgb_params),
        "CatBoost": CatBoostClassifier(**cat_params),
        "KNN": make_pipeline(
            StandardScaler(with_mean=False),
            KNeighborsClassifier(**knn_params),
        ),
    }


# =====================================================================
# SECTION 2: STACKED + BLENDED ENSEMBLES
# =====================================================================

def add_ensembles(models: dict[str, Any]) -> dict[str, Any]:
    """Add the 7 stacked/blended ensembles to the models dict.

    Each ensemble uses the already-instantiated base models from
    `build_base_models`. The ensembles are:

        Stacked_Ensemble           — XGBoost → LR meta
        Tree_Super_Stack           — LGBM + XGB + Cat + RF → LR meta
        Hybrid_Stack               — KNN + RF + LGBM → XGB meta
        Weighted_Blend             — soft VotingClassifier (LGBM:RF:XGB = 0.5:0.3:0.2)
        Tree_Super_Stack_LGBMmeta  — RF + XGB + LGBM + Cat → LGBM meta
        Stacked_Hetero_XGBmeta     — LR + KNN + DT + RF → XGB meta
        Boosting_Fusion_LGBMmeta   — XGB + LGBM + Cat → LGBM meta

    Returns the same dict for chaining convenience.

    Args:
        models: Dict from `build_base_models()`. Mutated in place.

    Returns:
        The mutated models dict, with 7 ensemble entries added.
    """
    # Simple Stack — XGB → LR meta
    models["Stacked_Ensemble"] = StackingClassifier(
        estimators=[("xgb", models["XGBoost"])],
        final_estimator=LogisticRegression(
            max_iter=3000, solver="lbfgs", class_weight="balanced",
        ),
        passthrough=True, n_jobs=1,
    )

    # Tree Super Stack — LGBM + XGB + Cat + RF → LR meta
    models["Tree_Super_Stack"] = StackingClassifier(
        estimators=[
            ("lgbm", models["LightGBM"]),
            ("xgb", models["XGBoost"]),
            ("cat", models["CatBoost"]),
            ("rf", models["RandomForest"]),
        ],
        final_estimator=LogisticRegression(
            max_iter=3000, solver="lbfgs", class_weight="balanced",
        ),
        passthrough=True, n_jobs=1,
    )

    # Hybrid Stack — KNN + RF + LGBM → XGB meta
    models["Hybrid_Stack"] = StackingClassifier(
        estimators=[
            ("knn", models["KNN"]),
            ("rf", models["RandomForest"]),
            ("lgbm", models["LightGBM"]),
        ],
        final_estimator=XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            random_state=42, scale_pos_weight=5,
            use_label_encoder=False, eval_metric="logloss",
        ),
        passthrough=True, n_jobs=1,
    )

    # Weighted Blend — soft VotingClassifier
    models["Weighted_Blend"] = VotingClassifier(
        estimators=[
            ("lgbm", models["LightGBM"]),
            ("rf", models["RandomForest"]),
            ("xgb", models["XGBoost"]),
        ],
        voting="soft", weights=[0.5, 0.3, 0.2], n_jobs=1,
    )

    # Tree Super Stack v2 — RF + XGB + LGBM + Cat → LGBM meta
    models["Tree_Super_Stack_LGBMmeta"] = StackingClassifier(
        estimators=[
            ("rf", models["RandomForest"]),
            ("xgb", models["XGBoost"]),
            ("lgbm", models["LightGBM"]),
            ("cat", models["CatBoost"]),
        ],
        final_estimator=LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            num_leaves=15, random_state=42, is_unbalance=True,
        ),
        passthrough=True, n_jobs=1,
    )

    # Heterogeneous Stack — LR + KNN + DT + RF → XGB meta
    models["Stacked_Hetero_XGBmeta"] = StackingClassifier(
        estimators=[
            ("lr", models["LogisticRegression"]),
            ("knn", models["KNN"]),
            ("dt", models["DecisionTree"]),
            ("rf", models["RandomForest"]),
        ],
        final_estimator=XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3,
            random_state=42, scale_pos_weight=5,
            use_label_encoder=False, eval_metric="logloss",
        ),
        passthrough=True, n_jobs=1,
    )

    # Boosting Fusion — XGB + LGBM + Cat → LGBM meta
    models["Boosting_Fusion_LGBMmeta"] = StackingClassifier(
        estimators=[
            ("xgb", models["XGBoost"]),
            ("lgbm", models["LightGBM"]),
            ("cat", models["CatBoost"]),
        ],
        final_estimator=LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            num_leaves=31, random_state=42, is_unbalance=True,
        ),
        passthrough=True, n_jobs=1,
    )

    return models


# =====================================================================
# SECTION 3: CV STRATEGY SELECTOR
# =====================================================================

def build_cv_splitter(
    cv_mode: str,
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    X: pd.DataFrame | None = None,
    group_col: str | None = None,
) -> Any:
    """Return a configured sklearn CV splitter based on cv_mode.

    Supported modes:
        strat5     — StratifiedKFold(n_splits=5, shuffle=True)
        strat10    — StratifiedKFold(n_splits=10, shuffle=True)
        repeated   — RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
        group      — StratifiedGroupKFold(n_splits=5, shuffle=True)
        timeseries — TimeSeriesSplit(n_splits=5)

    Note: the original notebook also supported a "nested" mode that
    triggered a GridSearchCV+cross_val_score sequence inside the
    branch. That branch was experimental and produced its results
    inside the if-block rather than returning a clean CV splitter
    for downstream use. The rebuild preserves the conceptual options
    but does not include the nested branch — it was an outlier in
    the original control flow.

    Args:
        cv_mode: One of the supported mode strings.
        n_splits: Number of folds. Default 5.
        n_repeats: Used only for "repeated" mode. Default 3.
        random_state: Seed for splits. Default 42.
        X: Optional DataFrame. Required for "group" mode to verify
           the group column exists.
        group_col: Group column name for "group" mode.

    Returns:
        A configured CV splitter object.

    Raises:
        ValueError: If cv_mode is unrecognized, or if group mode is
            requested but group_col is missing from X.
    """
    if cv_mode == "strat5":
        logger.info("Using Stratified 5-Fold CV")
        return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    if cv_mode == "strat10":
        logger.info("Using Stratified 10-Fold CV")
        return StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    if cv_mode == "repeated":
        logger.info("Using Repeated Stratified 5x3-Fold CV")
        return RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
        )

    if cv_mode == "group":
        if X is None or group_col is None or group_col not in X.columns:
            raise ValueError(
                f"Group CV mode requires X with column '{group_col}'"
            )
        logger.info("Using Stratified Group K-Fold CV (grouped by %r)", group_col)
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state,
        )

    if cv_mode == "timeseries":
        logger.info("Using TimeSeriesSplit (ordered by claim_date)")
        return TimeSeriesSplit(n_splits=n_splits)

    raise ValueError(
        f"Invalid cv_mode={cv_mode!r}. "
        "Choose from: strat5, strat10, repeated, group, timeseries."
    )


# =====================================================================
# SECTION 4: CV EVALUATION LOOP
# =====================================================================

def evaluate_model_cv(
    name: str,
    model: Any,
    X_df: pd.DataFrame,
    y_arr: np.ndarray,
    kf: Any,
    *,
    use_oversampling: bool = False,
    threshold_grid_start: float = 0.20,
    threshold_grid_end: float = 0.90,
    threshold_grid_n: int = 30,
    random_state: int = 42,
) -> tuple[dict, tuple]:
    """Cross-validate one model with per-fold F1 threshold tuning.

    For each fold:
        1. Fit the model (optionally inside a RandomOverSampler pipeline).
        2. Get class-1 probabilities (or scaled decision-function output).
        3. Sweep a threshold grid; record the best-F1 threshold per fold.
        4. Compute ROC curve and AUC for the fold.

    Aggregates across folds:
        - Mean of best-F1 scores
        - Mean of best thresholds (per-fold optimum, averaged)
        - Mean ROC TPR interpolated onto a common FPR grid
        - Mean and std of AUC

    Args:
        name: Model name for logging.
        model: Estimator instance.
        X_df: Feature DataFrame (must support iloc).
        y_arr: Target array (1D numpy).
        kf: CV splitter from `build_cv_splitter()`.
        use_oversampling: If True, RandomOverSampler is applied inside
            each fold via imblearn Pipeline. Default False.
        threshold_grid_start: F1 threshold sweep low bound. Default 0.20.
        threshold_grid_end: F1 threshold sweep high bound. Default 0.90.
        threshold_grid_n: Number of threshold candidates. Default 30.
        random_state: Seed for RandomOverSampler.

    Returns:
        Tuple ``(report_entry, roc_data)`` where:
            report_entry: dict with keys F1, Best_Threshold, AUC
            roc_data: tuple (mean_fpr, mean_tpr, mean_auc, std_auc)
    """
    logger.info("Training model: %s", name)

    sampler = RandomOverSampler(random_state=random_state)
    steps: list = []
    if use_oversampling:
        steps.append(("over", sampler))
    steps.append(("clf", model))
    pipe = ImbPipeline(steps)

    thresholds: list[float] = []
    f1s: list[float] = []
    aucs: list[float] = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs: list[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_df, y_arr), 1):
        X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        # Fit
        pipe.fit(X_tr, y_tr)

        # Probabilities for class 1
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_va)[:, 1]
        elif hasattr(pipe, "decision_function"):
            raw = pipe.decision_function(X_va)
            proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        else:
            raise ValueError(f"{name} lacks probability outputs")

        # Threshold tuning for F1
        ths = np.linspace(threshold_grid_start, threshold_grid_end, threshold_grid_n)
        f1_vals = [
            f1_score(y_va, (proba >= t).astype(int), zero_division=0) for t in ths
        ]
        best_i = int(np.argmax(f1_vals))
        thresholds.append(ths[best_i])
        f1s.append(f1_vals[best_i])

        # ROC / AUC
        fpr, tpr, _ = roc_curve(y_va, proba)
        auc = roc_auc_score(y_va, proba)
        aucs.append(auc)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        logger.info(
            "  Fold %d: F1=%.4f (thr=%.2f) | AUC=%.4f",
            fold, f1_vals[best_i], ths[best_i], auc,
        )

    # Aggregate over folds
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    mean_f1 = float(np.mean(f1s))
    mean_thr = float(np.mean(thresholds))

    return (
        {"F1": mean_f1, "Best_Threshold": mean_thr, "AUC": mean_auc},
        (mean_fpr, mean_tpr, mean_auc, std_auc),
    )


# =====================================================================
# ORCHESTRATOR
# =====================================================================

def run_modeling(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: Any,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run Step 5 end-to-end: build models, CV split, evaluate all 14 models.

    Args:
        X: Feature matrix (post feature_selection).
        y: Target Series.
        cfg: PipelineConfig instance.

    Returns:
        Tuple ``(comparison_df, models_dict, roc_curves)``:
            comparison_df: DataFrame indexed by model name, columns
                F1 / Best_Threshold / AUC, sorted by F1 descending.
            models_dict: Original models dict (for downstream optimization).
            roc_curves: Dict mapping model name to its ROC tuple.
    """
    logger.info("=== Step 5: Model construction + CV evaluation ===")

    # 1. Build all 14 models (7 base + 7 ensemble)
    models = build_base_models(
        lr_params=cfg.lr_params,
        dt_params=cfg.dt_params,
        rf_params=cfg.rf_params,
        xgb_params=cfg.xgb_params,
        lgb_params=cfg.lgb_params,
        cat_params=cfg.cat_params,
        knn_params=cfg.knn_params,
    )
    models = add_ensembles(models)
    logger.info("Loaded %d models for CV evaluation:", len(models))
    for name in models:
        logger.info("  - %s", name)

    # 2. Build CV splitter
    kf = build_cv_splitter(
        cfg.cv_mode,
        n_splits=cfg.cv_n_splits,
        n_repeats=cfg.cv_n_repeats,
        random_state=cfg.random_state,
        X=X,
    )

    # 3. Evaluate each model
    y_arr = np.asarray(y)
    logger.info("Training on %d samples, %d features\n", len(X), X.shape[1])

    report: dict[str, dict] = {}
    roc_curves: dict[str, tuple] = {}

    for name, mdl in models.items():
        report_entry, roc_data = evaluate_model_cv(
            name, mdl, X, y_arr, kf,
            threshold_grid_start=cfg.threshold_grid_start,
            threshold_grid_end=cfg.threshold_grid_end,
            threshold_grid_n=cfg.threshold_grid_n,
            random_state=cfg.random_state,
        )
        report[name] = report_entry
        roc_curves[name] = roc_data

    # 4. Build comparison table
    comparison = (
        pd.DataFrame(report)
        .T.sort_values(by="F1", ascending=False)
        .round(4)
    )

    logger.info("\n=== MODEL COMPARISON SUMMARY ===\n%s", comparison)

    return comparison, models, roc_curves