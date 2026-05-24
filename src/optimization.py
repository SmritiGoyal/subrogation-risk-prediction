"""
optimization.py
===============
Step 6: SMOTE-balanced cross-validated high-F1 optimization.

After Step 5's broad CV comparison, this stage takes the top N models
and runs a refined optimization loop with:

    1. SMOTE oversampling inside each fold (training fold only — the
       validation fold is left untouched, so SMOTE never sees the data
       it's evaluated on).
    2. F1-optimal threshold via `precision_recall_curve` (more precise
       than the linspace grid in Step 5 — sweeps every unique threshold
       implied by the predicted probabilities).
    3. Aggregated metrics across folds: F1, threshold, precision,
       recall, ROC-AUC, PR-AUC.

The function ``refit_on_full_train`` exists to produce the in-sample
refit number (F1 ~0.66 on the original Travelers competition data).
This number is **not** the headline metric for the project — the CV
F1 ~0.60 is. The refit metric exists for two purposes:
    - To produce test-set probabilities (the model fit on full train
      is what's deployed)
    - As a sanity check that the full-train fit hasn't gone wildly off
      vs the CV mean

Reporting the refit F1 as the headline would be misleading (it's
training data). The honest framing — used in the README and methodology
docs — is "F1 0.60 (CV)" with the refit number documented as a
secondary diagnostic.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# =====================================================================
# SECTION 1: SMOTE-BALANCED CV OPTIMIZATION
# =====================================================================

def smote_optimize_one_model(
    name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    kf: Any,
    *,
    smote_sampling_strategy: float = 0.8,
    random_state: int = 42,
) -> dict[str, float]:
    """Run SMOTE-balanced CV optimization for one model.

    For each fold:
        1. Apply SMOTE to the training fold only (via imblearn Pipeline).
        2. Fit the model.
        3. Predict probabilities on the (untouched) validation fold.
        4. Sweep all thresholds via `precision_recall_curve` and pick
           the one maximizing F1.
        5. Compute final fold metrics at the F1-optimal threshold.

    Args:
        name: Model name for logging.
        model: Estimator instance from the models dict.
        X: Feature DataFrame.
        y: Target Series.
        kf: CV splitter.
        smote_sampling_strategy: SMOTE oversampling ratio. Default 0.8
            (minority class oversampled to 80% the size of majority).
        random_state: Seed for SMOTE.

    Returns:
        Dict of aggregated metrics:
            Mean_F1, Std_F1, Mean_Threshold, Precision, Recall,
            ROC_AUC, PR_AUC.
    """
    logger.info("Cross-validated tuning for: %s", name)

    smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)

    f1s: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    aucs: list[float] = []
    pr_aucs: list[float] = []
    best_thrs: list[float] = []

    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        # SMOTE is applied only to the training fold via imblearn Pipeline
        pipe = ImbPipeline([
            ("smote", smote),
            ("clf", model),
        ])
        pipe.fit(X_tr, y_tr)

        probs = pipe.predict_proba(X_va)[:, 1]

        # Threshold tuning via precision_recall_curve
        # (more precise than linspace grid — uses every predicted prob)
        prec, rec, thr = precision_recall_curve(y_va, probs)
        f1_curve = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(np.argmax(f1_curve))
        best_thr = float(thr[best_idx])
        best_f1 = float(f1_curve[best_idx])
        best_thrs.append(best_thr)

        # Evaluate at the F1-optimal threshold
        y_pred = (probs >= best_thr).astype(int)
        f1s.append(best_f1)
        precisions.append(precision_score(y_va, y_pred, zero_division=0))
        recalls.append(recall_score(y_va, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_va, probs))
        pr_aucs.append(average_precision_score(y_va, probs))

        logger.info("  Fold %d: F1=%.4f | thr=%.3f", fold, best_f1, best_thr)

    return {
        "Mean_F1": float(np.mean(f1s)),
        "Std_F1": float(np.std(f1s)),
        "Mean_Threshold": float(np.mean(best_thrs)),
        "Precision": float(np.mean(precisions)),
        "Recall": float(np.mean(recalls)),
        "ROC_AUC": float(np.mean(aucs)),
        "PR_AUC": float(np.mean(pr_aucs)),
    }


def run_smote_optimization(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Any],
    top_model_names: tuple[str, ...],
    *,
    smote_sampling_strategy: float = 0.8,
    cv_n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run SMOTE-balanced CV for each of the top N models.

    Args:
        X: Feature DataFrame from feature_selection.
        y: Target Series.
        models: Full models dict from modeling.run_modeling().
        top_model_names: Names of models to optimize (from cfg).
        smote_sampling_strategy: SMOTE ratio. Default 0.8.
        cv_n_splits: Number of CV folds. Default 5.
        random_state: Seed for SMOTE and CV split.

    Returns:
        DataFrame indexed by model name, columns:
            Mean_F1, Std_F1, Mean_Threshold, Precision, Recall,
            ROC_AUC, PR_AUC. Sorted by Mean_F1 descending.
    """
    logger.info(
        "=== Step 6: SMOTE-balanced CV optimization on top %d models ===",
        len(top_model_names),
    )
    logger.info("Top models: %s", list(top_model_names))

    # Fresh stratified 5-fold for this stage — matches original
    kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)

    cv_results: dict[str, dict] = {}
    for name in top_model_names:
        if name not in models:
            logger.warning("  Model %r not in models dict — skipping", name)
            continue

        cv_results[name] = smote_optimize_one_model(
            name, models[name], X, y, kf,
            smote_sampling_strategy=smote_sampling_strategy,
            random_state=random_state,
        )

    # Sort by Mean_F1 descending
    cv_summary = (
        pd.DataFrame(cv_results)
        .T.sort_values("Mean_F1", ascending=False)
        .round(4)
    )

    logger.info("\n=== CROSS-VALIDATED HIGH-F1 RESULTS ===\n%s", cv_summary)

    return cv_summary


# =====================================================================
# SECTION 2: REFIT BEST MODEL ON FULL TRAINING DATA
# =====================================================================

def refit_on_full_train(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Any],
    cv_summary: pd.DataFrame,
    *,
    smote_sampling_strategy: float = 0.8,
    random_state: int = 42,
) -> tuple[ImbPipeline, str, float, dict[str, float]]:
    """Refit the best-CV model on full training data with its tuned threshold.

    The model with the highest CV Mean_F1 from `run_smote_optimization`
    is re-instantiated inside an imblearn Pipeline with SMOTE, then fit
    on the full training set (no holdout). The tuned threshold is the
    mean threshold across the CV folds.

    **Important honesty note:** the F1 / precision / recall returned
    here are **in-sample** metrics (computed on the same data the model
    was fit on). They will be optimistically biased vs the CV mean.
    This function exists for two purposes only:

        1. To produce a fitted pipeline ready for test-set inference.
        2. To sanity-check that the full-train refit isn't catastrophically
           different from the CV mean (a ~5-10pp gap is normal; a 30pp
           gap would indicate a bug).

    The CV F1 from `cv_summary` is the headline metric, not the
    in-sample refit F1.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        models: Full models dict.
        cv_summary: DataFrame from `run_smote_optimization`.
        smote_sampling_strategy: SMOTE ratio. Default 0.8.
        random_state: Seed for SMOTE.

    Returns:
        Tuple ``(final_pipe, best_model_name, best_threshold, refit_metrics)``:
            final_pipe: Fitted ImbPipeline (SMOTE + best model).
            best_model_name: Name of the best CV model.
            best_threshold: The tuned threshold to use for predictions.
            refit_metrics: dict with F1, Precision, Recall, ROC_AUC, PR_AUC
                computed in-sample (label this as "refit, in-sample"
                wherever reported).
    """
    best_model_name = cv_summary.index[0]
    best_thr = float(cv_summary.loc[best_model_name, "Mean_Threshold"])
    logger.info(
        "Refitting best CV model (%s) on full training data using thr=%.3f",
        best_model_name, best_thr,
    )

    smote = SMOTE(sampling_strategy=smote_sampling_strategy, random_state=random_state)
    final_pipe = ImbPipeline([
        ("smote", smote),
        ("clf", models[best_model_name]),
    ])
    final_pipe.fit(X, y)

    probs_full = final_pipe.predict_proba(X)[:, 1]
    y_pred_full = (probs_full >= best_thr).astype(int)

    refit_metrics = {
        "F1": float(f1_score(y, y_pred_full)),
        "Precision": float(precision_score(y, y_pred_full, zero_division=0)),
        "Recall": float(recall_score(y, y_pred_full, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, probs_full)),
        "PR_AUC": float(average_precision_score(y, probs_full)),
    }

    logger.info(
        "In-sample refit metrics — F1=%.4f | Precision=%.4f | Recall=%.4f",
        refit_metrics["F1"], refit_metrics["Precision"], refit_metrics["Recall"],
    )
    logger.info(
        "  ROC AUC=%.4f | PR AUC=%.4f",
        refit_metrics["ROC_AUC"], refit_metrics["PR_AUC"],
    )
    logger.info(
        "  (REMINDER: these are IN-SAMPLE — for honest reporting use "
        "the CV Mean_F1=%.4f from the cv_summary table.)",
        cv_summary.loc[best_model_name, "Mean_F1"],
    )

    return final_pipe, best_model_name, best_thr, refit_metrics