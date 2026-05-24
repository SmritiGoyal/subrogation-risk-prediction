"""
run_pipeline.py
===============
End-to-end orchestrator for the subrogation risk prediction pipeline.

Runs all 6 stages in order, calling out to the modular src/* files:

    Step 1 — Data loading + cleaning                     (data_loading)
    Step 2-3 — Feature engineering + transforms          (feature_engineering)
    Step 4 — LightGBM wrapper feature selection (top-60) (feature_selection)
    Step 5 — Build 14 models + CV evaluation             (modeling)
    Step 6 — SMOTE-balanced high-F1 optimization         (optimization)
    Step 7 — Refit best model + write submission         (this file)

Saves four output artifacts to the configured output directory:
    - submission_final.csv             — test set predictions (binary)
    - cv_results.csv                   — Step 5 model comparison
    - optimization_results.csv         — Step 6 CV summary on top 3 models
    - feature_importance.csv           — Step 4 LightGBM importances

To run:
    python -m src.run_pipeline

Or directly:
    python src/run_pipeline.py

Configuration is read from config.py at the repository root. Copy
config.example.py to config.py first if you need to override defaults.

Honest framing note: this pipeline's headline metric is the CV Mean_F1
from Step 6 (typically ~0.60). The in-sample refit F1 (~0.66) that the
final stage computes is a diagnostic artifact, not a reported result.
The README and methodology reflect this convention.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Make the repo root importable so `config` resolves regardless of where
# this script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configuration
try:
    from config import PipelineConfig
except ImportError as exc:
    raise ImportError(
        "Could not import PipelineConfig from config.py. "
        "Run `cp config.example.py config.py` from the repo root first."
    ) from exc

# Module imports
from data_loading import run_data_loading
from feature_engineering import run_feature_engineering
from feature_selection import run_feature_selection
from modeling import run_modeling
from optimization import refit_on_full_train, run_smote_optimization


logger = logging.getLogger(__name__)


# =====================================================================
# SUBMISSION WRITER
# =====================================================================

def write_submission(
    test_feature_reduced: pd.DataFrame,
    test_raw: pd.DataFrame,
    final_pipe,
    best_threshold: float,
    id_col: str,
    target_col: str,
    output_path: Path,
) -> None:
    """Generate test predictions and write the submission CSV.

    Args:
        test_feature_reduced: Test feature matrix after Step 4 selection.
        test_raw: Original test DataFrame (used to source the ID column).
        final_pipe: Fitted ImbPipeline from `refit_on_full_train`.
        best_threshold: Tuned decision threshold from CV.
        id_col: Name of the ID column.
        target_col: Name of the target column.
        output_path: Where to write the submission CSV.
    """
    logger.info("Generating test predictions on %d rows", len(test_feature_reduced))

    p_test = final_pipe.predict_proba(test_feature_reduced)[:, 1]
    y_test_hat = (p_test >= best_threshold).astype(int)

    logger.info(
        "Test prediction summary — positive: %d (%.1f%%) | negative: %d (%.1f%%)",
        int(y_test_hat.sum()), 100 * y_test_hat.mean(),
        int((y_test_hat == 0).sum()), 100 * (y_test_hat == 0).mean(),
    )

    submission = pd.DataFrame({
        id_col: test_raw[id_col].values,
        target_col: y_test_hat,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Submission written: %s (%d rows)", output_path, len(submission))


# =====================================================================
# OUTPUT WRITERS
# =====================================================================

def write_cv_results(
    comparison: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> None:
    """Write the Step 5 model comparison table."""
    path = output_dir / filename
    comparison.to_csv(path)
    logger.info("Step 5 CV results: %s (%d models)", path, len(comparison))


def write_optimization_results(
    cv_summary: pd.DataFrame,
    output_dir: Path,
    filename: str,
) -> None:
    """Write the Step 6 SMOTE optimization summary table."""
    path = output_dir / filename
    cv_summary.to_csv(path)
    logger.info("Step 6 optimization results: %s (%d models)", path, len(cv_summary))


def write_feature_importance(
    importances: pd.Series,
    output_dir: Path,
    filename: str,
) -> None:
    """Write the Step 4 LightGBM feature importances table."""
    path = output_dir / filename
    importances.rename("importance").to_csv(path, header=True)
    logger.info(
        "Step 4 feature importance: %s (%d features)", path, len(importances),
    )


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main(config: PipelineConfig | None = None) -> dict:
    """Run the full subrogation pipeline end-to-end.

    Args:
        config: Optional PipelineConfig override. Default uses
            PipelineConfig() with all default values.

    Returns:
        Dict containing key intermediate artifacts:
            id_col, comparison, cv_summary, importances,
            best_model_name, best_threshold, refit_metrics.
    """
    cfg = config or PipelineConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: data loading + cleaning
    # -----------------------------------------------------------------
    logger.info("\n=== Step 1: Data loading + cleaning ===")
    train_cleaned, test_cleaned, id_col = run_data_loading(
        cfg.train_path, cfg.test_path,
    )
    # Keep the raw test DataFrame around — we need its ID column at the end
    test_raw = test_cleaned[[id_col]].copy()

    # -----------------------------------------------------------------
    # Step 2-3: feature engineering
    # -----------------------------------------------------------------
    logger.info("\n=== Step 2-3: Feature engineering ===")
    train_feature, test_feature = run_feature_engineering(
        train_cleaned, test_cleaned, reference_year=cfg.reference_year,
    )

    # -----------------------------------------------------------------
    # Step 4: LightGBM wrapper feature selection
    # -----------------------------------------------------------------
    logger.info("\n=== Step 4: LightGBM feature selection ===")
    train_feature_reduced, test_feature_reduced, importances = run_feature_selection(
        train_feature, test_feature,
        target_col=cfg.target_col,
        top_n=cfg.top_n_features,
        lgb_params=cfg.fs_lgb_params,
        random_state=cfg.random_state,
    )

    write_feature_importance(
        importances, cfg.output_dir, cfg.feature_importance_filename,
    )

    # Split target from features for downstream stages
    y = train_feature_reduced[cfg.target_col].astype(int)
    X = train_feature_reduced.drop(columns=[cfg.target_col])

    # -----------------------------------------------------------------
    # Step 5: build models + CV evaluation across 14 models
    # -----------------------------------------------------------------
    logger.info("\n=== Step 5: Model CV evaluation (14 models) ===")
    comparison, models, _roc_curves = run_modeling(X, y, cfg)
    write_cv_results(comparison, cfg.output_dir, cfg.cv_results_filename)

    # -----------------------------------------------------------------
    # Step 6: SMOTE-balanced CV optimization on top 3
    # -----------------------------------------------------------------
    logger.info("\n=== Step 6: SMOTE-balanced high-F1 optimization ===")
    cv_summary = run_smote_optimization(
        X, y, models,
        top_model_names=cfg.top_models_for_optimization,
        smote_sampling_strategy=cfg.smote_sampling_strategy,
        cv_n_splits=cfg.cv_n_splits,
        random_state=cfg.random_state,
    )
    write_optimization_results(
        cv_summary, cfg.output_dir, cfg.optimization_results_filename,
    )

    # -----------------------------------------------------------------
    # Step 7: refit best model + write submission
    # -----------------------------------------------------------------
    logger.info("\n=== Step 7: Refit best model + test predictions ===")
    final_pipe, best_model_name, best_threshold, refit_metrics = refit_on_full_train(
        X, y, models, cv_summary,
        smote_sampling_strategy=cfg.smote_sampling_strategy,
        random_state=cfg.random_state,
    )

    write_submission(
        test_feature_reduced, test_raw, final_pipe, best_threshold,
        id_col=id_col,
        target_col=cfg.target_col,
        output_path=cfg.submission_path,
    )

    # -----------------------------------------------------------------
    # Final summary log
    # -----------------------------------------------------------------
    cv_f1 = float(cv_summary.loc[best_model_name, "Mean_F1"])
    logger.info("\n=== Pipeline complete ===")
    logger.info("Best model:       %s", best_model_name)
    logger.info("Headline CV F1:   %.4f  (out-of-sample, the honest number)", cv_f1)
    logger.info(
        "In-sample refit F1: %.4f  (NOT a reportable headline — see methodology.md)",
        refit_metrics["F1"],
    )
    logger.info("Tuned threshold:  %.3f", best_threshold)

    return {
        "id_col": id_col,
        "comparison": comparison,
        "cv_summary": cv_summary,
        "importances": importances,
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "refit_metrics": refit_metrics,
    }


def _configure_logging() -> None:
    """Configure root logging for CLI execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


if __name__ == "__main__":
    _configure_logging()
    results = main()
    print(f"\n✅ Pipeline complete. Best model: {results['best_model_name']}")