"""
config.example.py
=================
Template for local configuration. Copy this file to `config.py` and fill
in any path overrides needed for your environment. `config.py` is
gitignored.

Usage:
    cp config.example.py config.py
    # then edit config.py if you need to override defaults

The PipelineConfig dataclass below is the single source of truth for
every hyperparameter, path, and tunable in this pipeline. It is consumed
by `src/run_pipeline.py` at runtime.

Defaults are tuned for the original Travelers subrogation competition
dataset (~18k claims, ~23% positive class). If you're running on a
different dataset, the most likely things to tune are:
    - target_class_balance (for SMOTE sampling_strategy)
    - reference_year (used to compute driver_age, vehicle_age)
    - top_n_features (LightGBM wrapper selection threshold)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PipelineConfig:
    """All hyperparameters and paths for the subrogation pipeline.

    Instances are immutable (``frozen=True``) so configuration can be
    safely shared across function calls without risk of mutation.
    """

    # Directories ----------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # Input file names within ``data_dir`` --------------------------
    # The original competition dataset was named ``Training_TriGuard.csv`` /
    # ``Testing_TriGuard.csv`` — "TriGuard" is the masked carrier name
    # used in the original challenge. We preserve those filenames so the
    # pipeline runs against the original competition data unchanged.
    train_filename: str = "Training_TriGuard.csv"
    test_filename: str = "Testing_TriGuard.csv"

    # Output file names within ``output_dir`` -----------------------
    submission_filename: str = "submission_final.csv"
    cv_results_filename: str = "cv_results.csv"
    optimization_results_filename: str = "optimization_results.csv"
    feature_importance_filename: str = "feature_importance.csv"

    # Schema --------------------------------------------------------
    target_col: str = "subrogation"
    id_col_options: tuple[str, ...] = ("claim_number", "claim_nbr")

    # Reproducibility -----------------------------------------------
    random_state: int = 42

    # Reference year for age derivations ----------------------------
    # driver_age = reference_year - year_of_born
    # vehicle_age = reference_year - vehicle_made_year
    # Set to the year the model is intended for inference. Original
    # competition used 2025.
    reference_year: int = 2025

    # Feature selection ---------------------------------------------
    # LightGBM-based wrapper selection: keep the top N features by
    # split-gain importance, drop the rest. 60 was selected via deck
    # observation that the cumulative gain plateaus around 50-60 features.
    top_n_features: int = 60

    # Feature-selection LightGBM hyperparameters --------------------
    fs_lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": -1,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    })

    # Cross-validation ----------------------------------------------
    # Original notebook used CV_MODE = "strat5" (StratifiedKFold n_splits=5).
    # Switching to "strat10" or "repeated" trades stability for runtime.
    cv_mode: str = "strat5"
    cv_n_splits: int = 5
    cv_n_repeats: int = 3  # only used if cv_mode == "repeated"

    # SMOTE for class balancing during high-F1 optimization --------
    # 0.8 means oversample the minority class to 80% the size of the
    # majority class. Values between 0.5 and 1.0 are reasonable.
    smote_sampling_strategy: float = 0.8

    # Threshold tuning ----------------------------------------------
    # First-pass grid (used in initial CV evaluation):
    threshold_grid_start: float = 0.20
    threshold_grid_end: float = 0.90
    threshold_grid_n: int = 30

    # Top models for the SMOTE-balanced high-F1 optimization loop.
    # Original notebook selected these three from the initial CV results.
    top_models_for_optimization: tuple[str, ...] = (
        "Tree_Super_Stack",
        "XGBoost",
        "Stacked_Hetero_XGBmeta",
    )

    # Base model hyperparameters ------------------------------------
    # All preserved verbatim from the original Colab notebook.
    lr_params: dict = field(default_factory=lambda: {
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
        "n_jobs": 1,
    })

    dt_params: dict = field(default_factory=lambda: {
        "max_depth": 12,
        "min_samples_split": 40,
        "min_samples_leaf": 25,
        "random_state": 42,
        "class_weight": "balanced",
    })

    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_leaf": 15,
        "max_features": 0.5,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": 1,
    })

    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.5,
        "reg_lambda": 2,
        "scale_pos_weight": 3.5,
        "random_state": 42,
        "n_jobs": 1,
        "use_label_encoder": False,
        "verbosity": 0,
        "eval_metric": "aucpr",
    })

    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "max_depth": -1,
        "scale_pos_weight": 3.5,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 1.0,
        "is_unbalance": False,
        "metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })

    cat_params: dict = field(default_factory=lambda: {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
    })

    knn_params: dict = field(default_factory=lambda: {
        "n_neighbors": 10,
        "weights": "distance",
        "metric": "minkowski",
        "n_jobs": 1,
    })

    @property
    def train_path(self) -> Path:
        """Full path to the training CSV."""
        return self.data_dir / self.train_filename

    @property
    def test_path(self) -> Path:
        """Full path to the test CSV."""
        return self.data_dir / self.test_filename

    @property
    def submission_path(self) -> Path:
        """Full path where the submission CSV will be written."""
        return self.output_dir / self.submission_filename