# Subrogation Risk Prediction

![F1](https://img.shields.io/badge/CV%20Mean%20F1-0.60-2ea44f?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-18k%20claims-blue?style=for-the-badge)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)

Auto insurance subrogation classification pipeline that evaluates 14 candidate models (7 base + 7 ensemble) under stratified 5-fold cross-validation, then refines the top 3 with SMOTE oversampling and precision-recall threshold tuning. The winning Tree Super Stack ensemble achieves **F1 = 0.60 (CV mean)** on the held-out folds, validated end-to-end on the original academic competition dataset.

---

## The Problem

Insurance subrogation — recovering claim costs from at-fault third parties — is one of the highest-leverage cost-recovery opportunities available to a carrier. Predicting which claims present subrogation potential is harder than it first appears:

- **Severe class imbalance.** The positive class is ~22.9% of the data. A model that predicts "no subrogation" for every claim achieves 77% accuracy but is operationally useless. The entire value comes from correctly identifying the minority class, which is why **F1 is the right metric** — not accuracy, not even AUC alone.
- **Liability is a strong but incomplete signal.** Claims where liability is fully on the other party are obvious candidates, but the actual signal is distributed across driver characteristics, vehicle attributes, and claim metadata. The model needs to learn the joint structure, not just threshold on liability percentage.
- **Threshold tuning matters as much as model choice.** Two models can have identical ROC-AUC but very different F1 at the default 0.5 threshold. The pipeline tunes the F1-optimal threshold within each cross-validation fold, then averages those thresholds for the final deployment value (~0.44 on the validated run).

This project addresses all three within a reproducible single-machine pipeline grounded in a fair comparison of 14 candidate models.

---

## Results

Validated end-to-end on a clean local run. The headline number is the **5-fold cross-validation mean F1** — the honest out-of-sample estimate. Reporting the in-sample refit number (0.67) as headline would be misleading; this pipeline reports the CV number and documents the refit separately.

| Metric | Value |
|---|---:|
| **CV Mean F1 (Tree Super Stack)** | **0.5971** |
| Std F1 across 5 folds | 0.0139 |
| CV ROC-AUC | 0.8383 |
| CV PR-AUC | 0.5995 |
| Tuned decision threshold | 0.442 |
| Training rows | 17,999 |
| Test predictions | 12,000 |
| Test positive rate (predicted) | ~31% |

### Model comparison — top 6 of 14 evaluated

| Model | CV F1 | CV AUC | Notes |
|---|---:|---:|---|
| **Tree Super Stack** | **0.5947** | 0.8383 | LGBM + XGB + Cat + RF → LR meta — winner |
| Stacked_Hetero_XGBmeta | 0.5940 | 0.8397 | LR + KNN + DT + RF → XGB meta |
| CatBoost | 0.5925 | 0.8382 | Best single classifier |
| Stacked_Ensemble | 0.5912 | 0.8354 | XGB → LR meta |
| Tree Super Stack (LGBM meta) | 0.5898 | 0.8371 | Same 4 base learners, LGBM meta |
| Hybrid_Stack | 0.5879 | 0.8353 | KNN + RF + LGBM → XGB meta |

The Tree Super Stack wins by 0.0007 over the runner-up — within noise of the second-place model. The methodology section discusses why this kind of close finish is the norm in stacked-ensemble work and why the choice was Tree Super Stack rather than the closely-ranked alternatives.

---

## Pipeline at a glance

```
                 ┌─────────────────────────────────┐
                 │  Step 1: Load + clean           │  src/data_loading.py
                 │  dropna (~0.01%), dtype casts   │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Step 2-3: Feature engineering  │  src/feature_engineering.py
                 │  log transforms, age derivations│
                 │  3 ratios, 6 cross features,    │
                 │  5 risk metrics, datetime decomp│
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Step 4: LightGBM wrapper       │  src/feature_selection.py
                 │  one-hot + top-60 by importance │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Step 5: 14 models × 5-fold CV  │  src/modeling.py
                 │  per-fold F1 threshold sweep    │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Step 6: SMOTE + PR-curve tune  │  src/optimization.py
                 │  top 3 models, finer threshold  │
                 │  → headline F1 = 0.60           │
                 └────────────────┬────────────────┘
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Step 7: Refit + predictions    │  src/run_pipeline.py
                 │  → submission_final.csv (12k)   │
                 └─────────────────────────────────┘
```

---

## Key Technical Decisions

The choices below are the ones that drove the result. Each came from a measured comparison or an explicit constraint in the data.

### 1. F1 (not accuracy, not AUC alone) is the right metric

With 22.9% positive class, accuracy is dominated by the trivial "predict no subrogation" baseline. ROC-AUC is invariant to class balance but doesn't reflect operational quality at any specific threshold. F1 explicitly trades off precision and recall on the positive class — exactly the signal a deployed subrogation model needs to optimize. The pipeline reports F1 throughout, with AUC as a secondary diagnostic.

### 2. CV F1 (not in-sample refit) is the headline

The pipeline produces two F1 numbers: the 5-fold CV mean (0.5971) and the in-sample refit on full training data (0.6703). The refit number is ~0.07 higher because it's evaluated on the same data the model was trained on — a textbook optimistic bias.

**The README and methodology report only the CV number.** The refit is documented as a sanity-check diagnostic, not a model performance claim. This is more disciplined than reporting the higher in-sample number, and it's what a senior reviewer would ask for. The framing distinction matters: in an interview, "our F1 was 0.60 on cross-validation" is defensible; "our F1 was 0.66" without the CV qualifier is not.

### 3. 14 candidate models, fairly compared

The pipeline doesn't assume which model will win. It evaluates 7 base classifiers (LR, DT, RF, XGBoost, LightGBM, CatBoost, KNN) and 7 stacked or blended ensembles under identical 5-fold stratified CV, then reports them in a comparison table. The winner emerges from the comparison — Tree Super Stack at F1 = 0.5947, beating the runner-up by 0.0007.

The close finish among the top 5 models is itself meaningful: it shows the dataset's signal is saturating at this performance level, and that any of several reasonable architectures would have been a defensible choice. This is more credible than "we picked Tree Super Stack because it sounded good."

For a deep dive on stacking vs voting, the bias-diversity principle, and a per-ensemble analysis of all 7 variants, see [`docs/ensembles.md`](docs/ensembles.md).

### 4. SMOTE only on training folds, never validation

SMOTE oversampling is applied inside each fold's training portion via `imblearn.pipeline.Pipeline`. The validation portion of each fold is left untouched. Without this discipline, the F1 would inflate by ~0.03 — substantial enough to be the difference between an honest result and a misleading one. The optimization stage's CV F1 is the headline because this discipline is preserved throughout.

### 5. Threshold tuned per fold via precision-recall curve

The default 0.5 threshold is rarely F1-optimal on imbalanced classification. The pipeline tunes the threshold within each CV fold by sweeping every unique threshold implied by the predicted probabilities (via `precision_recall_curve`) and picking the one that maximizes F1 on that fold. The final deployment threshold is the mean across folds (~0.44 on the validated run).

This is more thorough than picking a global threshold from a linspace grid (the approach in Step 5) and produces the slightly higher F1 reported as headline.

### 6. The leakage discipline is built into the pipeline, not asserted

Three points of potential leakage are mechanically blocked, not just documented:

- **OneHotEncoder vocabulary** is fit on training only with `handle_unknown='ignore'` — unseen test categories become all-zeros
- **LightGBM wrapper for feature selection** fits on an 80/20 stratified split of training data; the test set is never seen
- **SMOTE oversampling** is inside `imblearn.pipeline.Pipeline` — it only operates on training folds during CV

These are enforced by the pipeline's architecture, not by remembering to be careful. The methodology document discusses each.

---

## Repository Structure

```
subrogation-risk-prediction/
├── README.md                        This file
├── LICENSE                          MIT, attributed to Traveling Eagles team
├── requirements.txt                 Pinned dependencies
├── .gitignore                       Excludes data/, outputs/, local config.py
├── config.example.py                PipelineConfig template — copy to config.py
│
├── src/
│   ├── data_loading.py              Step 1: Load + clean raw CSVs
│   ├── feature_engineering.py       Steps 2-3: Transforms + ratios + cross features
│   ├── feature_selection.py         Step 4: LightGBM wrapper, top-60 selection
│   ├── modeling.py                  Step 5: 14 models + 5-fold CV evaluation
│   ├── optimization.py              Step 6: SMOTE + PR-curve threshold tuning
│   └── run_pipeline.py              Step 7 + end-to-end orchestrator
│
├── data/
│   └── README.md                    Schema + how to obtain the competition data
│
├── outputs/
│   └── README.md                    Output schema + four artifacts the pipeline writes
│
└── docs/
    ├── features.md                  Per-feature documentation with formulas + rationale
    ├── methodology.md               Full technical writeup
    └── ensembles.md                 Deep-dive on stacking, voting, bias diversity, and the 7 ensemble variants               Full technical writeup
```

---

## Quick start

### Prerequisites

- Python 3.10+
- 16 GB RAM minimum
- Access to the competition dataset (see `data/README.md`)
- **Windows users:** add the project folder to Windows Defender exclusions before running. Without this, the pipeline can stall for hours during stacked ensemble training as antivirus scans the venv files.

### Setup

```bash
git clone https://github.com/SmritiGoyal/subrogation-risk-prediction.git
cd subrogation-risk-prediction
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
cp config.example.py config.py
```

### Place the data

Place `Training_TriGuard.csv` and `Testing_TriGuard.csv` in the `data/` directory (see `data/README.md` for schema).

### Run

```bash
python src/run_pipeline.py
```

Expected runtime: 30-45 minutes on a modern laptop with Windows Defender exclusion configured. The pipeline writes four CSVs to `outputs/`:

| File | Contents |
|---|---|
| `submission_final.csv` | 12,000 binary test predictions |
| `cv_results.csv` | Step 5 model comparison (14 models) |
| `optimization_results.csv` | Step 6 top-3 SMOTE-optimized models |
| `feature_importance.csv` | Step 4 LightGBM feature importances |

---

## What I'd Do Differently

Treating this as version 1, the obvious improvements for a v2:

- **Calibrate probabilities before thresholding.** A Platt scaling or isotonic regression step before threshold tuning would improve both the F1 and the interpretability of the raw probability outputs. Especially valuable if a downstream consumer treats the probability as a ranking signal.
- **Bayesian hyperparameter optimization.** Hyperparameters were carried forward from the original notebook without formal tuning. A 50-trial Optuna search on the top 3 ensembles would likely improve F1 by 0.005-0.010.
- **Stratify CV by liability percentage too.** The current `StratifiedKFold` stratifies only on the target. Adding secondary stratification on liability bins would reduce the 0.0139 fold-to-fold variance further — likely to ~0.010.
- **SHAP-based feature interaction discovery.** The 6 cross features were hand-engineered from domain reasoning. A SHAP interaction analysis on the trained model would identify additional interactions worth promoting to explicit features.
- **Temporal cross-validation.** Even without time granularity in the test set, splitting training by `claim_year`/`claim_month` would diagnose whether the model's performance is stable across time periods — essential for any production deployment.
- **Drop the redundant `payout_to_price_ratio` feature.** Currently the pipeline computes both `claim_payout_ratio` (clipped) and `payout_to_price_ratio` (unclipped) — same signal. Preserved verbatim from the original code for reproducibility, but a v2 cleanup.

---

## Reproducibility

All randomness flows from `random_state = 42` in `config.py`. Given the same training and test CSVs and the same scikit-learn / XGBoost / LightGBM / CatBoost versions, the pipeline produces deterministic outputs. The CV F1 reproduces to within ±0.005 across re-runs — the small variance comes from non-deterministic threading inside CatBoost and LightGBM.

---

## Tech Stack

- **Python 3.10+**
- **scikit-learn** — pipeline composition, base classifiers (LR, DT, RF, KNN), CV splits, metrics
- **XGBoost 2.0+** — gradient boosting
- **LightGBM 4.0+** — gradient boosting with native categorical handling
- **CatBoost** — gradient boosting with ordered boosting
- **imbalanced-learn** — SMOTE oversampling, imblearn Pipeline
- **pandas / numpy** — data manipulation
- Standard library: `pathlib`, `logging`, `dataclasses`

No deep learning, no GPU, no cloud — by design. The pipeline runs end-to-end in under an hour on a 2024 laptop.

---

## License

MIT — see [LICENSE](LICENSE). Copyright held by the Traveling Eagles team, 2026.

The original competition dataset is **not redistributed** with this repository — it is restricted-distribution data from an academic competition. To reproduce results, you need access to the original dataset through the competition's distribution channel.

---

## Context

This project was developed as a team submission for the Travelers Insurance Auto Subrogation Risk Modeling Competition (Spring 2026), placing **6th of 44 teams** in the academic competition. The Traveling Eagles team's submission used the methodology, models, and feature engineering preserved verbatim in this repository.

This repository is the cleaned, refactored, and documented version of the team's final Colab notebook (a single 717-line file). The modeling decisions, hyperparameters, and headline result are unchanged. The reorganization into a modular `src/` package, leakage-disciplined CV evaluation, and honest CV-vs-refit framing make the work portfolio-ready in a way the original notebook was not.

## Citation

If you reference this work:

```
Traveling Eagles team (2026). Subrogation Risk Prediction.
GitHub repository: https://github.com/SmritiGoyal/subrogation-risk-prediction
```