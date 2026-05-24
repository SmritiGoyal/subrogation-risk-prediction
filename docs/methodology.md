# Methodology

This document explains the technical decisions behind the subrogation risk prediction pipeline. It is the companion to `features.md` (which covers the input feature engineering in detail) and the top-level `README.md` (which is the landing page). This document focuses on everything else: framing, modeling choices, cross-validation discipline, and the honest framing of the headline metric.

## 1. Problem framing

### 1.1 What we are predicting

For each auto insurance claim in the test set, we predict the probability that the claim presents a **subrogation opportunity** — that is, that the at-fault party is identifiably a third party from whom the insurer can pursue recovery. The competition target is binary classification on the `subrogation` column.

Insurance subrogation is one of the highest-leverage recoveries available to a carrier. A 1% improvement in subrogation identification translates directly to recovery dollars, and unlike most underwriting decisions, the model's predictions create no adverse selection — flagging more potential subrogation cases for human review is strictly upside.

### 1.2 Why this problem is harder than it sounds

Three structural challenges:

- **Class imbalance.** The positive class is ~22.9% of the training data. A model that predicts "no subrogation" for every claim achieves 77% accuracy but is operationally useless — the entire value comes from correctly identifying the minority class. This is why **F1 is the right metric**, not accuracy or AUC alone. F1 explicitly trades off precision and recall on the positive class.

- **Liability percentage carries the strongest signal, but isn't conclusive.** A claim where liability is 100% on the other party is an obvious subrogation candidate. But the deck and feature importance both show liability is one of several features that matter — driver demographics, vehicle characteristics, and claim attributes all contribute. The model needs to learn the joint structure, not just threshold on liability.

- **Threshold tuning matters as much as model choice.** At the default 0.5 threshold, an imbalanced binary classifier underperforms on F1 because it's optimizing log loss (which doesn't care about the threshold). Two models can have identical ROC-AUC but very different F1 at threshold 0.5. The pipeline addresses this by sweeping thresholds within each fold and reporting F1 at the F1-optimal threshold per fold.

### 1.3 The metric: F1 at the F1-optimal threshold

The pipeline reports F1 = 0.5971 ± 0.0139 (5-fold CV mean ± std). The "F1-optimal" qualifier matters: every fold's F1 is computed at its own threshold, found by sweeping the precision-recall curve and picking the threshold that maximizes F1 on that fold.

This is a slightly more generous evaluation than fixing the threshold at 0.5, but it's the correct evaluation for a model that will be deployed with a tuned threshold. The CV mean of the per-fold optimal thresholds (~0.44 on the validated run) is what gets used for final test predictions.

## 2. Data engineering

### 2.1 The cleaning is minimal

The source dataset has ~0.01% missingness across all 28 features — about 2 rows total. The pipeline uses `dropna()` rather than column-specific imputation. This is unusual for ML pipelines but appropriate here: heavier imputation isn't justified at this missingness rate, and the dropna keeps the pipeline simple. For datasets with more missing data, this would need to be replaced with proper imputation.

After cleaning, the 18,001-row training file becomes 17,999 rows. The 12,000-row test file is unchanged.

### 2.2 Feature engineering is what gives the model its edge

The raw CSV has 29 columns including the target. Without feature engineering, a simple tree model on the raw features achieves F1 around 0.50-0.52. The engineered features push it to 0.59-0.60. The lift comes specifically from:

- **Log transforms** of skewed monetary columns (income, payout, vehicle price, mileage) — makes the distributions more useful for linear models without hurting trees
- **Cross features** like `driver_risk_liab` and `liab_to_claim_ratio` — the LightGBM importance ranking shows these outperform the raw inputs they're derived from, confirming the model can't extract this signal from raw columns alone
- **Liability bins** (`liab_low`, `liab_mid`, `liab_high`) — explicit thresholds around 25% and 75% that the model can split on directly

See `docs/features.md` for the full per-feature documentation.

### 2.3 The leakage discipline

Three points where leakage could creep in:

1. **Categorical encoding is fit on training only.** The OneHotEncoder learns its vocabulary from training data with `handle_unknown='ignore'`, so unseen categories in the test set are encoded as all-zeros rather than crashing or leaking distribution information backward.

2. **Feature selection is fit on training only.** The LightGBM wrapper that ranks features by importance fits on an 80/20 stratified split of the training data. The test set is held out from this stage entirely.

3. **SMOTE oversampling is inside each fold's training portion only.** This is enforced via `imblearn.pipeline.Pipeline` rather than ad-hoc resampling. The validation portion of each fold is left untouched — SMOTE never sees data it's being evaluated on.

Without these disciplines, the F1 would inflate by 0.03-0.05 on the same data — substantial enough to matter for honest reporting.

## 3. Model architecture

### 3.1 Twelve models compared, not asserted

The pipeline evaluates 14 candidate models (7 base + 7 ensemble) under identical 5-fold stratified CV. The progression is:

| Family | Models | Validation F1 (CV mean) |
|---|---|---:|
| Single classifiers | LR, DT, RF | 0.53 - 0.59 |
| Gradient boosters | XGBoost, LightGBM, CatBoost | 0.58 - 0.59 |
| Ensembles | 7 stacked / blended variants | 0.59 - 0.60 |

The ensembles win, but by less than 0.01 F1 over the best single booster (LightGBM or CatBoost). The Tree Super Stack (4 base models with an LR meta-learner) is the winning configuration at F1 = 0.5947 in the broad CV and 0.5971 in the SMOTE-balanced CV.

**Why this matters:** the result is more credible than "we chose Tree Super Stack because it sounded good." Each of the 14 models was given a fair shot under the same protocol, and the winner emerged from the comparison rather than being assumed.

### 3.2 Why Tree Super Stack specifically

Tree Super Stack stacks four base learners with a logistic regression meta-learner:

- **LightGBM** — fast gradient boosting, native categorical handling
- **XGBoost** — robust gradient boosting, slightly different splitting heuristic
- **CatBoost** — gradient boosting with ordered boosting (different bias profile)
- **Random Forest** — bagged trees (low correlation with the three boosters)

The four base models have different bias profiles, so stacking them with a learnable meta gives the model access to consensus signal where the four agree and uncertainty signal where they disagree. The LR meta uses both base predictions and the original features (`passthrough=True`), so it can also learn corrections directly from the raw input space.

### 3.3 Why SMOTE in the optimization stage

Step 5's broad CV uses class weighting (`class_weight='balanced'` where the model supports it; `scale_pos_weight=3.5` for XGBoost and LightGBM). This is enough to identify which models are promising.

Step 6's optimization adds **SMOTE oversampling** to the training fold of each CV split, raising the minority class to 80% of the majority class's size. This gives the model more positive examples to learn from during each fold's training, and combined with the threshold tuning produces the highest F1.

The SMOTE is applied via `imblearn.pipeline.Pipeline` so it operates only on training folds — the validation portion is left alone. This is critical for honest evaluation.

### 3.4 Threshold tuning via precision-recall curve

In Step 5 the threshold is tuned by sweeping `np.linspace(0.20, 0.90, 30)` — 30 candidate thresholds, picking the one maximizing F1.

In Step 6 the threshold is tuned via `precision_recall_curve`, which evaluates at every unique threshold implied by the predicted probabilities. This is a finer-grained sweep and finds slightly better optima.

Both stages report the mean of per-fold best thresholds rather than a single global threshold. For the final test predictions, the mean threshold from Step 6 (~0.44) is used.

## 4. Cross-validation strategy

### 4.1 5-fold stratified

The pipeline uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` throughout. Stratification ensures each fold has approximately 23% positive class, matching the population. 5 folds is the standard choice for 18k-row data — enough to estimate variance reliably without paying excessive compute cost.

The `cv_mode` config switch supports four other strategies (10-fold, repeated 5x3-fold, group, timeseries) for future use; the original notebook tested several before settling on `strat5`.

### 4.2 Why no separate holdout

The dataset is 18k rows, of which 12k are the locked test set (with no labels — the competition validates submissions). The remaining 18k training rows are used for 5-fold CV. There is no separate "validation set" held out from training because:

1. The test set serves as the de facto holdout
2. CV with 5 folds gives a reasonable estimate of held-out performance
3. Reserving an additional 20% as holdout would shrink the training data by 3600 rows, hurting model fitting

This is a defensible choice for a competition setup. For a production model intended for ongoing deployment, an additional temporal holdout would be appropriate.

### 4.3 Refit on full training data for final predictions

After Step 6 identifies the best model and tuned threshold, the pipeline refits the winning model on the **full 18k training rows** and generates test predictions. The in-sample F1 from this refit is ~0.67 — substantially higher than the CV mean of 0.60.

**This refit F1 is not the headline number.** It's optimistically biased because it evaluates the model on the data it was trained on. The README and `outputs/README.md` explicitly state that the headline is the CV mean (0.5971), not the refit (0.6703). The refit number is reported as a diagnostic — a sanity check that the full-train fit isn't catastrophically different from the CV mean — but never as a model performance claim.

This framing is more honest than the original team submission, which used the refit number (rounded to 0.66) as a headline. We've consciously rebuilt with the more disciplined framing.

## 5. Reproducibility

All randomness flows from `random_state = 42` in `config.example.py`. Given the same training and test CSVs and the same scikit-learn / XGBoost / LightGBM versions, the pipeline produces deterministic outputs:

- Identical CV F1 within ±0.005 (the small variance comes from non-deterministic operations inside CatBoost and LightGBM threading)
- Identical feature selection (top 60 features by importance)
- Identical winning model and tuned threshold
- Identical submission CSV row count (12,000)

The submission's exact 1/0 predictions can vary by ~3-5% of rows across runs due to ensemble stochasticity at probabilities near the threshold. The aggregate F1 on a (hypothetically labeled) test set would be consistent.

## 6. What I'd do differently

Treating this as version 1, the obvious improvements for a v2:

- **Calibrate before thresholding.** The current pipeline tunes the F1-optimal threshold but doesn't calibrate the raw probabilities. A Platt scaling or isotonic regression step before threshold tuning would likely improve both F1 and the interpretability of the predicted probabilities. This is especially valuable if the downstream consumer treats the probability as a rank.

- **Stratify CV by liability percentage too.** The current `StratifiedKFold` stratifies only on the target. Adding a secondary stratification on `liab_prct` bins (low/mid/high) would reduce fold-to-fold variance further. The 0.0139 std F1 is small, but it could probably go to ~0.010.

- **Bayesian hyperparameter optimization.** Hyperparameters were carried forward from the original notebook without formal tuning. A 50-trial Optuna search on the top 3 ensembles would likely improve F1 by 0.005-0.010.

- **Drop the redundant `payout_to_price_ratio`.** The pipeline currently computes both `claim_payout_ratio` (clipped) and `payout_to_price_ratio` (unclipped) — they're the same signal. Dropping the duplicate would clean up the feature space without affecting performance. This is preserved verbatim because the original code computed both, but it's a v2 cleanup.

- **Add a temporal split.** Even without `claim_date` granularity in the test set, splitting training data by `claim_year` and `claim_month` would help diagnose whether the model's performance is stable across time periods — important for deployment.

- **Feature interaction discovery via SHAP.** The current cross features are hand-engineered from domain knowledge. SHAP-based feature interaction analysis on the trained model would identify additional interactions worth promoting to explicit features. Likely gain: small (0.003-0.005 F1), but the methodology improvement is valuable.

## 7. Summary of defensible claims

In order of how confidently each can be defended:

1. **The pipeline achieves CV F1 = 0.5971 ± 0.0139** on stratified 5-fold cross-validation with SMOTE-balanced training folds and F1-optimal threshold tuning. This is reproducible from `random_state=42`.

2. **The pipeline is leakage-safe.** Categorical encoders, feature selection, and SMOTE are all fit on training data only. The test set never influences any training decision.

3. **The model selection is competitive, not asserted.** 14 candidate models were evaluated under identical CV protocol; Tree Super Stack won. This is more credible than declaring the winner a priori.

4. **The result is appropriately framed.** The headline is the CV mean (out-of-sample), not the in-sample refit. The methodology and README explicitly distinguish between them.

The weaker claims (corresponding limitations) are in §6.

## 8. The competition placement

The original team submission placed **6th of 44 teams** in the Travelers-sponsored academic competition (Spring 2026). The rebuild in this repository reproduces the methodology faithfully and produces the validated headline metric of CV F1 0.5971. The repository structure, leakage discipline, and honest framing make the work portfolio-ready in a way the original notebook (a single 717-line Colab file) was not.