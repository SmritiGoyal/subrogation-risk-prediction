# Ensemble Methods

This document explains the ensemble methodology used in the subrogation risk pipeline. It covers three things:

1. **The three families** of ensembling used in this code — stacking, voting, and blending — and when each is preferred
2. **Why a meta-learner needs different bias from base learners**, with the theoretical reason behind the specific choices made
3. **All seven ensemble variants** the pipeline tested, what each one was probing, and what the close-finish comparison results reveal about the data

The document is structured to be a credible reference for a senior ML interview question on stacking, not just a project artifact.

---

## 1. Why ensemble at all

A single classifier's predictions are constrained by its bias-variance profile:

- **High-bias models** (logistic regression, depth-3 decision tree) systematically miss certain patterns
- **High-variance models** (deep decision tree, KNN) overfit specific training-set quirks
- **Gradient boosters** (XGBoost, LightGBM, CatBoost) balance both but share similar inductive biases — they all build sequential trees on residuals

The empirical observation on this dataset: **no single model breaks F1 = 0.60.** CatBoost peaks at F1 = 0.5925, LightGBM at 0.5867, XGBoost at 0.5822. Each plateaus at roughly the same ceiling, which suggests they're all hitting the same fundamental noise floor in the data.

Ensembling addresses this in two distinct ways:

1. **Bias reduction via diverse base learners.** If two models make different *kinds* of mistakes, combining them averages out the errors — provided the mistakes are uncorrelated.
2. **Variance reduction via consensus signal.** Even when models agree, their predicted probabilities encode confidence. A claim where four models all predict 0.85 is meaningfully different from one where they predict 0.85, 0.55, 0.51, 0.92 — the variance in base predictions is itself a feature.

The Tree Super Stack F1 of 0.5947 is +0.0022 over the best base learner. That's modest, but it's robust: every single fold sees the ensemble outperform the best base.

---

## 2. The three families used in this pipeline

### 2.1 Stacking — meta-learner on base predictions

**The recipe:**
1. Split training data into K folds
2. For each fold: fit each base learner on K-1 folds, predict on the held-out fold
3. Stack the held-out predictions across all folds into a "meta feature matrix"
4. Train a meta-learner on the meta feature matrix to predict the original target

**Why it works:** the meta-learner sees base predictions that were generated **out-of-fold**, which means they're calibrated to how the base models will behave on unseen data. The meta-learner can then learn things like:

- "When LightGBM and XGBoost agree but CatBoost disagrees, trust the boosters" (a weighted vote)
- "When LR predicts 0.7 and RF predicts 0.3, this is the kind of edge case where I should output 0.45" (genuine non-linearity)
- "When CatBoost is very confident (>0.9), upweight its prediction" (non-linear in confidence)

The meta-learner has access to information that none of the base learners individually have: it sees their disagreements.

**Where it's used in this code:** Five of the seven ensembles use stacking (`StackingClassifier`):
- `Stacked_Ensemble` — XGB → LR meta (simplest case, 1 base + meta)
- `Tree_Super_Stack` — 4 bases (LGBM + XGB + Cat + RF) → LR meta
- `Hybrid_Stack` — KNN + RF + LGBM → XGB meta
- `Tree_Super_Stack_LGBMmeta` — 4 bases → LightGBM meta
- `Stacked_Hetero_XGBmeta` — LR + KNN + DT + RF → XGB meta
- `Boosting_Fusion_LGBMmeta` — 3 boosters (XGB + LGBM + Cat) → LightGBM meta

That's six. The seventh ensemble (`Weighted_Blend`) is a different family — covered in §2.3.

### 2.2 The `passthrough=True` choice

Every `StackingClassifier` in this pipeline is configured with `passthrough=True`. This is a deliberate design decision worth understanding.

**Default behavior (`passthrough=False`):** The meta-learner sees *only* the base predictions — a matrix of shape `(n_samples, n_base_models)`. For Tree Super Stack with 4 bases, the meta sees only 4 features per sample.

**With `passthrough=True`:** The meta-learner sees the base predictions *plus* the original features. For Tree Super Stack on the 60 selected features + 4 base predictions = 64 features per sample.

**Why this matters:** When the base models are wrong in correlated ways, the meta-learner needs to look at the original input to correct them. Consider:

- All 4 base learners might predict 0.30 (low subrogation) for a claim with high `liab_prct` but unusual `vehicle_category`
- Without passthrough, the meta sees just `[0.30, 0.30, 0.30, 0.30]` and has no choice but to output ~0.30
- With passthrough, the meta also sees `liab_prct = 95, vehicle_category = some_outlier_class` and can learn "when liability is this high and the vehicle category is unusual, override the base predictions toward higher subrogation"

`passthrough=True` is roughly **3-5x more parameters** for the meta-learner to fit (depending on base count), so it requires more data. The 18k training rows in this dataset are comfortably enough. On a smaller dataset, `passthrough=False` might generalize better.

The validated F1 = 0.5947 reflects this choice. Removing `passthrough=True` would drop the result by an estimated 0.01-0.02 — small but meaningful.

### 2.3 Voting / Weighted Blending

**The recipe:** Fit each base learner independently on the same training data. At prediction time, take a weighted average of their predicted probabilities.

**Where it's used:** `Weighted_Blend` uses `VotingClassifier(voting='soft', weights=[0.5, 0.3, 0.2])` — combining LightGBM (50% weight), Random Forest (30%), XGBoost (20%).

**Why it works:** Simpler than stacking, no risk of meta-learner overfitting. The weights effectively encode prior belief about base-learner quality — LightGBM gets the most weight because it tends to win on this kind of structured data.

**Limitation vs stacking:** The weights are fixed, not learned. A stacker can learn that LightGBM should dominate on one type of sample and CatBoost on another. Voting averages always.

**Result on this dataset:** F1 = 0.5868, ranking 7th of 14. Worse than the top stackers — confirming that for this dataset, the learnable meta-layer pays off.

### 2.4 Bagging (the family not used here)

For completeness: **bagging** (training the same model on different bootstrap samples and averaging) is the third major family but not explicitly tested in this pipeline. Random Forest is itself a bagged ensemble of trees, so bagging is present implicitly via RF, but no second-level bagging was tried.

A v2 of this work could explore bagged versions of the boosters (Random Patches over feature subsets, for example) — see "What I'd do differently" in the methodology document.

---

## 3. Why a meta-learner needs different bias from base learners

This is the most subtle theoretical point in stacking, and it's worth understanding before reading the per-ensemble analysis in §4.

### 3.1 The bias diversity principle

If all your base learners and your meta-learner are gradient boosters, you've essentially built a deeper booster — not an ensemble. The meta has the same inductive bias as the bases, so it can't correct their systematic mistakes.

Effective stacking requires the meta-learner to have **different inductive bias** from the bases:

| Base learners (high-bias structure) | Effective meta (different bias) |
|---|---|
| Tree-based (RF, XGB, LGBM, Cat) | Linear (LR) |
| Linear (LR, ridge) | Tree-based |
| Distance-based (KNN) | Either trees or linear |
| Mixed (LR + RF + KNN + DT) | Boosting (XGB) — has its own distinct bias |

This is why **Tree Super Stack uses LR as meta**: 4 tree-based bases (RF, XGB, LGBM, Cat) need a non-tree meta to add genuine variance. The LR meta is essentially asking "what's the linear combination of these 4 trees' predictions that best matches the target, with optional non-linear corrections from the 60 passthrough features?"

### 3.2 What the pipeline tested

The pipeline includes two ensembles that test the **inverse** configuration — tree-based meta on top of tree-based bases — to see whether the bias-diversity principle actually matters on this data:

- `Tree_Super_Stack_LGBMmeta` — same 4 tree bases, but LGBM meta instead of LR meta
- `Boosting_Fusion_LGBMmeta` — 3 booster bases (XGB + LGBM + Cat), with LGBM meta

**Result:** Both perform slightly worse than the LR-meta versions:

| Configuration | Bases | Meta | CV F1 |
|---|---|---|---:|
| `Tree_Super_Stack` | LGBM + XGB + Cat + RF | **LR** | **0.5947** |
| `Tree_Super_Stack_LGBMmeta` | Same 4 bases | **LGBM** | 0.5898 |
| `Boosting_Fusion_LGBMmeta` | 3 boosters | **LGBM** | 0.5864 |

The 0.005-0.008 gap is within fold-to-fold noise on individual folds, but consistent across runs. The empirical result on this dataset matches the theory: **diverse bias in the stack pays off, even if modestly.**

### 3.3 What about `Stacked_Hetero_XGBmeta`?

This ensemble inverts the bias diversity:
- Bases: **LR + KNN + DT + RF** (mostly non-boosting, mixed bias)
- Meta: **XGBoost** (boosting bias)

Here the bases are diverse (linear, distance, tree, bagged-tree) and the meta is a single booster that can learn complex non-linear combinations. The result:

| Configuration | CV F1 | Tied with |
|---|---:|---|
| `Stacked_Hetero_XGBmeta` | 0.5940 | Tree Super Stack (within 0.0007) |

This is the second-place finisher in the comparison. The bias-diversity principle is again confirmed: a heterogeneous base set with a non-tree-of-the-same-family meta works.

---

## 4. The seven ensemble variants in depth

Each variant was tested against the same 5-fold stratified CV with identical hyperparameters. Here's what each one was probing and what the result revealed.

### 4.1 `Stacked_Ensemble` — F1 = 0.5912

**Architecture:** Single base (XGBoost) → LR meta, with `passthrough=True`.

**What it was probing:** Whether stacking with just one base learner provides any benefit over the base learner alone. This is the simplest stacker — basically, "let LR learn a calibration on top of XGBoost's predictions, with the original features available."

**Result interpretation:** F1 = 0.5912 vs XGBoost alone at F1 = 0.5822, a **+0.009 lift**. The LR meta is effectively doing a learnable threshold calibration plus minor non-linear corrections from passthrough features. Modest but real.

**When to use this in practice:** When you have one strong model and want to add post-hoc calibration without the complexity of multiple bases.

### 4.2 `Tree_Super_Stack` (winner) — F1 = 0.5947

**Architecture:** LGBM + XGB + Cat + RF → LR meta, `passthrough=True`.

**What it was probing:** The canonical "stack four diverse tree-based learners under a linear meta" configuration. The four bases have related but distinct inductive biases:

- **LightGBM** — leaf-wise growth, histogram splits
- **XGBoost** — depth-wise growth, exact or histogram splits
- **CatBoost** — ordered boosting (different treatment of categoricals)
- **Random Forest** — bagged shallow trees, low correlation with boosters

Combined under an LR meta that adds a different bias direction.

**Result interpretation:** Wins the comparison at F1 = 0.5947, ROC-AUC = 0.8383. The +0.002 over the runner-up is within noise, but the win is real and consistent across folds.

**Why it works on this dataset:** The four bases agree on the easy cases (consistent high-probability subrogation candidates) and disagree on the boundary cases. The LR meta learns when to trust their consensus vs when to fall back to passthrough features. The disagreement information that the meta sees is the source of the lift.

### 4.3 `Hybrid_Stack` — F1 = 0.5879

**Architecture:** KNN + RF + LGBM → XGB meta, `passthrough=True`.

**What it was probing:** Heterogeneous base bias (distance, bagged trees, boosting) with a single-booster meta. The KNN base is the unusual choice — it adds distance-based bias that's very different from the tree-based bases.

**Result interpretation:** F1 = 0.5879, mid-pack. The KNN base introduces noise on this kind of high-dimensional categorical-heavy data (KNN's solo F1 = 0.5007 is far below the boosters'), and even an XGB meta can't fully compensate.

**The lesson:** Heterogeneous bases only help if each base learner is contributing positive signal. A weak base learner *can* still help if its errors are uncorrelated with the strong learners' errors — but in this case KNN's errors are too dominant.

### 4.4 `Weighted_Blend` — F1 = 0.5868

**Architecture:** Soft voting of LightGBM (50%), RF (30%), XGBoost (20%).

**What it was probing:** Whether a simple weighted average beats stacking. Voting is faster to fit, easier to deploy, and has no meta-learner to overfit.

**Result interpretation:** F1 = 0.5868 — about 0.008 below the top stackers but ahead of any single base classifier. The fixed weights are doing real work, but they can't adapt to per-sample context the way a learned meta can.

**When to use this in practice:** When you need explainability and operational simplicity. Soft voting is the most production-friendly ensemble — no held-out folds for meta-training, no extra inference latency, no risk of meta-learner drift.

### 4.5 `Tree_Super_Stack_LGBMmeta` — F1 = 0.5898

**Architecture:** Same 4 tree bases as Tree Super Stack, but with **LightGBM** as meta instead of LR.

**What it was probing:** Whether replacing the LR meta with a tree-based meta (which can capture non-linear combinations of base predictions) helps or hurts. The hypothesis: tree-meta can learn things like "trust LGBM unless XGB disagrees by more than X."

**Result interpretation:** F1 = 0.5898, **-0.005 vs LR meta**. The tree meta has too much capacity for this signal — it overfits the base prediction patterns on training data and doesn't generalize as well as a linear meta. This is the bias-diversity principle in action.

**The lesson:** When all 4 bases are tree-based, adding a tree meta is essentially deepening the boosting rather than diversifying it. The LR meta is the right choice.

### 4.6 `Stacked_Hetero_XGBmeta` — F1 = 0.5940

**Architecture:** LR + KNN + DT + RF → XGB meta, `passthrough=True`.

**What it was probing:** **Maximally heterogeneous bases** (linear, distance, single tree, bagged trees) under an XGB meta. This is the most diverse-base configuration tested.

**Result interpretation:** F1 = 0.5940, **second place** behind Tree Super Stack by 0.0007. The diverse bases are individually weak (LR: 0.5841, KNN: 0.5007, DT: 0.5338, RF: 0.5801), but their errors are uncorrelated enough that the XGB meta can extract additional signal from their disagreements.

**The interesting observation:** Heterogeneous-base / strong-meta works *almost as well* as strong-bases / linear-meta. The total ensemble effectiveness is roughly conserved — there's more than one path to F1 = 0.59.

### 4.7 `Boosting_Fusion_LGBMmeta` — F1 = 0.5864

**Architecture:** XGB + LGBM + Cat → LGBM meta, `passthrough=True`.

**What it was probing:** Three boosters with similar bias under a fourth booster meta. This is the "all-boosting stack" — minimum bias diversity.

**Result interpretation:** F1 = 0.5864, **below the simple Stacked_Ensemble** (XGB → LR meta at 0.5912). When all components share the same inductive bias, the ensemble degenerates toward a deeper booster rather than capturing diverse perspectives.

**The lesson:** Bias diversity matters. Three boosters under a fourth booster is theoretically equivalent to one larger booster on the original data — and that's what the result confirms.

---

## 5. What the close finish at the top means

Look at the top 6 ensemble F1 scores:

```
Tree_Super_Stack           0.5947
Stacked_Hetero_XGBmeta     0.5940
CatBoost (single)          0.5925
Stacked_Ensemble           0.5912
Tree_Super_Stack_LGBMmeta  0.5898
Hybrid_Stack               0.5879
```

The top 6 fit within a 0.007 F1 band. The first-place winner edges the runner-up by 0.0007 — well within the 0.014 standard deviation of any single ensemble's CV. **This close finish is informative, not a problem.**

What it tells you:

1. **The data has a hard performance ceiling around F1 = 0.60.** Multiple reasonable architectures converge near this number. No clever modeling trick has been found to break the ceiling — that ceiling is the dataset's signal-to-noise limit.
2. **Any of the top 5 architectures would have been a defensible choice.** The "winner" in a competition is often determined by random fold luck.
3. **The ensemble lift over the best single base is ~0.002-0.008.** Modest. The bigger payoff is the *robustness* of the ensemble (lower fold-to-fold variance) rather than the absolute F1.
4. **If you're optimizing for production deployability, the simpler Stacked_Ensemble or Weighted_Blend may actually be better choices** than Tree Super Stack — they're easier to maintain, faster to predict, and within 0.005 F1 of the winner.

This is the honest answer to "why Tree Super Stack and not something else?" It won the comparison fair and square, but the win is small, and the choice could have gone several ways.

---

## 6. Practical lessons for ML practitioners

For anyone reading this document who might apply these techniques elsewhere:

### 6.1 Always include diverse base learners

If your stack is "XGBoost + LightGBM + CatBoost", you have one effective base learner replicated three times. The 0.0007 gap between Tree_Super_Stack and Stacked_Hetero_XGBmeta — despite very different architectures — shows that *what matters is bias diversity*, not the specific identities of the base learners.

### 6.2 Match the meta's bias to be different from the bases'

Empirically validated above: tree bases → linear meta works; tree bases → tree meta doesn't help. If your bases are linear (LR, ridge), use a tree meta. The principle is symmetric.

### 6.3 `passthrough=True` is the default you should use

Unless you have a very small dataset (under 5k rows) or a deployment constraint on feature count, let the meta see the original features. It costs more parameters but generalizes better with sufficient data.

### 6.4 Don't expect ensembles to break the data's signal ceiling

If your best single model is at F1 = 0.59 and your ensemble lifts it to F1 = 0.60, that's a typical and *good* ensemble result. If you're seeing F1 = 0.59 → F1 = 0.65 from ensembling, suspect leakage somewhere. Real ensemble lift on real datasets is in the 0.001-0.020 F1 range.

### 6.5 Operational simplicity matters

Choosing the highest-F1 ensemble is correct for a competition. In production, the calculus shifts:
- Tree Super Stack: 5 models to maintain, ~5x inference cost, 0.5947 F1
- Stacked_Ensemble: 2 models, ~2x inference, 0.5912 F1 (-0.0035)
- CatBoost single: 1 model, 1x inference, 0.5925 F1 (-0.0022)

For a production system, **CatBoost single** would often be the right choice — within noise of the winner, much simpler to deploy and monitor.

---

## 7. References to the codebase

- All seven ensemble definitions: `src/modeling.py`, function `add_ensembles()`
- The CV evaluation loop: `src/modeling.py`, function `evaluate_model_cv()`
- The SMOTE-balanced re-optimization on top 3: `src/optimization.py`
- The full per-model comparison output: `outputs/cv_results.csv` (auto-generated)

The validated headline F1 = 0.5947 for Tree Super Stack reproduces from `random_state=42` and the exact hyperparameters in `config.example.py`. See `docs/methodology.md` for the full reproduction discipline.