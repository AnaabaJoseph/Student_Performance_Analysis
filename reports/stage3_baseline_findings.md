# Stage 3 — Baseline Modeling: Findings

**Models:** Logistic Regression (nominal multiclass), Logistic Regression (ordinal, via Frank &
            Hall 2001 binary decomposition — see `src/models/ordinal_classifier.py`), k-NN (k=5).

**Evaluation:** 5-fold stratified CV (fixed folds from Stage 2), encoders refit per fold to
                avoid leakage. Metrics: macro-F1 (primary), accuracy (secondary), MAE of grade index (key
                metric for the nominal vs. ordinal question).

Full numbers: `reports/stage3_baseline_results.csv`. Chart: `reports/figures/06_baseline_model_comparison.png`.
--------------------------------------------------------------------
| Feature set    | Model        | Macro-F1      | Accuracy | MAE   |
--------------------------------------------------------------------
| with_course    | LR (nominal) | 0.247 ± 0.052 | 0.303    | 1.455 |
| with_course    | k-NN (k=5)   | 0.244 ± 0.038 | 0.338    | 1.676 |
| with_course    | LR (ordinal) | 0.244 ± 0.058 | 0.283    | 1.379 |
| without_course | k-NN (k=5)   | 0.211 ± 0.070 | 0.331    | 1.917 |
| without_course | LR (nominal) | 0.143 ± 0.048 | 0.193    | 2.048 |
| without_course | LR (ordinal) | 0.122 ± 0.055 | 0.152    | 2.048 |
--------------------------------------------------------------------
## Finding 1: course_id carries most of the learnable signal

Every model performs substantially better with `course_id` included (macro-F1 roughly +0.07 to
+0.12 absolute, MAE roughly 0.4–0.6 grade-points lower). This is the direct, quantified
confirmation of the Stage 1 EDA finding that course identity is not a nuisance variable, it is the
dominant predictor available in this dataset. 

## Finding 2: nominal vs. ordinal framing — a genuine, non-obvious result

Macro-F1 and accuracy are essentially tied between nominal and ordinal logistic regression
(within one standard deviation of each other in both feature-set conditions) — ordinal framing
does **not** improve raw classification correctness here. But **MAE tells a different story**:
under `with_course`, ordinal LR has the lowest MAE of all six configurations (1.379 vs. 1.455
for nominal LR), meaning **when the ordinal model is wrong, it tends to be wrong by less**, it
predicts BB instead of BA rather than Fail instead of AA. Nominal classification has no
mechanism to prefer "close" mistakes over "far" mistakes, since it treats all 8 classes as
unrelated labels; ordinal classification's structure inherently penalizes distant errors more
during training.

## Finding 3: simple baselines are all roughly in the same ballpark

No baseline dramatically outperforms the others within a feature-set condition, macro-F1 spans
only ~0.24–0.25 (with_course) or ~0.12–0.21 (without_course). This sets the floor that Stage 4
(Random Forest / Gradient Boosting) needs to clear meaningfully to justify the added complexity,
not just nominally beat by a fraction of a point.

## Comparison to the literature benchmark

The original study (Yılmaz & Şekeroğlu, 2020) reports ~88% accuracy with a Radial Basis Function
Neural Network. Our best accuracy here (k-NN, with_course: 33.8%) is far below that. This gap is
worth investigating rather than treating as a failure 

## Known limitation

`course_id` includes categories with very few students (course 2: n=2). In some CV folds,
validation students from a thin course had a `course_id` value not seen during that fold's
training split — `handle_unknown='ignore'` in the encoder handles this gracefully (treated as
all-zero dummy) but effectively means the model has no course-specific information for those
students in that fold. This is a structural limitation of the dataset, not a bug.

## Carried into Stage 4

- Try Random Forest and Gradient Boosting under both feature-set conditions, same fixed folds, same three metrics.
- Investigate per-class performance (confusion matrix) for the best baseline — macro-F1 alone hides which classes are failing.
- Revisit the literature benchmark gap once tree-based models are in hand.
