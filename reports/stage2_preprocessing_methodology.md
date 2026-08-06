# Stage 2 — Preprocessing & Feature Engineering: Methodology

## 1. Feature typing: ordinal vs. nominal

Every predictor arrives integer-coded, but integer coding does not imply a numeric scale.
Each of the 30 student-level predictors was individually inspected and classified
as ordinal (has a real order, kept as-is) or nominal (no real order, one-hot encoded). Full
reasoning per feature: [`src/features/feature_types.py`](../src/features/feature_types.py).

Result: **15 ordinal, 15 nominal.** Three borderline cases worth flagging explicitly:

- `class_attendance` (always/sometimes/never) is ordinal but the scale runs in the *opposite*
  direction to "more attendance is better" — kept as ordinal since the order itself is real,
  but coefficient/SHAP signs must be interpreted with this in mind later.

- `project_impact` (positive/negative/neutral) and `midterm_prep_timing` (closest-to-exam/
  regularly-during-semester/never)
  
  These two were coded by the original survey as 1/2/3, but those
  numbers do **not** correspond to a quality ranking — treating them as ordinal would silently
  tell the model "neutral > negative" and "never preparing > cramming," which is not a
  defensible claim. Both are treated as nominal.

## 2. course_id is a separate experimental axis, not a routine encoding choice

Per the Stage 1 finding (course is the strongest predictor, p<0.00001, but with wildly uneven
sample sizes per course) and the decision to model both ways: `build_features.py` produces two
parallel feature sets:

- **`include_course=True`**: course_id one-hot encoded alongside the other 30 predictors
- **`include_course=False`**: course_id excluded entirely

Every model in Stages 3–5 will be trained and evaluated under **both** conditions, on identical
CV folds, so the contrast itself becomes a reportable result (e.g., "how much of model
performance is actually course identity vs. genuine behavioral signal?").

## 3. Encoding

- **Ordinal features**: passed through unchanged (already a valid integer scale, e.g.
  1=primary education ... 6=PhD).

- **Nominal features (+ course_id when included)**: one-hot encoded, `drop='if_binary'` to avoid
  redundant columns for true binary fields (sex, additional_work, etc.).

- Result: 30 raw predictors → 56 encoded columns (without course) or 65 (with course; 9 course
  dummies replacing the binary drop). Verified in `src/features/build_features.py`.

- **Encoding fit happens inside the CV loop, per fold** to avoid any leakage and will be enforced via   `sklearn.pipeline.Pipeline` in Stage 3.


## 4. Multicollinearity decision: `gpa_last_semester` vs. `gpa_expected_graduation`

Stage 1 found these two moderately correlated (Spearman ρ=0.65): the only multicollinear pair
in the dataset. Decision: **keep both, and won't be dropped or merged.**

Reasoning:
- ρ=0.65 is moderate, not severe (VIF will be checked explicitly in Stage 3 for the linear
  baseline; tree-based ensembles are not meaningfully affected by this level of collinearity).

- They are conceptually distinct: one is documented prior performance, the other is a
  self-reported *expectation*, the gap between the two (over- or under-confidence) may itself
  carry signal that neither variable alone captures. Dropping one pre-emptively would discard
  that without testing it.

- If the linear baseline in Stage 3 shows unstable coefficients for this pair, I will revisit.
  (e.g., add an explicit "expectation gap" engineered feature)

## 5. Train/validation protocol

Given n=145 and a smallest class of 8 (Fail), a single train/test holdout would leave too few
Fail examples in whichever split they land in to evaluate reliably. Decision: **5-fold stratified
cross-validation**, fixed and saved once (`random_state=42`) to `data/processed/cv_folds.csv` via
`src/data/make_cv_folds.py`, so every model in every later stage is compared on **identical**
folds, required for the model and feature-set comparisons to be valid rather than
artifacts of different random splits.

Fold sizes: 29/29/29/29/29. Class 0 (Fail, n=8) is spread 1–2 per fold thin, but unavoidable
given the class size; will be flagged as a limitation in the final write-up, and macro-F1
will be reported with appropriate caution for this class.

## Artifacts produced this stage

- `src/features/feature_types.py` — ordinal/nominal classification + justification
- `src/features/build_features.py` — `build_preprocessor()`, `prepare_dataset()`
- `src/data/make_cv_folds.py` — fixed CV fold generator
- `data/processed/cv_folds.csv` — saved fold assignment (student_id, grade, fold)
- `notebooks/02_preprocessing_feature_engineering.ipynb` — reproducible walkthrough
