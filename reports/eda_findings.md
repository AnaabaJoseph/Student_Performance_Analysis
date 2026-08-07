# Stage 1 — Exploratory Data Analysis: Findings

**Data:** 145 students, 32 predictors, 1 ordinal target (`grade`, 0=Fail to 7=AA). No missing values, no duplicates.

## 1. Target distribution

`reports/figures/01_target_distribution.png`

Moderately imbalanced. The two largest classes are DD (n=35) and DC (n=24); the smallest is Fail (n=8). Grades AA, BA, BB, CB, CC are all in the 10–21 range. This rules out plain accuracy as a sufficient metric. **Decision: macro-F1 / balanced accuracy will be used as primary metrics, accuracy as secondary.**

## 2. Course is a dominant factor and a structural confound

`reports/figures/02_grade_by_course.png`

A Kruskal-Wallis test shows grade differs significantly across the 9 courses (H = 80.79, p < 0.00001). Mean grade by course ranges from 1.36 (course 8) to 6.33 (course 7), which is a huge spread, larger than the spread explained by any individual student-level predictor. Course sample sizes are also wildly uneven (course 1: n=66; course 2: n=2).

**IMPLICATION:** Course identity is likely the single strongest predictor in the dataset. There is two options:

    (a) Include `course_id` as a categorical feature and let the model use it - defensible.

    (b) Analyze student-behavior effects **within course**, or control for course explicitly (e.g., stratify, or model grade residualized against course).

## 3. Sex shows a strong marginal association but it is confounded by course

`reports/figures/03_grade_by_sex.png`

Pooled across all courses, sex is significantly associated with grade (Mann-Whitney U, p = 0.00007; Cramér's V = 0.365, the second-highest of any feature). **However**, cross-tabulating sex against course reveals severe imbalance: course 9 (n=21, mean grade 2.19) is 100% female; course 8 (n=14, mean grade 1.36) is 79% female. These are two of the three lowest-scoring courses. Meanwhile course 1 (the largest, n=66) is 74% male.

This is a textbook confound: sex correlates with grade largely *because* sex correlates with which course a student happens to be in, and courses differ enormously in grading. Looking only within course 1 (n=66, the one course with enough of both groups for a meaningful comparison), the male–female gap shrinks substantially (mean 2.47 vs. 1.76) present, but far less dramatic than the pooled figure implies.

## 4. Prior academic performance is the most defensible behavioral predictor

`reports/figures/04_grade_by_prior_gpa.png`

`gpa_last_semester` is significantly and monotonically associated with `grade` (Spearman ρ = 0.351, p < 0.0001) and it is conceptually unconfounded: past performance plausibly *causes* future performance through student-level ability/habits. It's the most "real" signal in the dataset besides course.

`gpa_expected_graduation` (students' self-reported expectation) correlates with `grade` too (ρ = 0.272) but is itself correlated with `gpa_last_semester` (ρ = 0.654), meaningful multicollinearity. Including both is partly double-counting "prior performance perception." **Decision to make in Stage 2:** keep both and let regularization handle it, or engineer a single combined feature  document.

## 5. Other features have weak-to-moderate marginal associations

Ranked by Cramér's V against `grade` (full table: `reports/eda_feature_associations.csv`):

| Feature                   | Cramér's V | Spearman's ρ | p-value |
|:--------------------------|-----------:|-------------:|---------:|
| course_id                 |      0.376 |        0.135 |    0.106 |
| sex                       |      0.365 |        0.331 |  <0.0001 |
| gpa_last_semester         |      0.262 |        0.351 |  <0.0001 |
| age_group                 |      0.228 |       -0.070 |    0.401 |
| project_impact            |      0.201 |       -0.185 |    0.026 |
| scholarship_type          |      0.180 |        0.065 |    0.441 |
| gpa_expected_graduation   |      0.169 |        0.272 |    0.001 |

Most "study habit" self-report items (weekly study hours, note-taking, listening, discussion benefit, flipped-classroom attitude, midterm prep style) show **weak, often non-significant** marginal association with final grade. It either means 

(a) these self-reported behaviors genuinely don't predict outcome much in this cohort, 
(b) self-report measurement is noisy, or 
(c) their effect is masked by the much larger course-level effect and would show up once we control for course. Stage 3 modeling (with and without course-level controls) should help distinguish (c) from (a)/(b)

## 6. Multicollinearity check

`reports/figures/05_predictor_correlation_heatmap.png`

Only one predictor pair exceeds |ρ| = 0.5: `gpa_last_semester` ↔ `gpa_expected_graduation` (ρ = 0.654), already discussed in §4. No other concerning collinearity among the 30 predictors, tree-based models won't be troubled, and even for linear/logistic baselines, likely don't need aggressive dimensionality reduction.

## Decisions carried into Stage 2 (preprocessing)

1. Primary metric: macro-F1.
2. Stratified train/test or stratified k-fold (target imbalance + small n).
3. Model with and without `course_id` as separate experimental conditions.
4. Keep `sex` in the model but explicitly discuss the course confound in the results/discussion section; the pooled sex effect will not be presented without that caveat.
5. Decide GPA feature handling (keep both vs. combine) before finalizing the feature set.
