# Student Performance Analysis

An academic-style machine learning project predicting university students' end-of-term grades from demographic, family, and study-habit features, with the goal of producing thesis-quality methodology, evaluation, and write-up.

## Author

[Your name] — Teaching/Research Assistant, Department of Physics, KNUST. Project advised in collaboration with Claude (Anthropic) acting as a research mentor.

## Research questions

1. Which demographic, socioeconomic, and study-behavior factors are most predictive of end-of-term academic grade in this cohort?
2. How well can machine learning models classify student performance (8-class outcome) compared to the benchmark reported in the source literature?
3. Do predictive patterns and feature importances replicate across courses, or are they course-specific (i.e., is pooling justified)?

## Dataset

**Higher Education Students Performance Evaluation Dataset**
Yılmaz, N., & Şekeroğlu, B. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C51G82
Collected from the Faculty of Engineering and Faculty of Educational Sciences, Near East University (Cyprus), 2019. 145 students, 32 self-report predictors, 1 target (GRADE, 8 ordinal classes).

Full variable coding: [`docs/data_dictionary.md`](docs/data_dictionary.md)

Benchmark from source study: Radial Basis Function Neural Network, ~88% classification accuracy (Yılmaz & Şekeroğlu, 2020, ICSCCW-2019 proceedings, AISC vol. 1095).

## Project structure

```
├── data/
│   ├── raw/            # Original, untouched dataset
│   └── processed/       # Cleaned/engineered datasets (generated, not hand-edited)
├── notebooks/            # Exploratory and modeling notebooks, numbered by stage
├── src/
│   ├── data/             # Data loading & cleaning scripts
│   ├── features/         # Feature engineering
│   ├── models/           # Training & evaluation scripts
│   └── visualization/    # Plotting utilities
├── reports/
│   ├── figures/          # Exported plots for the write-up
│   └── literature/       # Annotated literature review notes
└── docs/
    └── data_dictionary.md
```

## Project stages (tracked via Git commits)

- [x] **Stage 0** — Project scaffold, data dictionary, literature grounding
- [x] **Stage 1** — Exploratory Data Analysis (univariate, bivariate, target imbalance, course-level effects) — see [`reports/eda_findings.md`](reports/eda_findings.md)
- [x] **Stage 2** — Preprocessing & feature engineering pipeline — see [`reports/stage2_preprocessing_methodology.md`](reports/stage2_preprocessing_methodology.md)
- [x] **Stage 3** — Baseline modeling (logistic regression, k-NN) — see [`reports/stage3_baseline_findings.md`](reports/stage3_baseline_findings.md)
- [ ] **Stage 4** — Ensemble modeling (Random Forest, Gradient Boosting) + hyperparameter tuning
- [ ] **Stage 5** — Model evaluation (cross-validated, macro-F1, confusion matrices, comparison to literature benchmark)
- [ ] **Stage 6** — Interpretability (permutation importance, SHAP)
- [ ] **Stage 7** — Thesis-style write-up

## Methodological notes

- All 32 predictors are ordinal/nominal-coded survey responses, not raw continuous measurements — treated accordingly in preprocessing and EDA (see data dictionary). Each was individually classified as ordinal vs. nominal rather than assumed from its integer coding (see `src/features/feature_types.py`).
- The target is imbalanced (Fail class is the smallest); evaluation uses macro-averaged metrics, not raw accuracy alone.
- `course_id` is the strongest single predictor (Kruskal-Wallis p<0.00001) but has highly uneven sample sizes across courses and is partially confounded with other features (e.g. sex). Decision: every model is trained and evaluated **with and without** `course_id` as a deliberate experimental contrast, not a silent default.
- Given n = 145, all reported model performance uses a single fixed 5-fold stratified cross-validation split (saved to `data/processed/cv_folds.csv`) rather than re-splitting per model, and results are interpreted with appropriate caution about generalizability.

## Reproducing this project

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## License

Code: MIT License (see `LICENSE`). Dataset: CC BY 4.0, per UCI Machine Learning Repository terms — credit Yılmaz & Şekeroğlu (2019) when reusing.
