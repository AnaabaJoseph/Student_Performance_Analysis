# Data Dictionary — Higher Education Students Performance Evaluation Dataset

**Source:** Yılmaz, N., & Şekeroğlu, B. (2019). *Higher Education Students Performance Evaluation* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51G82

**Original study:** Yılmaz, N., & Şekeroğlu, B. (2020). Student Performance Classification Using Artificial Intelligence Techniques. In *10th International Conference on Theory and Application of Soft Computing, Computing with Words and Perceptions (ICSCCW-2019)*, Advances in Intelligent Systems and Computing, vol. 1095. Springer, Cham.

**Collection context:** Faculty of Engineering and Faculty of Educational Sciences, Near East University (Cyprus), 2019. Self-report questionnaire administered to students, end-of-term course grade as outcome.

**Shape:** 145 students × 33 columns (32 predictors + 1 target), no missing values, no duplicate rows.

| # | Column (raw) | Description | Coding |
|---|---|---|---|
| — | STUDENT ID| Anonymized student identifier | string, drop before modeling |
| 1 | Student Age | Age bracket | 1: 18–21, 2: 22–25, 3: above 26 |
| 2 | Sex | Sex | 1: female, 2: male |
| 3 | Graduated high-school type | Type of high school | 1: private, 2: state, 3: other |
| 4 | Scholarship type | Scholarship coverage | 1: None, 2: 25%, 3: 50%, 4: 75%, 5: Full |
| 5 | Additional work | Works alongside studies | 1: Yes, 2: No |
| 6 | Regular artistic/sports activity | Participates regularly | 1: Yes, 2: No |
| 7 | Has a partner | Romantic partner | 1: Yes, 2: No |
| 8 | Total salary (if any) | Monthly income bracket (USD) | 1: 135–200, 2: 201–270, 3: 271–340, 4: 341–410, 5: above 410 |
| 9 | Transportation to university | Mode | 1: Bus, 2: Private car/taxi, 3: Bicycle, 4: Other |
| 10 | Accommodation type | Living situation in Cyprus | 1: rental, 2: dormitory, 3: with family, 4: other |
| 11 | Mother's education | Highest level | 1: primary, 2: secondary, 3: high school, 4: university, 5: MSc, 6: PhD |
| 12 | Father's education | Highest level | same scale as #11 |
| 13 | # siblings | Sibling count bracket | 1:1, 2:2, 3:3, 4:4, 5: 5+ |
| 14 | Parental status | Marital status of parents | 1: married, 2: divorced, 3: deceased (one/both) |
| 15 | Mother's occupation | Job category | 1: retired, 2: housewife, 3: government officer, 4: private sector, 5: self-employed, 6: other |
| 16 | Father's occupation | Job category | 1: retired, 2: government officer, 3: private sector, 4: self-employed, 5: other |
| 17 | Weekly study hours | Self-reported study time | 1: none, 2: <5h, 3: 6–10h, 4: 11–20h, 5: >20h |
| 18 | Reading freq. (non-scientific) | Books/journals | 1: none, 2: sometimes, 3: often |
| 19 | Reading freq. (scientific) | Books/journals | 1: none, 2: sometimes, 3: often |
| 20 | Seminar/conference attendance | Department-related events | 1: Yes, 2: No |
| 21 | Impact of projects/activities on success | Self-assessed | 1: positive, 2: negative, 3: neutral |
| 22 | Class attendance | Frequency | 1: always, 2: sometimes, 3: never |
| 23 | Midterm prep — companionship | Study mode | 1: alone, 2: with friends, 3: not applicable |
| 24 | Midterm prep — timing | Study schedule | 1: closest to exam, 2: regularly during semester, 3: never |
| 25 | Note-taking in class | Frequency | 1: never, 2: sometimes, 3: always |
| 26 | Listening in class | Frequency | 1: never, 2: sometimes, 3: always |
| 27 | Discussion improves interest/success | Self-assessed | 1: never, 2: sometimes, 3: always |
| 28 | Flipped-classroom attitude | Perceived usefulness | 1: not useful, 2: useful, 3: not applicable |
| 29 | Cumulative GPA (last semester, /4.00) | Prior performance | 1: <2.00, 2: 2.00–2.49, 3: 2.50–2.99, 4: 3.00–3.49, 5: >3.49 |
| 30 | Expected graduation GPA (/4.00) | Self-expectation | same scale as #29 |
| 31 | COURSE ID | Course identifier (9 distinct courses) | categorical, 1–9 |
| 32 | **GRADE (target)** | End-of-term output grade | 0: Fail, 1: DD, 2: DC, 3: CC, 4: CB, 5: BB, 6: BA, 7: AA |

## Notes for modeling

- **All predictors are ordinal/nominal-coded integers**, not continuous measurements — this has direct implications for preprocessing (treat as categorical/ordinal, not scale numerically without justification) and for choice of EDA statistics (mode/Cramér's V over mean/Pearson r for nominal fields).
- **Target imbalance observed in this copy:** class 0 (Fail) has only 8 of 145 students; class 1 (DD) has 35. This must be addressed explicitly (stratified splits, appropriate metrics — macro-F1 rather than accuracy alone, possibly class-weighting) and reported as a limitation.
- **COURSE ID is unevenly distributed** (66/145 students from a single course) — a potential confound/clustering effect worth testing (do grading patterns differ by course?) before pooling all courses into one model.
- **GRADE 29/30 (GPA brackets) are themselves bucketed**, so treating them as ordinal-numeric (1–5) is a reasonable simplification but should be stated as an assumption, not silently assumed.
- Original study (Yılmaz & Şekeroğlu, 2019) modeled this as an 8-class classification problem and reported best results with a Radial Basis Function Neural Network (~88% accuracy on their split) — this is a useful **benchmark to cite and compare against**, not a target to chase blindly, since their evaluation protocol details (CV vs. holdout) should be checked before treating 88% as directly comparable.
