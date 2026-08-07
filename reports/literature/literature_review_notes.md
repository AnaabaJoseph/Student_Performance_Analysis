# Literature Review — Seed Notes

Working bibliography to build out as the project progresses. Format: citation, venue/quality signal, relevance, key takeaway.

## Primary source (dataset origin)

**Yılmaz, N., & Şekeroğlu, B. (2020).** Student Performance Classification Using Artificial Intelligence Techniques. In *10th International Conference on Theory and Application of Soft Computing, Computing with Words and Perceptions (ICSCCW-2019)*, Advances in Intelligent Systems and Computing, vol. 1095, Springer, Cham.
- Introduces this exact dataset (145 students, Near East University, Cyprus).
- Best reported model: Radial Basis Function Neural Network, ~88% accuracy on the 8-class GRADE target.

## Survey / methodology grounding

**Zhang, Y., Yun, Y., An, R., Cui, J., Dai, H., & Shang, X. (2021).** Educational Data Mining Techniques for Student Performance Prediction: Method Review and Comparison Analysis. *Frontiers in Psychology*. DOI: 10.3389/fpsyg.2021.698490

**Alyahyan, E., & Düştegör, D. (2020).** Predicting academic success in higher education: literature review and best practices. *International Journal of Educational Technology in Higher Education*, 17(1), 3.

## Comparable small-sample, similarly-coded datasets (contrast/benchmark candidates)

**Cortez, P., & Silva, A. (2008).** Student Performance [Dataset]. UCI Machine Learning Repository. DOI: 10.24432/C5TG7T.
- The well-known Portuguese secondary-school dataset (~649 students, demographic/social/school features, math & Portuguese grades). Different educational level (secondary vs. higher ed) and culture (Portugal vs. Cyprus) — useful as a discussion point on generalizability, not for direct merging.

## Recent applied work using related techniques (to expand)

**Wang, S., & Luo, B. (2024).** Academic achievement prediction in higher education through interpretable modeling. *PLOS ONE*. DOI: 10.1371/journal.pone.0309838
- XGBoost + SHAP for grade prediction (n=87, Japanese-language course, Wuhan). Strong precedent for combining a gradient-boosted model with SHAP interpretability on a small higher-ed sample — directly relevant to our planned Stage 6 (interpretability).
