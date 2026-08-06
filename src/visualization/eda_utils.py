"""Reusable EDA statistics for the Student Performance Analysis project."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V association strength between two categorical series."""
    confusion = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion)[0]
    n = confusion.sum().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")


def feature_target_associations(df: pd.DataFrame, target: str, exclude: list[str] | None = None) -> pd.DataFrame:
    """Compute Cramer's V and Spearman rho of every predictor against the target column."""
    exclude = set(exclude or []) | {target}
    predictors = [c for c in df.columns if c not in exclude]
    rows = []
    for col in predictors:
        cv = cramers_v(df[col], df[target])
        rho, p = spearmanr(df[col], df[target])
        rows.append({"feature": col, "cramers_v": round(cv, 3), "spearman_rho": round(rho, 3), "spearman_p": round(p, 4)})
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
