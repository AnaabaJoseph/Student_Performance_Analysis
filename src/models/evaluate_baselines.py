"""
Stage 3 — Baseline model evaluation.

Runs every baseline model (nominal logistic regression, ordinal logistic regression via
Frank & Hall decomposition, k-NN) under both feature-set conditions (with / without course_id),
using the FIXED stratified folds saved in data/processed/cv_folds.csv. Encoding (OneHotEncoder)
is fit fresh inside each training fold, never on validation data, to avoid leakage.

Metrics reported:
  - macro_f1   : primary metric (treats all 8 classes equally regardless of size)
  - accuracy   : secondary, reported for comparability with the original study's benchmark
  - mae        : mean absolute error of the predicted grade index vs true grade index --
                 the key metric for the nominal-vs-ordinal comparison, since it captures
                 whether a model's mistakes are "close" (predicting BB instead of BA) or
                 wildly wrong (predicting AA instead of Fail). Nominal classifiers have no
                 notion of "close," so this metric is where ordinal modeling should show
                 its advantage, if it has one.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_raw
from src.features.build_features import build_preprocessor, prepare_dataset
from src.models.ordinal_classifier import OrdinalLogisticClassifier

CV_FOLDS_PATH = "data/processed/cv_folds.csv"
RESULTS_PATH = "reports/stage3_baseline_results.csv"


def get_models():
    return {
        "logistic_regression_nominal": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "logistic_regression_ordinal": OrdinalLogisticClassifier(max_iter=2000, random_state=42),
        "knn_k5": KNeighborsClassifier(n_neighbors=5),
    }


def run_evaluation():
    df = load_raw()
    cv = pd.read_csv(CV_FOLDS_PATH)
    n_folds = cv["fold"].nunique()

    results = []

    for include_course in (True, False):
        condition = "with_course" if include_course else "without_course"
        X_raw, y = prepare_dataset(df, include_course=include_course)

        for model_name, model in get_models().items():
            fold_metrics = {"macro_f1": [], "accuracy": [], "mae": []}

            for fold in range(n_folds):
                train_idx = cv.index[cv["fold"] != fold]
                val_idx = cv.index[cv["fold"] == fold]

                X_train, X_val = X_raw.loc[train_idx], X_raw.loc[val_idx]
                y_train, y_val = y.loc[train_idx], y.loc[val_idx]

                preprocessor: ColumnTransformer = build_preprocessor(include_course=include_course)
                pipe = Pipeline([
                    ("preprocess", preprocessor),
                    ("scale", StandardScaler(with_mean=False)),  # safe with sparse-like OHE output
                    ("model", model),
                ])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_val)

                fold_metrics["macro_f1"].append(f1_score(y_val, preds, average="macro", zero_division=0))
                fold_metrics["accuracy"].append(accuracy_score(y_val, preds))
                fold_metrics["mae"].append(mean_absolute_error(y_val, preds))

            results.append({
                "feature_set": condition,
                "model": model_name,
                "macro_f1_mean": np.mean(fold_metrics["macro_f1"]),
                "macro_f1_std": np.std(fold_metrics["macro_f1"]),
                "accuracy_mean": np.mean(fold_metrics["accuracy"]),
                "accuracy_std": np.std(fold_metrics["accuracy"]),
                "mae_mean": np.mean(fold_metrics["mae"]),
                "mae_std": np.std(fold_metrics["mae"]),
            })

    results_df = pd.DataFrame(results).sort_values("macro_f1_mean", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    df = run_evaluation()
    print(df.to_string(index=False))
