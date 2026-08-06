"""
Preprocessing pipeline for the Student Performance Analysis project.

Produces, from the raw dataframe:
  - X_with_course / X_without_course : encoded feature matrices (two experimental conditions,
    per the project decision to model course_id's effect explicitly rather than assume it)
  - y_nominal   : raw GRADE (0-7) for nominal multiclass classification
  - y_ordinal   : same values, exposed separately so downstream code is explicit about which
                  problem framing (nominal vs ordinal) is being used -- never silently shared

Encoding rules (see feature_types.py for the full justification of every choice):
  - Ordinal features: kept as their original integer coding (already a valid 1..k ordinal scale)
  - Nominal features: one-hot encoded (drop='if_binary' to avoid redundant binary columns)
  - course_id: one-hot encoded only when included (it is categorical, not ordinal -- course
    IDs 1-9 are arbitrary labels, not a meaningful scale)
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.features.feature_types import ORDINAL_FEATURES, NOMINAL_FEATURES, COURSE_FEATURE


def build_preprocessor(include_course: bool) -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot encodes nominal features (and course_id, if
    included) while passing ordinal features through unchanged."""
    nominal_cols = list(NOMINAL_FEATURES)
    if include_course:
        nominal_cols = nominal_cols + [COURSE_FEATURE]

    ordinal_cols = list(ORDINAL_FEATURES)

    return ColumnTransformer(
        transformers=[
            ("nominal_ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), nominal_cols),
            ("ordinal_passthrough", "passthrough", ordinal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_columns(include_course: bool) -> list[str]:
    """Raw (pre-encoding) column names used for a given experimental condition."""
    cols = list(NOMINAL_FEATURES) + list(ORDINAL_FEATURES)
    if include_course:
        cols = cols + [COURSE_FEATURE]
    return cols


def prepare_dataset(df: pd.DataFrame, include_course: bool):
    """Return (X_raw, y) for a given experimental condition. X_raw is unencoded -- pass it
    through build_preprocessor() inside a sklearn Pipeline at fit time, per fold, to avoid
    any leakage from fitting the encoder on the full dataset."""
    cols = get_feature_columns(include_course)
    X_raw = df[cols].copy()
    y = df["grade"].copy()
    return X_raw, y


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.load_data import load_raw

    df = load_raw()
    for include_course in (True, False):
        X_raw, y = prepare_dataset(df, include_course=include_course)
        pre = build_preprocessor(include_course=include_course)
        X_enc = pre.fit_transform(X_raw)
        label = "WITH course_id" if include_course else "WITHOUT course_id"
        print(f"{label}: raw shape {X_raw.shape} -> encoded shape {X_enc.shape}")
