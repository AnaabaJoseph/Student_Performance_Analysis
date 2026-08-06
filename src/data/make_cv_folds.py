"""
Generate a fixed stratified k-fold cross-validation assignment and save it to disk.

This is run ONCE. Every modeling stage (3, 4, 5) loads data/processed/cv_folds.csv
rather than calling StratifiedKFold independently -- this guarantees every model
(baseline, ensemble, with-course, without-course) is evaluated on the exact same
splits, which is required for the model comparisons in this project to be valid.

random_state is fixed and documented for full reproducibility.
"""
import sys

sys.path.insert(0, ".")

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.load_data import load_raw

N_FOLDS = 5
RANDOM_STATE = 42
OUTPUT_PATH = "data/processed/cv_folds.csv"


def main():
    df = load_raw()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_assignment = pd.Series(index=df.index, dtype=int, name="fold")
    for fold_num, (_, val_idx) in enumerate(skf.split(df, df["grade"])):
        fold_assignment.iloc[val_idx] = fold_num

    cv_df = pd.DataFrame({
        "student_id": df["student_id"],
        "grade": df["grade"],
        "fold": fold_assignment,
    })
    cv_df.to_csv(OUTPUT_PATH, index=False)

    print(f"{N_FOLDS}-fold stratified CV assignment saved to {OUTPUT_PATH} "
          f"(random_state={RANDOM_STATE}).")
    print("\nClass balance per fold:")
    print(pd.crosstab(cv_df["fold"], cv_df["grade"]))
    print("\nFold sizes:")
    print(cv_df["fold"].value_counts().sort_index())


if __name__ == "__main__":
    main()
