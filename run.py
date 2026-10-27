"""Reproduce our best AIcrowd submission.

    python run.py

Steps: load the data with helpers.load_csv_data, fit the preprocessing
pipeline on the full training set, train L2-regularised logistic regression
with Newton's method, choose the decision threshold that maximises F1 on the
training set, and write predictions in {-1, 1} to submission.csv.
The whole procedure is deterministic (no random number is drawn).
"""

import os
import time

import numpy as np

from helpers import create_csv_submission, load_csv_data
from models import best_threshold, weighted_logistic_newton
from preprocessing import Preprocessor, read_column_names

DATA_DIR = "data/dataset"  # folder containing x_train.csv, y_train.csv, x_test.csv
OUTPUT = "submission.csv"
LAMBDA = 1e-4  # selected by 5-fold cross-validation (experiments.py)


def main():
    t0 = time.time()
    print("Loading data (the CSV parsing takes ~1-2 minutes)...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_DIR)
    y01 = (y_train == 1).astype(float)  # {-1, 1} -> {0, 1}
    names = read_column_names(os.path.join(DATA_DIR, "x_train.csv"))

    print("Preprocessing...")
    prep = Preprocessor(names)  # default config = full pipeline of the report
    f_train = prep.fit_transform(x_train)
    f_test = prep.transform(x_test)
    print(f"  {f_train.shape[1]} features "
          f"({len(prep.cat_cols)} categorical, {len(prep.num_cols)} continuous)")  # fmt: skip

    print("Training regularised logistic regression (Newton)...")
    w = weighted_logistic_newton(y01, f_train, LAMBDA, verbose=True)

    scores_train = f_train.dot(w)
    threshold, f1_train = best_threshold(y01, scores_train)
    print(f"  decision threshold on the logit: {threshold:.4f} "
          f"(probability {1 / (1 + np.exp(-threshold)):.3f}), training F1 = {f1_train:.4f}")  # fmt: skip

    pred = np.where(f_test.dot(w) > threshold, 1, -1)
    create_csv_submission(test_ids, pred, OUTPUT)
    print(f"Wrote {OUTPUT}: {np.mean(pred == 1):.1%} predicted positive "
          f"({time.time() - t0:.0f}s total)")  # fmt: skip


if __name__ == "__main__":
    main()
