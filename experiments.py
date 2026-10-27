"""Cross-validation experiments reported in the paper.

Runs (5-fold stratified CV, preprocessing re-fitted inside every fold):
  1. Baselines built on the six required methods of implementations.py.
  2. Regularisation sweep of the final model (Newton logistic regression).
  3. Ablation study: each preprocessing step switched off one at a time.

Results are appended to results/results.json (already finished experiments are
skipped, so the script can be interrupted and resumed).  Usage:
    python experiments.py            # everything (~1 hour on a laptop)
    python experiments.py --figures  # only redraw the figures from the json
"""

import json
import os
import sys
import time

import numpy as np

from helpers import load_csv_data
from implementations import least_squares, reg_logistic_regression, ridge_regression
from models import (
    best_threshold,
    class_balanced_weights,
    cross_validate,
    f1_score,
    roc_auc,
    stratified_k_fold,
    weighted_logistic_newton,
)
from preprocessing import Preprocessor, read_column_names

DATA_DIR = "data/dataset"
CACHE = "cache/data.npz"
RESULTS = "results/results.json"
K_FOLDS = 5
SEED = 0
BEST_LAMBDA = 1e-4  # updated after the sweep (see results/results.json)

RAW_CONFIG = {  # "no feature engineering" baseline
    "drop_admin": False,
    "max_nan": 1.0,
    "one_hot": False,
    "decode_special": False,
    "log_transform": False,
    "clip": False,
    "missing_indicator": False,
}

ABLATIONS = {
    "full": {},
    "no_one_hot": {"one_hot": False},
    "no_special_codes": {"decode_special": False},
    "no_log_clip": {"log_transform": False, "clip": False},
    "no_missing_ind": {"missing_indicator": False},
    "keep_admin_cols": {"drop_admin": False},
    "plus_squares": {"squares": True},
    "raw_features": RAW_CONFIG,
}


def load_data():
    """Load the training set (cached as .npz after the first slow CSV read)."""
    if not os.path.exists(CACHE):
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        xtr, xte, ytr, trid, teid = load_csv_data(DATA_DIR)
        np.savez(CACHE, xtr=xtr, xte=xte, ytr=ytr, trid=trid, teid=teid)
    d = np.load(CACHE)
    return d["xtr"], (d["ytr"] == 1).astype(float)


def feature_maker(names, config):
    """Return make_features(x_tr, x_va) for cross_validate."""

    def make(x_tr, x_va):
        prep = Preprocessor(names, config)
        return prep.fit_transform(x_tr), prep.transform(x_va)

    return make


def newton_scorer(lambda_, positive_weight=1.0):
    """Return fit_scores for the Newton logistic model."""

    def fit(y_tr, f_tr, f_va):
        s = class_balanced_weights(y_tr, positive_weight)
        w = weighted_logistic_newton(y_tr, f_tr, lambda_, s)
        return f_tr.dot(w), f_va.dot(w)

    return fit


def closed_form_scorer(method, lambda_=None):
    """Least squares / ridge on {0,1} targets, scores = x^T w."""

    def fit(y_tr, f_tr, f_va):
        f_tr64 = f_tr.astype(np.float64)
        if lambda_ is None:
            w, _ = method(y_tr, f_tr64)
        else:
            w, _ = method(y_tr, f_tr64, lambda_)
        return f_tr64.dot(w), f_va.astype(np.float64).dot(w)

    return fit


def gd_logistic_scorer(lambda_, max_iters):
    """reg_logistic_regression from implementations.py with gamma = 1/L."""

    def fit(y_tr, f_tr, f_va):
        x = f_tr.astype(np.float64)
        # Lipschitz constant of the logistic gradient: lambda_max(X^T X)/(4N)
        v = np.ones(x.shape[1])
        for _ in range(30):  # power iteration
            v = x.T.dot(x.dot(v))
            v /= np.linalg.norm(v)
        lip = v.dot(x.T.dot(x.dot(v))) / (4 * len(y_tr)) + 2 * lambda_
        w, _ = reg_logistic_regression(
            y_tr, x, lambda_, np.zeros(x.shape[1]), max_iters, 1.0 / lip
        )
        return x.dot(w), f_va.astype(np.float64).dot(w)

    return fit


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_result(results, key, res, seconds):
    results[key] = {
        "f1": float(res["f1"]),
        "f1_std": float(res["f1_std"]),
        "acc": float(res["acc"]),
        "auc": float(res["auc"]),
        "seconds": round(seconds, 1),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    r = results[key]
    print(f"{key:28s} F1={r['f1']:.4f}±{r['f1_std']:.4f} acc={r['acc']:.4f} "
          f"AUC={r['auc']:.4f} ({seconds:.0f}s)", flush=True)  # fmt: skip


def run(key, results, x, y, make, fit):
    if key in results:
        print(f"{key:28s} (cached)")
        return
    t = time.time()
    try:
        res = cross_validate(x, y, make, fit, k=K_FOLDS, seed=SEED)
    except np.linalg.LinAlgError as err:  # e.g. X^T X singular
        results[key] = {"error": str(err)}
        with open(RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{key:28s} FAILED: {err}", flush=True)
        return
    save_result(results, key, res, time.time() - t)


def lambda_sweep(results, x, y, names, lambdas):
    """All lambdas are fit on the same folds/features (features built once)."""
    pending = [lam for lam in lambdas if f"lambda={lam:g}" not in results]
    if not pending:
        return
    t = time.time()
    folds = stratified_k_fold(y, K_FOLDS, SEED)
    scores = {lam: {"f1": [], "acc": [], "auc": []} for lam in pending}
    for va in folds:
        tr = np.setdiff1d(np.arange(len(y)), va)
        f_tr, f_va = feature_maker(names, {})(x[tr], x[va])
        for lam in pending:
            w = weighted_logistic_newton(y[tr], f_tr, lam)
            thr, _ = best_threshold(y[tr], f_tr.dot(w))
            s_va = f_va.dot(w)
            pred = (s_va > thr).astype(int)
            scores[lam]["f1"].append(f1_score(y[va], pred))
            scores[lam]["acc"].append(np.mean(pred == y[va]))
            scores[lam]["auc"].append(roc_auc(y[va], s_va))
        del f_tr, f_va
    for lam in pending:
        s = scores[lam]
        res = {"f1": np.mean(s["f1"]), "f1_std": np.std(s["f1"]),
               "acc": np.mean(s["acc"]), "auc": np.mean(s["auc"])}  # fmt: skip
        save_result(results, f"lambda={lam:g}", res, (time.time() - t) / len(pending))


def threshold_study(results, x, y, names):
    """F1 with the naive 0.5 probability threshold vs. the tuned threshold."""
    for pw_name, pw in [("unweighted", 1.0), ("balanced", None)]:
        key = f"thr0.5_{pw_name}"
        if key in results:
            continue
        t = time.time()
        folds = stratified_k_fold(y, K_FOLDS, SEED)
        f1s, accs, aucs = [], [], []
        for va in folds:
            tr = np.setdiff1d(np.arange(len(y)), va)
            f_tr, f_va = feature_maker(names, {})(x[tr], x[va])
            w = weighted_logistic_newton(
                y[tr], f_tr, BEST_LAMBDA, class_balanced_weights(y[tr], pw)
            )
            s_va = f_va.dot(w)
            pred = (s_va > 0).astype(int)  # sigmoid(s) > 0.5
            f1s.append(f1_score(y[va], pred))
            accs.append(np.mean(pred == y[va]))
            aucs.append(roc_auc(y[va], s_va))
            del f_tr, f_va
        res = {"f1": np.mean(f1s), "f1_std": np.std(f1s),
               "acc": np.mean(accs), "auc": np.mean(aucs)}  # fmt: skip
        save_result(results, key, res, time.time() - t)


def make_figures(results):
    """Ablation bar chart + lambda curve used in the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("figures", exist_ok=True)
    lam_keys = sorted(
        [k for k in results if k.startswith("lambda=")],
        key=lambda k: float(k.split("=")[1]),
    )
    labels = {
        "full": "Full pipeline",
        "no_one_hot": "− one-hot encoding",
        "no_special_codes": "− special-code decoding",
        "no_log_clip": "− log + clipping",
        "no_missing_ind": "− missing indicators",
        "keep_admin_cols": "+ admin/weight columns",
        "plus_squares": "+ squared features",
        "raw_features": "Raw features only",
        "thr0.5_unweighted": "− threshold tuning",
        "thr0.5_balanced": "− thr. tuning, + class weights",
    }
    keys = [k for k in labels if k in results]
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )
    ax = axes[0]
    f1 = [results[k]["f1"] for k in keys]
    err = [results[k]["f1_std"] for k in keys]
    colors = ["#2a6f97" if k == "full" else "#9bb7c9" for k in keys]
    ax.barh(range(len(keys)), f1, xerr=err, color=colors, height=0.7)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([labels[k] for k in keys], fontsize=7)
    ax.invert_yaxis()
    lo = min(f1) - 0.02
    ax.set_xlim(max(0, lo), max(f1) + 0.01)
    ax.set_xlabel("5-fold CV F1", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("(a) Ablation", fontsize=8)

    ax = axes[1]
    lams = [float(k.split("=")[1]) for k in lam_keys]
    ax.errorbar(lams, [results[k]["f1"] for k in lam_keys],
                yerr=[results[k]["f1_std"] for k in lam_keys],
                marker="o", ms=3, color="#2a6f97", capsize=2)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$", fontsize=8)
    ax.set_ylabel("5-fold CV F1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(r"(b) L2 regularisation", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/ablation.pdf")
    fig.savefig("figures/ablation.png", dpi=150)
    print("figures written to figures/")


def main():
    results = load_results()
    if "--figures" in sys.argv:
        make_figures(results)
        return
    x, y = load_data()
    names = read_column_names(os.path.join(DATA_DIR, "x_train.csv"))
    full = feature_maker(names, {})

    print("== Regularisation sweep (Newton logistic regression)")
    lambda_sweep(results, x, y, names, [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])

    print("== Baselines from implementations.py (full preprocessing)")
    run("least_squares", results, x, y, full, closed_form_scorer(least_squares))
    run("ridge_1e-4", results, x, y, full,
        closed_form_scorer(ridge_regression, 1e-4))  # fmt: skip
    run("reg_logistic_gd_300it", results, x, y, full,
        gd_logistic_scorer(BEST_LAMBDA, 300))  # fmt: skip
    run("ridge_raw_features", results, x, y, feature_maker(names, RAW_CONFIG),
        closed_form_scorer(ridge_regression, 1e-4))  # fmt: skip

    print("== Threshold / class-weight study")
    threshold_study(results, x, y, names)
    run("balanced_weights_tuned_thr", results, x, y, full,
        newton_scorer(BEST_LAMBDA, None))  # fmt: skip

    print("== Ablation (Newton logistic regression)")
    for name, cfg in ABLATIONS.items():
        if name == "full" and f"lambda={BEST_LAMBDA:g}" in results:
            results["full"] = results[f"lambda={BEST_LAMBDA:g}"]
            continue
        run(name, results, x, y, feature_maker(names, cfg), newton_scorer(BEST_LAMBDA))
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    make_figures(results)


if __name__ == "__main__":
    main()
