"""Training, evaluation and cross-validation utilities (NumPy only)."""

import numpy as np

from implementations import sigmoid

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def weighted_logistic_newton(
    y, tx, lambda_, sample_weight=None, max_iters=15, tol=1e-4, verbose=False
):
    """L2-regularised, sample-weighted logistic regression via Newton's method.

    Minimises  sum_i s_i * [log(1 + exp(x_i^T w)) - y_i x_i^T w] / sum_i s_i
               + lambda_ * ||w_{1:}||^2
    (the bias, column 0 of tx, is not penalised).  Each Newton step solves
    (H + 2 lambda_ I) dw = g, with H = X^T diag(s * p * (1 - p)) X / sum(s).
    Newton converges in ~10-20 steps where plain gradient descent needs
    thousands, which makes 5-fold cross-validation affordable.

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features with a bias column first, shape (N, D)
        lambda_: L2 regularisation strength
        sample_weight: per-sample weights s_i, shape (N,) (default: all ones)
        max_iters: maximum number of Newton steps
        tol: stop when the max absolute update is below tol
        verbose: print the loss at every step

    Returns:
        w: fitted weights, shape (D,)
    """
    n, d = tx.shape
    s = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
    s = s / s.sum()
    reg = 2 * lambda_ * np.ones(d)
    reg[0] = 0.0  # do not penalise the bias
    w = np.zeros(d)
    for it in range(max_iters):
        z = tx.dot(w.astype(tx.dtype)).astype(np.float64)
        p = sigmoid(z)
        grad = tx.T.dot((s * (p - y)).astype(tx.dtype)).astype(np.float64)
        grad += reg * w
        h_diag = (s * p * (1 - p)).astype(tx.dtype)
        hess = (tx.T * h_diag).dot(tx).astype(np.float64)
        hess[np.diag_indices(d)] += reg + 1e-10
        step = np.linalg.solve(hess, grad)
        w -= step
        if verbose:
            loss = np.sum(s * (np.logaddexp(0, z) - y * z))
            print(
                f"  newton it {it:2d}  loss {loss:.5f}  |step| {np.abs(step).max():.2e}"
            )
        if np.abs(step).max() < tol:
            break
    return w


def class_balanced_weights(y, positive_weight=None):
    """Per-sample weights giving each class the same total weight.

    Args:
        y: labels in {0, 1}
        positive_weight: weight of a positive sample relative to a negative
            one; None -> N_neg / N_pos (perfect balance), 1 -> no weighting.
    """
    if positive_weight is None:
        positive_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    return np.where(y == 1, positive_weight, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def f1_score(y_true, y_pred):
    """F1 score of the positive class; labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return np.mean(y_true == y_pred)


def best_threshold(y_true, scores):
    """Threshold on scores maximising F1, found by sorting (exact search).

    Returns:
        (threshold, best_f1)
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)  # number predicted positive
    n_pos = y_sorted.sum()
    f1 = 2 * tp / (k + n_pos)
    i = int(np.argmax(f1))
    s_sorted = scores[order]
    thr = (
        s_sorted[i] if i + 1 == len(s_sorted) else 0.5 * (s_sorted[i] + s_sorted[i + 1])
    )
    return thr, f1[i]


def roc_auc(y_true, scores):
    """Area under the ROC curve (Mann-Whitney statistic, ties averaged)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    # average ranks over ties
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i : j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    ranks[order] = r
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def stratified_k_fold(y, k, seed=0):
    """Return k arrays of validation indices with the same class ratio."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        for f, part in enumerate(np.array_split(idx, k)):
            folds[f].append(part)
    return [np.sort(np.concatenate(f)) for f in folds]


def cross_validate(x_raw, y01, make_features, fit_scores, k=5, seed=0):
    """K-fold CV returning out-of-fold scores and per-fold metrics.

    The preprocessing is re-fitted on each training fold (no leakage), the
    F1-optimal threshold is chosen on the training fold's own predictions
    and then applied to the held-out fold.

    Args:
        x_raw: raw feature matrix (N, D_raw)
        y01: labels in {0, 1}
        make_features: function(x_tr_raw, x_va_raw) -> (f_tr, f_va)
        fit_scores: function(y_tr, f_tr, f_va) -> (scores_tr, scores_va)
        k: number of folds
        seed: random seed of the fold split

    Returns:
        dict with mean/std of F1, accuracy and AUC over folds, and the
        out-of-fold scores.
    """
    folds = stratified_k_fold(y01, k, seed)
    oof = np.zeros(len(y01))
    f1s, accs, aucs = [], [], []
    for va in folds:
        tr = np.setdiff1d(np.arange(len(y01)), va)
        f_tr, f_va = make_features(x_raw[tr], x_raw[va])
        s_tr, s_va = fit_scores(y01[tr], f_tr, f_va)
        thr, _ = best_threshold(y01[tr], s_tr)
        pred = (s_va > thr).astype(int)
        f1s.append(f1_score(y01[va], pred))
        accs.append(accuracy(y01[va], pred))
        aucs.append(roc_auc(y01[va], s_va))
        oof[va] = s_va
        del f_tr, f_va
    return {
        "f1": np.mean(f1s),
        "f1_std": np.std(f1s),
        "acc": np.mean(accs),
        "auc": np.mean(aucs),
        "oof": oof,
    }
