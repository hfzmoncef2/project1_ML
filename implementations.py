"""Implementations of the six basic ML methods required for Project 1.

All functions return ``(w, loss)`` where ``w`` is the last weight vector and
``loss`` is the corresponding (unregularised) cost.  Vectors are 1D arrays.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Loss and gradient helpers
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Mean squared error with the 1/2 factor: L(w) = 1/(2N) * ||y - Xw||^2.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        scalar loss (0-d numpy value)
    """
    e = y - tx.dot(w)
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss: -1/N * X^T (y - Xw).

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        gradient, shape (D,)
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(y)


def sigmoid(t):
    """Numerically stable logistic function 1 / (1 + exp(-t)).

    Args:
        t: array of any shape

    Returns:
        array of the same shape with values in (0, 1)
    """
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    pos = t >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-t[pos]))
    exp_t = np.exp(t[~pos])
    out[~pos] = exp_t / (1.0 + exp_t)
    return out


def compute_logistic_loss(y, tx, w):
    """Average negative log-likelihood for labels y in {0, 1}.

    Uses the identity log(1 + exp(z)) - y*z, computed with np.logaddexp
    for numerical stability.

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        scalar loss (0-d numpy value)
    """
    z = tx.dot(w)
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the average logistic loss: 1/N * X^T (sigmoid(Xw) - y).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        w: weights, shape (D,)

    Returns:
        gradient, shape (D,)
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / len(y)


# ---------------------------------------------------------------------------
# Required methods (Table 1)
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full gradient descent on the MSE loss.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weight vector, shape (D,)
        loss: MSE loss of the last weight vector
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using SGD (mini-batch size 1) on the MSE loss.

    At each iteration one sample is drawn uniformly at random and a gradient
    step is taken on that sample only.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of stochastic steps
        gamma: step size

    Returns:
        w: last weight vector, shape (D,)
        loss: MSE loss of the last weight vector on the full dataset
    """
    w = np.array(initial_w, dtype=float)
    n = len(y)
    for _ in range(max_iters):
        i = np.random.randint(n)
        w = w - gamma * compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression via the normal equations X^T X w = X^T y.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)

    Returns:
        w: optimal weights, shape (D,)
        loss: MSE loss of w
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations.

    Minimises 1/(2N)||y - Xw||^2 + lambda_ ||w||^2, whose solution is
    (X^T X + 2N lambda_ I) w = X^T y.

    Args:
        y: labels, shape (N,)
        tx: features, shape (N, D)
        lambda_: regularisation strength

    Returns:
        w: optimal weights, shape (D,)
        loss: MSE loss of w (without the penalty term)
    """
    n, d = tx.shape
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T.dot(y))
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weight vector, shape (D,)
        loss: logistic loss of the last weight vector
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularised logistic regression using gradient descent.

    Minimises logistic_loss(w) + lambda_ ||w||^2 (gradient 2 * lambda_ * w).

    Args:
        y: labels in {0, 1}, shape (N,)
        tx: features, shape (N, D)
        lambda_: regularisation strength
        initial_w: initial weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weight vector, shape (D,)
        loss: logistic loss of the last weight vector (without penalty)
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
