"""Synthetic data and tiny models shared by the tests. Everything is seeded."""

from __future__ import annotations

import numpy as np


def linear_problem(n=400, n_features=4, signal=1.0, seed=0):
    """y = signal * (first feature) + noise. Returns X, y, train rows, test rows."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = signal * X[:, 0] + rng.normal(size=n)
    cut = int(n * 0.7)
    return X, y, np.arange(cut), np.arange(cut, n)


def lstsq_fit(X_train, y_train):
    """Least squares with an intercept. Works for arrays and DataFrames."""
    X_train = np.asarray(X_train, dtype=float)
    design = np.column_stack([np.ones(len(X_train)), X_train])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=float)
        return np.column_stack([np.ones(len(X_new)), X_new]) @ weights

    return predict


def r_squared(y_true, y_pred):
    return float(1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def correlation(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def ar1(n, phi, rng):
    """An AR(1) series: x[t] = phi * x[t-1] + noise."""
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x
