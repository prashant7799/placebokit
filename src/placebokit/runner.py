"""run_controls and run_prediction_controls: train the real model once, then run each control."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ._data import as_row_positions, n_rows, take_rows
from .controls import Context, Control, Score, predict_with
from .report import ControlReport, ControlResult
from .stats import empirical_p_value


def run_controls(
    X: Any,
    y: Any,
    *,
    fit: Callable[[Any, np.ndarray], Any],
    score: Score,
    train_idx: Any,
    test_idx: Any,
    controls: Sequence[Control],
    B: int = 99,
    seed: int | None = 0,
    higher_is_better: bool = True,
) -> ControlReport:
    """Train the real model, score it, then run every control against it.

    X: features. A 2-D numpy array, a pandas DataFrame, or a list (for example texts).
    y: labels, one per row of X.
    fit: fit(X_train, y_train) trains a model and returns either a predict function
        (X -> predictions) or an object with a .predict method. X_train has the same type
        as X. y_train is a numpy array.
    score: score(y_true, y_pred) -> float. Both arguments are numpy arrays.
    train_idx, test_idx: row positions (or boolean masks) of the training and test rows.
        For time series, give them in time order and keep them apart.
    controls: the controls to run, for example [ShuffledLabels(), SignFlip()].
    B: placebo runs per control. The smallest possible p-value is 1 / (B + 1).
    seed: makes the run repeatable. Each control gets its own random stream from it, so
        adding a control at the end does not change the others.
    higher_is_better: set to False for losses such as mean squared error.
    """
    _check_settings(controls, B)
    total = n_rows(X)
    y_all = np.asarray(y)
    if y_all.shape[0] != total:
        raise ValueError(f"X has {total} rows but y has {y_all.shape[0]}")
    train = as_row_positions(train_idx, total, "train_idx")
    test = as_row_positions(test_idx, total, "test_idx")
    overlap = np.intersect1d(train, test)
    if overlap.size:
        raise ValueError(f"train_idx and test_idx share {overlap.size} rows; keep them apart")
    if test.size < 2:
        raise ValueError("need at least 2 test rows")

    X_train = take_rows(X, train)
    X_test = take_rows(X, test)
    y_train = y_all[train]
    y_test = y_all[test]

    model = fit(X_train, y_train)
    predictions = predict_with(model, X_test)
    real_score = float(score(y_test, predictions))

    ctx = Context(
        y_test=y_test,
        predictions=predictions,
        real_score=real_score,
        score=score,
        fit=fit,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
    )
    return _run_all(ctx, controls, B, seed, higher_is_better)


def run_prediction_controls(
    y_true: Any,
    y_pred: Any,
    *,
    score: Score,
    controls: Sequence[Control],
    B: int = 99,
    seed: int | None = 0,
    higher_is_better: bool = True,
) -> ControlReport:
    """Run the prediction controls when you only have test labels and predictions.

    Nothing is retrained, so ShuffledLabels and Ablate are not allowed here. The other
    arguments work as in run_controls.
    """
    _check_settings(controls, B)
    y_test = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if y_test.shape[0] != predictions.shape[0]:
        raise ValueError(
            f"y_true has {y_test.shape[0]} rows but y_pred has {predictions.shape[0]}"
        )
    if y_test.shape[0] < 2:
        raise ValueError("need at least 2 test rows")
    for control in controls:
        if control.needs_refit:
            raise ValueError(
                f"{control.label} retrains the model; use run_controls with fit= for it"
            )
    real_score = float(score(y_test, predictions))
    ctx = Context(y_test=y_test, predictions=predictions, real_score=real_score, score=score)
    return _run_all(ctx, controls, B, seed, higher_is_better)


def _check_settings(controls: Sequence[Control], B: int) -> None:
    """Check the settings before any model is trained."""
    if isinstance(B, bool) or not isinstance(B, (int, np.integer)) or B < 1:
        raise ValueError(f"B must be a positive integer, got {B!r}")
    if not controls:
        raise ValueError("controls is empty")
    labels = [c.label for c in controls]
    for label in labels:
        if labels.count(label) > 1:
            raise ValueError(f"two controls are both called {label!r}; give one a name=")


def _run_all(
    ctx: Context,
    controls: Sequence[Control],
    B: int,
    seed: int | None,
    higher_is_better: bool,
) -> ControlReport:
    """Run each control with its own random stream and collect the results."""
    if np.isnan(ctx.real_score):
        raise ValueError("the real score is NaN; check the score function and the test rows")

    # One independent random stream per control, all made from the one seed.
    streams = np.random.SeedSequence(seed).spawn(len(controls))

    results = []
    for control, stream in zip(controls, streams, strict=True):
        rng = np.random.default_rng(stream)
        scores, details = control.placebo_scores(ctx, int(B), rng)
        scores = np.asarray(scores, dtype=float)
        p_value = empirical_p_value(ctx.real_score, scores, higher_is_better)
        results.append(
            ControlResult(
                name=control.label,
                real_score=ctx.real_score,
                scores=scores,
                p_value=p_value,
                higher_is_better=higher_is_better,
                details=details,
            )
        )
    return ControlReport(
        real_score=ctx.real_score,
        results=results,
        B=int(B),
        seed=seed,
        higher_is_better=higher_is_better,
    )
