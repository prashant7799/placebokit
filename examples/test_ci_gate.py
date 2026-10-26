"""A CI gate you can copy: fail the build when the model stops beating its placebos.

Run it with:

    pytest examples/test_ci_gate.py

The data is synthetic sensor readings. In a real project, load a small fixed sample of
your own data instead, and keep B and the seed fixed so the test is repeatable.
"""

import numpy as np
import pandas as pd
import pytest

from placebokit import (
    Ablate,
    PermutedWeights,
    ShuffledLabels,
    assert_beats_placebos,
    assert_parity,
    check_causal,
    check_warmup,
    run_controls,
)

SERVING_WINDOW = 48  # rows of history the live server keeps


def make_features(raw):
    """The feature code shared by training and serving."""
    out = pd.DataFrame(index=raw.index)
    out["vs_12h_mean"] = raw["load"] - raw["load"].rolling(12).mean()
    out["change_1h"] = raw["load"].diff()
    out["vibration_6h"] = raw["vibration"].rolling(6).std()
    return out.dropna()


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(0)
    n = 3000
    vibration = rng.normal(size=n)
    load = np.cumsum(rng.normal(size=n)) + 0.5 * vibration
    hours = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"load": load, "vibration": vibration}, index=hours)


@pytest.fixture(scope="module")
def table(raw):
    features = make_features(raw)
    # Target: the load change over the next hour (known only one hour later).
    target = raw["load"].shift(-1) - raw["load"]
    target = target.reindex(features.index).dropna()
    return features.loc[target.index], target


def fit(X_train, y_train):
    design = np.column_stack([np.ones(len(X_train)), X_train.to_numpy()])
    weights, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return lambda X: np.column_stack([np.ones(len(X)), X.to_numpy()]) @ weights


def r_squared(y_true, y_pred):
    """Share of the variation explained. 0 means no better than predicting the mean."""
    return float(1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true))


def test_model_beats_placebos(table):
    X, y = table
    split = int(len(X) * 0.75)
    gap = 12  # leave the longest feature window between training and test rows
    report = run_controls(
        X, y, fit=fit, score=r_squared,
        train_idx=np.arange(split - gap), test_idx=np.arange(split, len(X)),
        controls=[
            ShuffledLabels(block=24),
            Ablate(["change_1h"], group="last change"),
            PermutedWeights(block=24),
        ],
        B=99, seed=0,
    )
    assert_beats_placebos(report, alpha=0.05)


def test_features_do_not_look_ahead(raw):
    assert_parity(check_causal(make_features, raw))


def test_serving_window_is_enough(raw):
    assert_parity(check_warmup(make_features, raw, SERVING_WINDOW))
