"""Null calibration: with no real skill, a control should call the model "skilled" at most
about alpha of the time. The trials are seeded, so the rates below are fixed numbers.

Each rate is computed by a small function, so the numbers quoted in the README can be
recomputed with the same code the tests run.
"""

import numpy as np
import pytest

from placebokit import (
    LagShift,
    PermutedWeights,
    RandomEntries,
    RandomWeights,
    ShuffledLabels,
    SignFlip,
    run_controls,
    run_prediction_controls,
)

from simdata import ar1, correlation, lstsq_fit, r_squared

ALPHA = 0.05
B = 19  # the smallest p-value is 1/20 = 0.05


def within_three_standard_errors(rate, trials, alpha=ALPHA):
    """For an exact test the rate should be close to alpha: not above it, and not far below."""
    return abs(rate - alpha) <= 3 * np.sqrt(alpha * (1 - alpha) / trials)


def shuffled_labels_null_rate(trials=1000):
    """Labels that ignore the features: how often is ShuffledLabels beaten?"""
    rng = np.random.default_rng(0)
    hits = 0
    for trial in range(trials):
        X = rng.normal(size=(120, 3))
        y = rng.normal(size=120)  # no link to X
        report = run_controls(
            X, y, fit=lstsq_fit, score=r_squared,
            train_idx=np.arange(80), test_idx=np.arange(80, 120),
            controls=[ShuffledLabels()], B=B, seed=trial,
        )
        hits += report["ShuffledLabels"].beaten(ALPHA)
    return hits / trials


def prediction_control_null_rate(control, trials=2000):
    """Predictions unrelated to the outcomes: how often is the control beaten?"""
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(trials):
        y = rng.normal(size=200)
        pred = rng.normal(size=200)  # symmetric around 0 and unrelated to y
        report = run_prediction_controls(
            y, pred, score=correlation, controls=[control], B=B, seed=trial
        )
        hits += report[control.label].beaten(ALPHA)
    return hits / trials


def random_entries_null_rate(trials=2000):
    """Events on random rows: how often is RandomEntries beaten?"""
    rng = np.random.default_rng(2)
    hits = 0

    def mean_outcome_at_events(y_true, events):
        return float(y_true[events != 0].mean())

    for trial in range(trials):
        y = rng.normal(size=300)
        events = (rng.random(300) < 0.1).astype(int)
        report = run_prediction_controls(
            y, events, score=mean_outcome_at_events, controls=[RandomEntries()],
            B=B, seed=trial,
        )
        hits += report["RandomEntries"].beaten(ALPHA)
    return hits / trials


SERIAL_CONTROLS = [PermutedWeights(), PermutedWeights(block=60), LagShift(min_lag=60)]


def serial_null_rates(trials=500):
    """Two unrelated slow-moving series: how often is each control beaten?"""
    rng = np.random.default_rng(3)
    hits = {c.label: 0 for c in SERIAL_CONTROLS}
    for trial in range(trials):
        y = ar1(600, 0.95, rng)
        pred = ar1(600, 0.95, rng)
        report = run_prediction_controls(
            y, pred, score=correlation, controls=SERIAL_CONTROLS, B=B, seed=trial
        )
        for result in report:
            hits[result.name] += result.beaten(ALPHA)
    return {label: count / trials for label, count in hits.items()}


def test_shuffled_labels_is_calibrated_when_labels_ignore_features():
    assert within_three_standard_errors(shuffled_labels_null_rate(1000), 1000)


@pytest.mark.parametrize(
    "control",
    [PermutedWeights(), RandomWeights(), SignFlip(), LagShift()],
    ids=lambda c: c.label,
)
def test_prediction_controls_are_calibrated_for_unrelated_iid_predictions(control):
    # Normal predictions, symmetric around 0: every one of these controls is exact here.
    assert within_three_standard_errors(prediction_control_null_rate(control, 2000), 2000)


def test_random_entries_is_calibrated_for_events_on_random_rows():
    assert within_three_standard_errors(random_entries_null_rate(2000), 2000)


def test_slow_drifts_fool_a_plain_shuffle_but_not_blocks_or_lag_shifts():
    # Two unrelated AR(1) series line up by chance. A plain shuffle of single rows
    # destroys the drift and is far too easy to beat. Block shuffles and lag shifts keep it.
    rates = serial_null_rates(500)
    assert rates["PermutedWeights"] > 0.2  # badly anti-conservative
    # Blocks and lag shifts are approximations here, so allow some room above 5%.
    assert rates["PermutedWeights(block=60)"] <= 0.1
    assert rates["LagShift(min_lag=60)"] <= 0.1
