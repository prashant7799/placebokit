"""assert_beats_placebos, assert_parity and the optional pytest plugin."""

import numpy as np
import pandas as pd
import pytest

from placebokit import (
    LagShift,
    PermutedWeights,
    SignFlip,
    assert_beats_placebos,
    assert_parity,
    check_causal,
    run_prediction_controls,
)
from placebokit.report import ControlReport, ControlResult


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def make_report(strength, B=49, controls=None):
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    pred = strength * y + rng.normal(size=300)
    return run_prediction_controls(
        y, pred, score=correlation,
        controls=controls or [PermutedWeights(), SignFlip()], B=B, seed=0,
    )


def test_a_skilled_model_passes():
    assert_beats_placebos(make_report(0.5))


def test_a_skill_free_model_fails_with_a_readable_message():
    with pytest.raises(AssertionError) as info:
        assert_beats_placebos(make_report(0.0))
    message = str(info.value)
    assert "did not beat" in message
    assert "PermutedWeights: p-value" in message


def test_only_the_listed_controls_are_checked():
    report = make_report(0.0)
    report.results[0].p_value = 0.01  # pretend one control was beaten
    assert_beats_placebos(report, controls=["PermutedWeights"])
    with pytest.raises(AssertionError):
        assert_beats_placebos(report, controls=["SignFlip"])
    with pytest.raises(KeyError):
        assert_beats_placebos(report, controls=["Nope"])
    with pytest.raises(ValueError, match="no controls"):
        assert_beats_placebos(report, controls=[])  # would otherwise pass without testing


def test_one_control_name_is_not_read_as_a_list_of_letters():
    report = make_report(0.0)
    report.results[0].p_value = 0.01  # pretend one control was beaten
    assert_beats_placebos(report, controls="PermutedWeights")
    with pytest.raises(AssertionError, match="SignFlip: p-value"):
        assert_beats_placebos(report, controls="SignFlip")


def test_too_few_runs_is_a_broken_test_not_a_failed_model():
    report = make_report(0.5, B=9)  # smallest p-value is 0.1
    with pytest.raises(ValueError, match="at least 19 runs"):
        assert_beats_placebos(report, alpha=0.05)
    assert_beats_placebos(report, alpha=0.1)
    fixed_lags = make_report(0.5, controls=[LagShift(lags=[5, 10])])
    with pytest.raises(ValueError, match="can never be"):
        assert_beats_placebos(fixed_lags)


def test_the_error_shows_the_exact_smallest_p_value():
    # 1/99 = 0.0101 is above alpha = 0.01, but at 3 decimals it would print as 0.010.
    with pytest.raises(ValueError) as info:
        assert_beats_placebos(make_report(0.5, B=98), alpha=0.01)
    assert "at least 1/99 = 0.0101 and can never be <= 0.01" in str(info.value)
    assert "at least 99 runs" in str(info.value)


def test_a_small_p_value_in_the_failure_message_does_not_print_as_zero():
    # 3/20000 = 0.00015 misses alpha = 0.0001, and at 3 decimals it would print as 0.000.
    result = ControlResult(name="x", real_score=1.0, scores=np.zeros(19999), p_value=3 / 20000)
    report = ControlReport(real_score=1.0, results=[result], B=19999, seed=0)
    with pytest.raises(AssertionError, match=r"x: p-value 0\.00015,"):
        assert_beats_placebos(report, alpha=0.0001)


def test_alpha_must_be_a_probability():
    with pytest.raises(ValueError):
        assert_beats_placebos(make_report(0.5), alpha=0)


def test_assert_parity():
    data = pd.DataFrame({"x": np.arange(20.0)})
    assert_parity(check_causal(lambda df: df["x"].rolling(3).mean(), data))
    with pytest.raises(AssertionError, match="causal check: FAILED"):
        assert_parity(check_causal(lambda df: df["x"].shift(-1), data))


PLUGIN_TEST = """
def test_values(placebo_b, placebo_seed):
    print("B", placebo_b, "seed", placebo_seed)
"""


def test_plugin_defaults(pytester):
    pytester.makepyfile(PLUGIN_TEST)
    result = pytester.runpytest("-p", "placebokit.pytest_plugin", "-s")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*B 99 seed 0*"])


def test_plugin_command_line_options(pytester):
    pytester.makepyfile(PLUGIN_TEST)
    result = pytester.runpytest(
        "-p", "placebokit.pytest_plugin", "-s", "--placebo-b=999", "--placebo-seed=7"
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*B 999 seed 7*"])


def test_plugin_rejects_a_bad_b(pytester):
    pytester.makepyfile(PLUGIN_TEST)
    result = pytester.runpytest("-p", "placebokit.pytest_plugin", "--placebo-b=0")
    assert result.ret != 0
