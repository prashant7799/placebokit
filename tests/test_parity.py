"""Train/serve parity checks, with planted look-ahead and warm-up bugs."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from placebokit import (
    check_causal,
    check_parity,
    check_repeatable,
    check_warmup,
    perturb_numbers,
)
from placebokit.parity import default_cuts


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    index = pd.date_range("2026-01-01", periods=600, freq="h")
    return pd.DataFrame(
        {
            "temp": 20 + np.cumsum(rng.normal(scale=0.1, size=600)),
            "count": rng.integers(0, 50, size=600),
            "door_open": rng.random(600) < 0.2,
            "site": ["north"] * 600,
        },
        index=index,
    )


def rolling_mean(df):
    return df["temp"].rolling(10).mean().dropna().rename("mean_10")


# --- check_parity -------------------------------------------------------------------


def test_identical_tables_match():
    a = pd.DataFrame({"x": [1.0, 2.0, np.nan], "s": ["a", "b", None]})
    report = check_parity(a, a.copy())
    assert report.ok
    assert report.n_compared == 6
    assert "passed" in str(report)


def test_tiny_float_noise_is_within_tolerance_and_a_real_change_is_not():
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert check_parity(a, a + 1e-13).ok
    changed = a.copy()
    changed.loc[1, "x"] = 2.5
    report = check_parity(a, changed)
    assert not report.ok
    assert report.n_mismatched == 1
    assert report.max_abs_diff == pytest.approx(0.5)
    assert "row 1, column 'x': 2 vs 2.5" in report.first_mismatch


def test_missing_values_only_match_missing_values():
    a = pd.DataFrame({"x": [1.0, np.nan]})
    b = pd.DataFrame({"x": [1.0, 0.0]})
    assert not check_parity(a, b).ok
    text_a = pd.DataFrame({"s": ["a", None]})
    text_b = pd.DataFrame({"s": ["a", "b"]})
    assert not check_parity(text_a, text_b).ok
    assert check_parity(text_a, text_a.copy()).ok


def test_first_mismatch_is_the_earliest_row():
    a = pd.DataFrame({"x": [0.0, 0.0, 0.0], "y": [0.0, 0.0, 0.0]})
    b = a.copy()
    b.loc[2, "x"] = 1.0
    b.loc[1, "y"] = 1.0
    assert check_parity(a, b).first_mismatch.startswith("row 1, column 'y'")


def test_rows_and_columns_in_one_table_only_fail():
    a = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0]}, index=[0, 1])
    b = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    report = check_parity(a, b)
    assert not report.ok
    assert report.n_mismatched == 0
    text = str(report)
    assert "columns only in the first table: ['y']" in text
    assert "1 rows only in the second table, starting at 2" in text


def test_series_and_arrays_are_accepted():
    s = pd.Series([1.0, 2.0], name="f")
    assert check_parity(s, s.copy()).ok
    assert check_parity(np.array([1.0, 2.0]), np.array([1.0, 2.0])).ok
    assert check_parity(np.ones((2, 2)), np.ones((2, 2))).ok
    with pytest.raises(TypeError):
        check_parity([1, 2], [1, 2])


def test_repeated_labels_raise():
    a = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 0])
    with pytest.raises(ValueError, match="repeated row labels"):
        check_parity(a, a)


@given(arrays(np.float64, (12, 3), elements=st.one_of(st.floats(-1e6, 1e6), st.just(np.nan))))
def test_any_table_matches_itself(values):
    table = pd.DataFrame(values, columns=["a", "b", "c"])
    assert check_parity(table, table.copy()).ok


# --- check_repeatable ---------------------------------------------------------------


def test_deterministic_features_are_repeatable(raw):
    assert check_repeatable(rolling_mean, raw).ok


def test_features_with_unseeded_noise_are_not_repeatable(raw):
    def noisy(df):
        return df["temp"] + np.random.default_rng().normal(size=len(df))

    assert not check_repeatable(noisy, raw).ok


def test_a_feature_function_that_changes_its_input_fails(raw):
    def adds_column(df):
        df["mean_10"] = df["temp"].rolling(10).mean()
        return df[["mean_10"]]

    report = check_repeatable(adds_column, raw)
    assert not report.ok
    assert "changed its input" in str(report)
    assert "mean_10" not in raw.columns  # the caller's data was protected


# --- check_causal -------------------------------------------------------------------


def test_trailing_window_is_causal(raw):
    report = check_causal(rolling_mean, raw)
    assert report.ok
    assert report.n_compared > 0


@pytest.mark.parametrize(
    "leaky",
    [
        lambda df: df["temp"].rolling(10, center=True).mean(),  # centred window
        lambda df: df["temp"].shift(-1),  # next row
        lambda df: df["temp"] - df["temp"].mean(),  # whole-sample mean
        lambda df: (df["count"] > df["count"].median()).astype(float),  # whole-sample median
        lambda df: df["temp"].rolling(10).mean().bfill(),  # warm-up filled from the future
        lambda df: df["temp"].diff().bfill(),  # first row filled from row 1
    ],
    ids=["centred", "next-row", "full-mean", "full-median", "bfill-warmup", "bfill-lag"],
)
def test_planted_look_ahead_is_caught(raw, leaky):
    report = check_causal(leaky, raw)
    assert not report.ok
    assert report.first_mismatch is not None


def test_whole_sample_minimum_is_caught_even_when_the_minimum_is_in_the_past(raw):
    # Moving the future only upwards could never change a minimum that lies in the past.
    # The default perturbation moves each row up or down, so this leak shows at a late cut.
    def distance_from_minimum(df):
        return df["temp"] - df["temp"].min()

    late_cut = len(raw) - 100
    assert int(np.argmin(raw["temp"].to_numpy())) < late_cut  # the minimum is in the past
    assert not check_causal(distance_from_minimum, raw, cuts=[late_cut]).ok

    def count_above_minimum(df):
        return (df["count"] - df["count"].min()).astype(float)

    assert not check_causal(count_above_minimum, raw, cuts=[late_cut]).ok


def test_relabelled_feature_rows_raise_instead_of_passing(raw):
    # After reset_index the feature rows no longer carry the data's labels, so nothing
    # could be matched. Without this check the leak below would "pass".
    def centred_then_reset(df):
        return df["temp"].rolling(10, center=True).mean().reset_index(drop=True)

    with pytest.raises(ValueError, match="labels are not in its input"):
        check_causal(centred_then_reset, raw)
    with pytest.raises(TypeError, match="DataFrame or Series"):
        check_causal(lambda df: df["temp"].to_numpy(), raw)
    with pytest.raises(ValueError, match="labels are not in its input"):
        check_warmup(centred_then_reset, raw, 50)


def test_relabelled_feature_rows_raise_with_the_default_row_labels():
    # With the default labels 0, 1, 2, ... the labels after reset_index are input labels
    # too, but they point at the wrong rows, so this causal feature used to "fail".
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": np.cumsum(rng.normal(size=1000))})

    def trailing_mean_then_reset(df):
        return df["x"].rolling(24).mean().dropna().reset_index(drop=True)

    with pytest.raises(ValueError, match="labels are not in its input"):
        check_causal(trailing_mean_then_reset, data)
    with pytest.raises(ValueError, match="labels are not in its input"):
        check_warmup(trailing_mean_then_reset, data, 50)

    # The same feature with its labels kept passes both checks.
    def trailing_mean(df):
        return df["x"].rolling(24).mean().dropna()

    assert check_causal(trailing_mean, data).ok
    assert check_warmup(trailing_mean, data, 24).ok


def test_a_check_that_compares_nothing_fails(raw):
    # A 500-row warm-up leaves no feature rows at or before any of these cuts.
    def long_warmup(df):
        return df["temp"].rolling(500).mean().dropna()

    report = check_causal(long_warmup, raw, cuts=[10, 100])
    assert report.n_compared == 0
    assert not report.ok
    assert "no past feature value was compared" in str(report)


def test_a_leak_across_a_gap_of_missing_values_needs_a_cut_inside_the_gap(raw):
    # Filling a gap from later rows looks ahead only at the rows of the gap. The default
    # cuts (..., 149, 299, 449) miss a gap at rows 300 to 305; a cut inside it finds it.
    gappy = raw.copy()
    gappy.iloc[300:306, gappy.columns.get_loc("temp")] = np.nan

    def interpolated(df):
        return df["temp"].interpolate()

    def back_filled(df):
        return df["temp"].bfill()

    for leaky in (interpolated, back_filled):
        assert check_causal(leaky, gappy).ok
        assert not check_causal(leaky, gappy, cuts=[302]).ok


def test_early_cuts_are_what_catch_backfilled_warmup(raw):
    def backfilled(df):
        return df["temp"].rolling(30).mean().bfill()

    assert check_causal(backfilled, raw, cuts=[300]).ok  # a late cut misses it
    assert not check_causal(backfilled, raw).ok  # the default early cuts find it


def test_default_cuts_double_then_spread():
    assert default_cuts(1000) == [0, 1, 2, 4, 8, 16, 32, 64, 128, 249, 499, 749]
    assert default_cuts(4) == [0, 1, 2]


def test_every_number_moves_under_the_default_perturbation(raw):
    future = raw.iloc[100:].copy()
    moved = perturb_numbers(future, np.random.default_rng(0))
    assert (moved["temp"] != future["temp"]).all()
    assert (moved["count"] != future["count"]).all()
    assert (moved["door_open"] != future["door_open"]).all()
    assert (moved["site"] == future["site"]).all()  # text is left alone
    assert moved["count"].dtype == future["count"].dtype
    # Each number moves by at least one step, up or down, and both directions occur.
    temp_step = (moved["temp"] - future["temp"]) / future["temp"].std(ddof=0)
    assert (temp_step.abs() >= 1 - 1e-9).all() and (temp_step.abs() <= 2 + 1e-9).all()
    assert (temp_step > 0).any() and (temp_step < 0).any()
    count_step = moved["count"] - future["count"]
    assert count_step.abs().between(1, 9).all()
    assert (count_step > 0).any() and (count_step < 0).any()


def test_unsigned_integers_only_move_up():
    rows = pd.DataFrame({"n": np.arange(50, dtype=np.uint16)})
    moved = perturb_numbers(rows, np.random.default_rng(0))
    assert ((moved["n"].astype(int) - rows["n"].astype(int)) >= 1).all()


def test_custom_perturbation_and_series_data():
    series = pd.Series(np.arange(50, dtype=float), name="x")

    def next_value(s):
        return s.shift(-1)

    assert not check_causal(next_value, series).ok
    report = check_causal(next_value, series, perturb=lambda rows, rng: rows * 0)
    assert not report.ok
    with pytest.raises(ValueError, match="same rows"):
        check_causal(next_value, series, perturb=lambda rows, rng: rows.iloc[1:])


def test_causal_input_errors(raw):
    text_only = pd.DataFrame({"site": ["a"] * 10})
    with pytest.raises(ValueError, match="no numeric columns"):
        check_causal(lambda df: df, text_only)
    with pytest.raises(ValueError, match="between"):
        check_causal(rolling_mean, raw, cuts=[599])
    with pytest.raises(ValueError, match="at least 4"):
        check_causal(rolling_mean, raw.iloc[:3])
    with pytest.raises(TypeError):
        check_causal(rolling_mean, raw.to_numpy())


# --- check_warmup -------------------------------------------------------------------


def test_rolling_window_needs_exactly_its_length(raw):
    report = check_warmup(rolling_mean, raw, range(1, 60))
    assert report.ok
    assert report.min_window == 10
    text = str(report)
    assert "smallest window that matches: 10 rows" in text
    # The counts describe the window that matched, not the smaller ones tried before it.
    assert report.n_compared == 5
    assert report.n_mismatched == 0
    assert report.first_mismatch is None
    assert "tried 10 window sizes, from 1 to 10 rows" in text


def test_one_window_can_be_asserted(raw):
    assert check_warmup(rolling_mean, raw, 48).ok
    assert not check_warmup(rolling_mean, raw, 5).ok


def test_expanding_features_never_match_a_window(raw):
    def expanding_mean(df):
        return df["temp"].expanding().mean()

    report = check_warmup(expanding_mean, raw, [50, 100, 200])
    assert not report.ok
    assert report.min_window is None
    assert "no window up to 200 rows" in str(report)
    assert report.n_compared == 5 and report.n_mismatched == 5  # the 200-row window
    assert report.first_mismatch.startswith("window 200:")


def test_exponential_smoothing_needs_a_long_history_for_tight_tolerance(raw):
    def smooth(df):
        return df["temp"].ewm(span=10).mean()

    tight = check_warmup(smooth, raw, range(10, 300, 10))
    loose = check_warmup(smooth, raw, range(10, 300, 10), rtol=1e-4)
    assert tight.ok and loose.ok
    assert loose.min_window < tight.min_window


def test_features_missing_at_the_last_rows_cannot_be_served(raw):
    def centred(df):
        return df["temp"].rolling(10, center=True).mean().dropna()

    report = check_warmup(centred, raw, 50)
    assert not report.ok
    assert "could not produce them" in str(report)


def test_warmup_input_errors(raw):
    with pytest.raises(ValueError, match="does not fit"):
        check_warmup(rolling_mean, raw, 400, rows=[100])
    with pytest.raises(ValueError, match="positive"):
        check_warmup(rolling_mean, raw, [0])
    with pytest.raises(ValueError, match="between"):
        check_warmup(rolling_mean, raw, 5, rows=[600])
