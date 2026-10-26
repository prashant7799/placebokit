"""PermutedWeights, RandomWeights, SignFlip, RandomEntries and LagShift."""

import numpy as np
import pytest

from placebokit import (
    Ablate,
    LagShift,
    PermutedWeights,
    RandomEntries,
    RandomWeights,
    ShuffledLabels,
    SignFlip,
    run_controls,
    run_prediction_controls,
)

from simdata import correlation, linear_problem, lstsq_fit, r_squared


class ScoreSpy:
    """A score function that remembers every set of predictions it was given."""

    def __init__(self, score):
        self.score = score
        self.seen = []

    def __call__(self, y_true, y_pred):
        self.seen.append(np.array(y_pred))
        return self.score(y_true, y_pred)


def signal_and_predictions(n=500, strength=1.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    predictions = strength * truth + rng.normal(size=n)
    return truth, predictions


def run(y, pred, controls, B=19, seed=0, score=correlation, **kw):
    return run_prediction_controls(y, pred, score=score, controls=controls, B=B, seed=seed, **kw)


# --- PermutedWeights ----------------------------------------------------------------


def test_permuted_weights_keeps_the_exact_values():
    y, pred = signal_and_predictions()
    spy = ScoreSpy(correlation)
    run(y, pred, [PermutedWeights()], B=5, score=spy)
    assert np.array_equal(spy.seen[0], pred)  # the real run comes first
    for placebo in spy.seen[1:]:
        assert np.array_equal(np.sort(placebo), np.sort(pred))
        assert not np.array_equal(placebo, pred)


def test_real_skill_beats_permuted_weights_and_no_skill_does_not():
    y, pred = signal_and_predictions(strength=0.5)
    assert run(y, pred, [PermutedWeights()], B=99)["PermutedWeights"].p_value == pytest.approx(0.01)
    y, pred = signal_and_predictions(strength=0.0, seed=1)
    assert run(y, pred, [PermutedWeights()], B=99)["PermutedWeights"].p_value > 0.05


def test_permuted_weights_handles_two_column_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    proba = np.column_stack([1 - y * 0.8, y * 0.8]) + 0.1

    def accuracy(y_true, p):
        return float(np.mean(np.argmax(p, axis=1) == y_true))

    report = run(y, proba, [PermutedWeights(), LagShift()], B=19, score=accuracy)
    assert report.passed()


# --- RandomWeights ------------------------------------------------------------------


def test_random_weights_match_mean_and_spread_exactly():
    y, pred = signal_and_predictions()
    pred = pred + 3.0
    spy = ScoreSpy(correlation)
    report = run(y, pred, [RandomWeights()], B=5, score=spy)
    for placebo in spy.seen[1:]:
        assert placebo.mean() == pytest.approx(pred.mean())
        assert placebo.std() == pytest.approx(pred.std())
    assert report["RandomWeights"].details["mean"] == pytest.approx(pred.mean())


def test_nonnegative_random_weights_stay_nonnegative_and_match_the_mean():
    rng = np.random.default_rng(0)
    budget = rng.gamma(2.0, 1.0, size=400)
    gain = rng.normal(size=400)
    spy = ScoreSpy(lambda g, w: float(np.sum(g * w)))
    run(gain, budget, [RandomWeights(nonnegative=True)], B=20, score=spy)
    spreads = []
    for placebo in spy.seen[1:]:
        assert placebo.min() >= 0
        assert placebo.mean() == pytest.approx(budget.mean())
        spreads.append(placebo.std())
    assert np.mean(spreads) == pytest.approx(budget.std(), rel=0.15)


def test_random_weights_edge_cases():
    y = np.arange(10.0)
    constant = np.full(10, 2.0)
    spy = ScoreSpy(lambda a, b: float(np.mean(b)))
    run(y, constant, [RandomWeights()], B=3, score=spy)
    assert all(np.array_equal(p, constant) for p in spy.seen)
    with pytest.raises(ValueError, match="below 0"):
        run(y, y - 5, [RandomWeights(nonnegative=True)], score=lambda a, b: 0.0)
    with pytest.raises(ValueError, match="1-D"):
        run(y, np.ones((10, 2)), [RandomWeights()], score=lambda a, b: 0.0)


# --- SignFlip -----------------------------------------------------------------------


def test_sign_flip_keeps_sizes_and_changes_directions():
    y, pred = signal_and_predictions()
    spy = ScoreSpy(correlation)
    report = run(y, pred, [SignFlip()], B=5, score=spy)
    for placebo in spy.seen[1:-1]:  # the last call is the full flip
        assert np.allclose(np.abs(placebo), np.abs(pred))
        assert np.any(np.sign(placebo) != np.sign(pred))
    assert np.allclose(spy.seen[-1], -pred)
    assert report["SignFlip"].details["full_flip_score"] == pytest.approx(-report.real_score)


def test_sign_flip_around_a_center_for_probabilities():
    rng = np.random.default_rng(0)
    proba = rng.uniform(0, 1, size=100)
    spy = ScoreSpy(lambda a, b: float(np.mean(b)))
    run(np.zeros(100), proba, [SignFlip(center=0.5)], B=3, score=spy)
    for placebo in spy.seen[1:-1]:
        flipped = ~np.isclose(placebo, proba)
        assert np.allclose(placebo[flipped], 1 - proba[flipped])
    assert np.allclose(spy.seen[-1], 1 - proba)


def test_block_sign_flip_flips_whole_blocks():
    y, pred = signal_and_predictions(n=100)
    spy = ScoreSpy(correlation)
    run(y, pred, [SignFlip(block=10)], B=3, score=spy)
    for placebo in spy.seen[1:-1]:
        ratio = placebo / pred
        for start in range(0, 100, 10):
            assert len(set(np.round(ratio[start : start + 10]).tolist())) == 1


def test_sign_flip_tells_direction_skill_from_magnitude_skill():
    # Outcomes are big in noisy rows, in a random direction. A model that only knows WHEN
    # rows are noisy does not beat SignFlip; a model that knows the direction does.
    rng = np.random.default_rng(0)
    n = 4000
    noisy = rng.random(n) < 0.5
    move = np.where(noisy, 2.0, 0.3) * rng.normal(size=n)
    outcome = np.where(move > 1, 1, np.where(move < -1, -1, 0))
    when_only = np.where(noisy, rng.choice([-1, 1], size=n), 0)
    with_direction = np.where(noisy, np.sign(move + rng.normal(scale=1.0, size=n)), 0)

    def hit_rate(y_true, calls):
        raised = calls != 0
        return float(np.mean(y_true[raised] == calls[raised]))

    a = run(outcome, when_only, [SignFlip(), RandomEntries()], B=99, score=hit_rate)
    b = run(outcome, with_direction, [SignFlip(), RandomEntries()], B=99, score=hit_rate)
    assert a["SignFlip"].p_value > 0.05
    assert a["RandomEntries"].p_value == pytest.approx(0.01)  # but the timing is real
    assert b["SignFlip"].p_value == pytest.approx(0.01)


# --- RandomEntries ------------------------------------------------------------------


def test_random_entries_keep_the_events_and_move_them():
    rng = np.random.default_rng(0)
    events = np.zeros(300, dtype=int)
    events[rng.choice(300, size=40, replace=False)] = rng.choice([-1, 1], size=40)
    spy = ScoreSpy(lambda a, b: float(np.sum(b)))
    report = run(np.zeros(300), events, [RandomEntries()], B=5, score=spy)
    for placebo in spy.seen[1:]:
        assert np.count_nonzero(placebo) == 40
        assert np.array_equal(np.sort(placebo[placebo != 0]), np.sort(events[events != 0]))
        assert not np.array_equal(placebo, events)
    assert report["RandomEntries"].details["events"] == 40


def test_random_entries_respect_the_eligible_mask():
    events = np.zeros(100)
    events[[1, 3, 5]] = 1.0
    eligible = np.zeros(100, dtype=bool)
    eligible[:50] = True
    spy = ScoreSpy(lambda a, b: 0.0)
    run(np.zeros(100), events, [RandomEntries(eligible=eligible)], B=10, score=spy)
    for placebo in spy.seen[1:]:
        assert np.all(placebo[50:] == 0)


def test_random_entries_refuse_real_events_on_ineligible_rows():
    # The placebos can only use eligible rows. If the real events sit elsewhere, the
    # comparison is unfair, so it must stop instead of returning a p-value.
    events = np.zeros(100)
    events[[60, 70, 80]] = 1.0
    eligible = np.arange(100) < 50
    with pytest.raises(ValueError, match="eligible marks False"):
        run(np.zeros(100), events, [RandomEntries(eligible=eligible)], score=lambda a, b: 0.0)


def test_random_entries_errors():
    zeros = np.zeros(10)
    with pytest.raises(ValueError, match="no events"):
        run(zeros, zeros, [RandomEntries()], score=lambda a, b: 0.0)
    events = np.ones(10)
    with pytest.raises(ValueError, match="eligible rows"):
        run(zeros, events, [RandomEntries(eligible=np.arange(10) < 5)], score=lambda a, b: 0.0)
    with pytest.raises(ValueError, match="length"):
        run(zeros, events, [RandomEntries(eligible=[True])], score=lambda a, b: 0.0)


# --- LagShift -----------------------------------------------------------------------


def test_explicit_lags_roll_the_predictions():
    y, pred = signal_and_predictions(n=50)
    spy = ScoreSpy(correlation)
    report = run(y, pred, [LagShift(lags=[1, -2, 7])], B=99, score=spy)
    result = report["LagShift(lags=1,-2,7)"]
    assert result.runs == 3  # B is ignored when lags are given
    assert result.details["lags"] == [1, -2, 7]
    assert np.array_equal(spy.seen[1], np.roll(pred, 1))
    assert np.array_equal(spy.seen[2], np.roll(pred, -2))


def test_random_lags_are_different_and_inside_the_allowed_range():
    y, pred = signal_and_predictions(n=200)
    report = run(y, pred, [LagShift(min_lag=30)], B=99)
    lags = np.array(report["LagShift(min_lag=30)"].details["lags"])
    assert len(lags) == 99
    assert len(set(lags.tolist())) == 99  # no shift is used twice
    assert lags.min() >= 30
    assert lags.max() <= 170


def test_lag_shift_uses_every_allowed_shift_when_b_is_large():
    y, pred = signal_and_predictions(n=50)
    result = run(y, pred, [LagShift(min_lag=10)], B=99)["LagShift(min_lag=10)"]
    assert result.details["lags"] == list(range(10, 41))  # all 31 shifts, once each
    assert result.runs == 31


def test_lag_shift_with_every_shift_is_an_exact_rank_test():
    # Take the 12 circular shifts of one set of predictions. Whichever of them is the
    # "real" one, LagShift compares it with the other 11, so the 12 p-values must be
    # 1/12, 2/12, ..., 12/12, each exactly once.
    rng = np.random.default_rng(0)
    y = rng.normal(size=12)
    base = rng.normal(size=12)
    p_values = []
    for shift in range(12):
        report = run(y, np.roll(base, shift), [LagShift()], B=99)
        p_values.append(report["LagShift"].p_value)
    assert np.allclose(sorted(p_values), np.arange(1, 13) / 12)


class LagShiftWithRepeats(LagShift):
    """LagShift as it would be if it drew its shifts with repeats. Only for comparison."""

    def placebo_scores(self, ctx, B, rng):
        n = len(ctx.predictions)
        lags = rng.choice(np.arange(self.min_lag, n - self.min_lag + 1), size=B)
        scores = [ctx.score_predictions(np.roll(ctx.predictions, int(lag))) for lag in lags]
        return np.array(scores), {"lags": lags.tolist()}


def short_lag_shift_null_rate(trials=3000, control=LagShift):
    """Unrelated predictions on 30-row test sets: how often is LagShift beaten?"""
    rng = np.random.default_rng(1)
    hits = 0
    for trial in range(trials):
        y = rng.normal(size=30)
        pred = rng.normal(size=30)
        report = run(y, pred, [control()], B=19, seed=trial)
        hits += report["LagShift"].beaten(0.05)
    return hits / trials


def test_lag_shift_is_calibrated_on_a_short_test_set():
    # Without repeats it is exact. Drawing shifts with repeats makes the p-value too
    # small on short test sets: about 6-7% false "beaten" calls at alpha = 0.05 on the
    # same trials.
    trials = 3000
    rate = short_lag_shift_null_rate(trials)
    assert abs(rate - 0.05) <= 3 * np.sqrt(0.05 * 0.95 / trials)
    with_repeats = short_lag_shift_null_rate(trials, LagShiftWithRepeats)
    assert 0.06 <= with_repeats <= 0.07


def test_lag_shift_errors_and_labels():
    y, pred = signal_and_predictions(n=20)
    with pytest.raises(ValueError, match="too large"):
        run(y, pred, [LagShift(min_lag=11)])
    with pytest.raises(ValueError, match="would not move"):
        run(y, pred, [LagShift(lags=[20])])
    with pytest.raises(ValueError):
        LagShift(lags=[])
    with pytest.raises(ValueError):
        LagShift(min_lag=0)
    assert LagShift().label == "LagShift"
    assert LagShift(lags=range(1, 11)).label == "LagShift(10 lags)"


# --- run_prediction_controls --------------------------------------------------------


def test_refit_controls_are_refused_without_fit():
    y, pred = signal_and_predictions()
    for control in [ShuffledLabels(), Ablate([0])]:
        with pytest.raises(ValueError, match="retrains"):
            run(y, pred, [control])


def test_prediction_controls_inside_run_controls_use_the_real_predictions():
    X, y, train, test = linear_problem()
    spy = ScoreSpy(r_squared)
    run_controls(X, y, fit=lstsq_fit, score=spy, train_idx=train, test_idx=test,
                 controls=[PermutedWeights()], B=3)
    real = lstsq_fit(X[train], y[train])(X[test])
    assert np.allclose(spy.seen[0], real)
    for placebo in spy.seen[1:]:
        assert np.allclose(np.sort(placebo), np.sort(real))


@pytest.mark.parametrize(
    "control",
    [PermutedWeights(), RandomWeights(), SignFlip(), LagShift(min_lag=5), PermutedWeights(block=7)],
    ids=lambda c: c.label,
)
def test_p_values_do_not_change_under_an_increasing_change_of_score(control):
    # The same seed gives the same placebos. If the score is passed through an increasing
    # function, every comparison keeps its direction, so every p-value stays the same.
    y, pred = signal_and_predictions(n=120, strength=0.2, seed=4)

    def squashed_score(y_true, y_pred):
        return float(np.exp(3 * correlation(y_true, y_pred)))

    plain = run(y, pred, [control], B=49, seed=9)
    squashed = run(y, pred, [control], B=49, seed=9, score=squashed_score)
    assert squashed[control.label].p_value == plain[control.label].p_value


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="rows"):
        run(np.zeros(5), np.zeros(4), [PermutedWeights()])


def test_a_custom_control_works_as_documented():
    from placebokit import Control

    class ConstantGuess(Control):
        """Predict the training mean everywhere (one run)."""

        def placebo_scores(self, ctx, B, rng):
            guess = np.full(len(ctx.y_test), ctx.y_train.mean())
            return np.array([ctx.score_predictions(guess)]), {"guess": float(guess[0])}

    class RetrainOnNoise(Control):
        needs_refit = True

        def placebo_scores(self, ctx, B, rng):
            scores = [
                ctx.fit_and_score(ctx.X_train, rng.normal(size=len(ctx.y_train)), ctx.X_test)
                for _ in range(B)
            ]
            return np.array(scores), {}

    X, y, train, test = linear_problem()
    report = run_controls(X, y, fit=lstsq_fit, score=r_squared, train_idx=train, test_idx=test,
                          controls=[ConstantGuess(), RetrainOnNoise()], B=19)
    assert report.names == ["ConstantGuess", "RetrainOnNoise"]
    assert report["ConstantGuess"].runs == 1
    assert report["RetrainOnNoise"].p_value == pytest.approx(0.05)
    with pytest.raises(ValueError, match="retrains"):
        run(y[test], y[test], [RetrainOnNoise()], score=r_squared)
