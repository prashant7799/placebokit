"""ShuffledLabels and Ablate: the controls that retrain the model."""

import numpy as np
import pandas as pd
import pytest

from placebokit import Ablate, ShuffledLabels, SignFlip, run_controls, run_prediction_controls

from simdata import linear_problem, lstsq_fit, mse, r_squared


class Spy:
    """Wraps fit and score and records what they were given."""

    def __init__(self):
        self.fit_calls = []
        self.score_calls = []

    def fit(self, X_train, y_train):
        self.fit_calls.append((X_train.copy(), np.array(y_train)))
        return lstsq_fit(X_train, y_train)

    def score(self, y_true, y_pred):
        self.score_calls.append((np.array(y_true), np.array(y_pred)))
        return r_squared(y_true, y_pred)


def run(X, y, train, test, controls, B=19, seed=0, fit=lstsq_fit, score=r_squared, **kw):
    return run_controls(
        X, y, fit=fit, score=score, train_idx=train, test_idx=test,
        controls=controls, B=B, seed=seed, **kw,
    )


# --- ShuffledLabels -----------------------------------------------------------------


def test_only_training_labels_are_shuffled():
    X, y, train, test = linear_problem()
    spy = Spy()
    run(X, y, train, test, [ShuffledLabels()], B=5, fit=spy.fit, score=spy.score)
    real_X, real_y = spy.fit_calls[0]
    assert len(spy.fit_calls) == 6  # the real fit plus 5 placebo fits
    for X_seen, y_seen in spy.fit_calls[1:]:
        assert np.array_equal(X_seen, real_X)  # features untouched
        assert np.array_equal(np.sort(y_seen), np.sort(real_y))  # same labels...
        assert not np.array_equal(y_seen, real_y)  # ...in a new order
    for y_true, _ in spy.score_calls:
        assert np.array_equal(y_true, y[test])  # always scored on the real test labels


def test_block_shuffle_keeps_runs_of_training_labels():
    X, y, train, test = linear_problem()
    spy = Spy()
    run(X, y, train, test, [ShuffledLabels(block=20)], B=3, fit=spy.fit, score=spy.score)
    real_y = spy.fit_calls[0][1]
    for _, y_seen in spy.fit_calls[1:]:
        # The first 20 shuffled labels are one whole block of the real labels.
        start = int(np.flatnonzero(real_y == y_seen[0])[0])
        assert np.array_equal(y_seen[:20], real_y[start : start + 20])


def test_shuffle_follows_the_order_of_train_idx():
    # train_idx in reverse order: blocks are cut along that order.
    X, y, train, test = linear_problem()
    spy = Spy()
    run(X, y, train[::-1], test, [ShuffledLabels(block=10)], B=2, fit=spy.fit, score=spy.score)
    real_y = spy.fit_calls[0][1]
    assert np.array_equal(real_y, y[train[::-1]])


def test_planted_signal_beats_shuffled_labels():
    X, y, train, test = linear_problem(signal=1.0)
    report = run(X, y, train, test, [ShuffledLabels()], B=99)
    assert report["ShuffledLabels"].p_value == pytest.approx(0.01)
    assert report["ShuffledLabels"].mean < 0.05 < report.real_score


def test_no_signal_does_not_beat_shuffled_labels():
    X, y, train, test = linear_problem(signal=0.0, seed=1)
    report = run(X, y, train, test, [ShuffledLabels()], B=99)
    assert report["ShuffledLabels"].p_value > 0.05


def test_lower_is_better_scores():
    X, y, train, test = linear_problem(signal=1.0)
    report = run(X, y, train, test, [ShuffledLabels()], B=19, score=mse, higher_is_better=False)
    assert report["ShuffledLabels"].p_value == pytest.approx(0.05)
    assert report["ShuffledLabels"].best > report.real_score  # the best placebo is the lowest


# --- Ablate -------------------------------------------------------------------------


def test_ablating_the_signal_column_hurts_and_a_noise_column_does_not():
    X, y, train, test = linear_problem(signal=1.0)
    report = run(
        X, y, train, test,
        [Ablate([0], group="signal"), Ablate([2, 3], group="noise")],
        B=49,
    )
    assert report["Ablate(signal)"].p_value == pytest.approx(0.02)
    assert report["Ablate(noise)"].p_value > 0.05
    assert report["Ablate(noise)"].mean == pytest.approx(report.real_score, abs=0.03)


def test_shuffle_moves_the_group_together_and_leaves_other_columns():
    X, y, train, test = linear_problem(n_features=4)
    spy = Spy()
    run(X, y, train, test, [Ablate([1, 2])], B=3, fit=spy.fit, score=spy.score)
    real_X = spy.fit_calls[0][0]
    for X_seen, _ in spy.fit_calls[1:]:
        assert np.array_equal(X_seen[:, [0, 3]], real_X[:, [0, 3]])
        # Columns 1 and 2 moved with the same row order, so their pairs survive.
        real_pairs = {tuple(row) for row in real_X[:, [1, 2]]}
        seen_pairs = {tuple(row) for row in X_seen[:, [1, 2]]}
        assert real_pairs == seen_pairs
        assert not np.array_equal(X_seen[:, 1], real_X[:, 1])


def test_drop_mode_retrains_once_without_the_columns():
    X, y, train, test = linear_problem(n_features=4)
    spy = Spy()
    report = run(X, y, train, test, [Ablate([1, 2], how="drop")], B=50,
                 fit=spy.fit, score=spy.score)
    result = report["Ablate(1,2, drop)"]
    assert result.runs == 1
    assert result.p_value in (0.5, 1.0)
    assert spy.fit_calls[1][0].shape[1] == 2


def test_keep_only_takes_away_every_other_column():
    X, y, train, test = linear_problem(n_features=4)
    spy = Spy()
    run(X, y, train, test, [Ablate([0], keep_only=True, how="drop")], B=1,
        fit=spy.fit, score=spy.score)
    assert np.array_equal(spy.fit_calls[1][0], X[train][:, [0]])

    report = run(X, y, train, test, [Ablate([0], group="signal", keep_only=True)], B=19)
    # Keeping only the signal column loses almost nothing.
    result = report["Ablate(all but signal)"]
    assert result.mean == pytest.approx(report.real_score, abs=0.05)


def test_ablate_works_with_dataframe_column_names():
    X, y, train, test = linear_problem(n_features=3)
    frame = pd.DataFrame(X, columns=["vibration", "temp", "humidity"])
    spy = Spy()
    report = run(frame, y, train, test, [Ablate(["vibration"]), Ablate("humidity", how="drop")],
                 B=9, fit=spy.fit, score=spy.score)
    assert report.names == ["Ablate(vibration)", "Ablate(humidity, drop)"]
    assert list(spy.fit_calls[-1][0].columns) == ["vibration", "temp"]
    assert report["Ablate(vibration)"].p_value == pytest.approx(0.1)


def test_ablate_takes_one_column_as_a_python_or_numpy_integer():
    X, y, train, test = linear_problem(n_features=3)
    for column in (0, np.int64(0)):
        ablate = Ablate(column, group="signal")
        assert ablate.columns == [0]
        report = run(X, y, train, test, [ablate], B=9)
        assert report["Ablate(signal)"].p_value == pytest.approx(0.1)


def test_ablate_keeps_the_column_types_of_a_dataframe():
    # A model that needs a categorical column must still get one after the shuffle.
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "load": rng.normal(size=120),
            "site": pd.Categorical(rng.choice(["north", "south"], size=120)),
            "count": rng.integers(0, 9, size=120),
        }
    )
    seen = []

    def fit(X_train, y_train):
        seen.append(X_train.dtypes.to_dict())
        return lambda X_new: np.full(len(X_new), float(np.mean(y_train)))

    run(frame, rng.normal(size=120), np.arange(80), np.arange(80, 120),
        [Ablate(["site", "count"])], B=3, fit=fit, score=r_squared)
    for dtypes in seen:
        assert dtypes == frame.dtypes.to_dict()


def test_ablate_rejects_repeated_column_names():
    X, y, train, test = linear_problem(n_features=3)
    frame = pd.DataFrame(X, columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="repeated column names"):
        run(frame, y, train, test, [Ablate(["a"])])


def test_ablate_rejects_unknown_columns_and_bad_options():
    X, y, train, test = linear_problem(n_features=3)
    with pytest.raises(KeyError):
        run(X, y, train, test, [Ablate([5])])
    with pytest.raises(KeyError):
        run(X, y, train, test, [Ablate(["temp"])])
    with pytest.raises(KeyError):
        run(pd.DataFrame(X), y, train, test, [Ablate(["temp"])])
    with pytest.raises(ValueError):
        Ablate([0], how="zero")
    with pytest.raises(ValueError):
        Ablate([0], how="drop", block=5)
    with pytest.raises(ValueError):
        Ablate([])
    with pytest.raises(ValueError, match="every column"):
        run(X, y, train, test, [Ablate([0, 1, 2], how="drop")])


# --- run_controls plumbing ----------------------------------------------------------


def test_same_seed_same_scores_and_new_seed_new_scores():
    X, y, train, test = linear_problem()
    controls = [ShuffledLabels(), Ablate([0])]
    a = run(X, y, train, test, controls, seed=5)
    b = run(X, y, train, test, controls, seed=5)
    c = run(X, y, train, test, controls, seed=6)
    for name in a.names:
        assert np.array_equal(a[name].scores, b[name].scores)
        assert not np.array_equal(a[name].scores, c[name].scores)


def test_adding_a_control_at_the_end_does_not_change_the_others():
    X, y, train, test = linear_problem()
    one = run(X, y, train, test, [ShuffledLabels()], seed=2)
    two = run(X, y, train, test, [ShuffledLabels(), SignFlip()], seed=2)
    assert np.array_equal(one["ShuffledLabels"].scores, two["ShuffledLabels"].scores)


def test_fit_may_return_an_object_with_predict():
    X, y, train, test = linear_problem()

    class Model:
        def __init__(self, X_train, y_train):
            self.predict_fn = lstsq_fit(X_train, y_train)

        def predict(self, X_new):
            return self.predict_fn(X_new)

    a = run(X, y, train, test, [ShuffledLabels()], fit=Model)
    b = run(X, y, train, test, [ShuffledLabels()])
    assert a.real_score == pytest.approx(b.real_score)


def test_fit_must_return_something_that_predicts():
    X, y, train, test = linear_problem()
    with pytest.raises(TypeError, match="predict"):
        run(X, y, train, test, [ShuffledLabels()], fit=lambda X_train, y_train: 42)


def test_list_features_are_supported():
    # For example texts: the model only sees a list.
    rng = np.random.default_rng(0)
    words = ["good", "bad"]
    texts = [words[i] for i in rng.integers(0, 2, size=200)]
    y = np.array([1.0 if t == "good" else 0.0 for t in texts]) + rng.normal(scale=0.3, size=200)

    def fit(X_train, y_train):
        mean_good = np.mean([v for t, v in zip(X_train, y_train, strict=True) if t == "good"])
        mean_bad = np.mean([v for t, v in zip(X_train, y_train, strict=True) if t == "bad"])
        return lambda X_new: np.array([mean_good if t == "good" else mean_bad for t in X_new])

    report = run(texts, y, np.arange(150), np.arange(150, 200), [ShuffledLabels()], B=19, fit=fit)
    assert report["ShuffledLabels"].p_value == pytest.approx(0.05)


def test_train_and_test_rows_must_not_overlap():
    X, y, train, test = linear_problem()
    with pytest.raises(ValueError, match="share"):
        run(X, y, train, np.concatenate([test, train[:3]]), [ShuffledLabels()])


def test_boolean_masks_work_as_row_selections():
    X, y, train, test = linear_problem()
    mask_train = np.zeros(len(y), dtype=bool)
    mask_train[train] = True
    a = run(X, y, mask_train, ~mask_train, [ShuffledLabels()])
    b = run(X, y, train, test, [ShuffledLabels()])
    assert np.array_equal(a["ShuffledLabels"].scores, b["ShuffledLabels"].scores)


@pytest.mark.parametrize(
    "train, test, message",
    [
        ([], [1, 2], "empty"),
        ([0, 0, 1], [5, 6], "repeats"),
        ([0, 1], [999], "outside"),
        (np.ones(3, dtype=bool), [5, 6], "length"),
        (np.linspace(0, 9.5, 10), [20, 21], "integer"),  # 9.5 must not become 9
    ],
)
def test_bad_row_selections_raise(train, test, message):
    X, y, _, _ = linear_problem()
    with pytest.raises(ValueError, match=message):
        run(X, y, train, test, [ShuffledLabels()])


def test_mismatched_lengths_and_bad_settings_raise():
    X, y, train, test = linear_problem()
    with pytest.raises(ValueError, match="rows"):
        run(X, y[:-1], train, test, [ShuffledLabels()])
    with pytest.raises(ValueError, match="B must"):
        run(X, y, train, test, [ShuffledLabels()], B=0)
    with pytest.raises(ValueError, match="empty"):
        run(X, y, train, test, [])
    with pytest.raises(ValueError, match="both called"):
        run(X, y, train, test, [ShuffledLabels(), ShuffledLabels()])
    with pytest.raises(ValueError, match="2 test rows"):
        run(X, y, train, test[:1], [ShuffledLabels()])
    with pytest.raises(ValueError, match="2 test rows"):
        run_prediction_controls([1.0], [1.0], score=lambda a, b: 0.0, controls=[SignFlip()])
    named = run(X, y, train, test, [ShuffledLabels(), ShuffledLabels(name="again")])
    assert named.names == ["ShuffledLabels", "again"]


def test_nan_real_score_raises():
    X, y, train, test = linear_problem()
    with pytest.raises(ValueError, match="NaN"):
        run(X, y, train, test, [ShuffledLabels()], score=lambda a, b: float("nan"))


def test_nan_placebo_scores_count_against_the_model():
    X, y, train, test = linear_problem()
    calls = []

    def score(y_true, y_pred):
        calls.append(1)
        return r_squared(y_true, y_pred) if len(calls) == 1 else float("nan")

    report = run(X, y, train, test, [ShuffledLabels()], B=9, score=score)
    assert report["ShuffledLabels"].p_value == 1.0
    assert report["ShuffledLabels"].n_failed == 9


def test_bad_settings_fail_before_any_training():
    X, y, train, test = linear_problem()
    spy = Spy()
    for kwargs in [{"B": 0}, {"controls": []}, {"controls": [SignFlip(), SignFlip()]}]:
        settings = {"controls": [ShuffledLabels()], "B": 5, **kwargs}
        with pytest.raises(ValueError):
            run(X, y, train, test, fit=spy.fit, score=spy.score, **settings)
    assert spy.fit_calls == []
