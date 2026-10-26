"""The negative controls.

A control builds placebo versions of your result: versions that cannot know the answer.
Each control returns a list of placebo scores. run_controls compares them with the real
score.

There are two kinds:

- Refit controls change the training data and train the model again
  (ShuffledLabels, Ablate). They need your fit function.
- Prediction controls change the real model's test predictions and score them again
  (PermutedWeights, RandomWeights, SignFlip, RandomEntries, LagShift). They never retrain,
  so they are cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ._data import column_positions, drop_columns, n_rows, shuffle_columns
from .shuffle import block_permutation, random_signs

Score = Callable[[np.ndarray, np.ndarray], float]


def predict_with(model: Any, X: Any) -> np.ndarray:
    """Run whatever fit returned: a predict function, or an object with a .predict method."""
    if hasattr(model, "predict"):
        out = model.predict(X)
    elif callable(model):
        out = model(X)
    else:
        raise TypeError(
            "fit must return a predict function or an object with a .predict method, "
            f"got {type(model).__name__}"
        )
    return np.asarray(out)


@dataclass
class Context:
    """What a control can see. run_controls builds it once and passes it to every control.

    The test labels and the real predictions are numpy arrays. The training and test
    features keep the type you passed in (array, DataFrame or list).
    """

    y_test: np.ndarray
    predictions: np.ndarray
    real_score: float
    score: Score
    fit: Callable[[Any, np.ndarray], Any] | None = None
    X_train: Any = None
    y_train: np.ndarray | None = None
    X_test: Any = None

    def score_predictions(self, predictions: np.ndarray) -> float:
        """Score a set of test predictions against the real test labels."""
        return float(self.score(self.y_test, predictions))

    def fit_and_score(self, X_train: Any, y_train: np.ndarray, X_test: Any) -> float:
        """Train on (X_train, y_train), predict X_test, and score against the real test labels."""
        if self.fit is None:
            raise ValueError("this control retrains the model, so it needs run_controls with fit=")
        model = self.fit(X_train, y_train)
        return self.score_predictions(predict_with(model, X_test))


class Control:
    """Base class for a control. Subclasses implement placebo_scores."""

    needs_refit: ClassVar[bool] = False

    @property
    def label(self) -> str:
        """The name shown in the report. A `name=` argument replaces the default."""
        custom = getattr(self, "name", None)
        return custom if custom else self.default_label()

    def default_label(self) -> str:
        return type(self).__name__

    def placebo_scores(
        self, ctx: Context, B: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the placebo scores and a dict of extra details for the report."""
        raise NotImplementedError


def _block_label(cls_name: str, block: int | None) -> str:
    return f"{cls_name}(block={block})" if block is not None else cls_name


def _one_number_per_row(predictions: np.ndarray, label: str) -> np.ndarray:
    if predictions.ndim != 1:
        raise ValueError(
            f"{label} needs one prediction per row (a 1-D array), got shape {predictions.shape}"
        )
    return predictions


# --------------------------------------------------------------------------- refit controls


@dataclass
class ShuffledLabels(Control):
    """Retrain on shuffled training labels, then score on the real test labels.

    Only the training labels move. The test labels stay real, so each placebo asks: can a
    model that learned from noise still score this well? With `block`, the training
    labels are shuffled in runs of `block` rows, in the order of train_idx, so each
    shuffled training set keeps the short-range patterns of the real one.
    """

    block: int | None = None
    name: str | None = None

    needs_refit: ClassVar[bool] = True

    def default_label(self) -> str:
        return _block_label("ShuffledLabels", self.block)

    def placebo_scores(self, ctx, B, rng):
        y_train = ctx.y_train
        scores = np.empty(B)
        for b in range(B):
            order = block_permutation(len(y_train), self.block, rng)
            scores[b] = ctx.fit_and_score(ctx.X_train, y_train[order], ctx.X_test)
        return scores, {}


@dataclass
class Ablate(Control):
    """Take one group of feature columns away from the model, retrain, and score.

    columns: column names (for a DataFrame) or integer positions (for a numpy array),
        or a single one.
    group: a short name for the group, shown in the report.
    how: "shuffle" (default) shuffles the group's rows, all columns together, in both
        the training and the test rows, B times. "drop" removes the columns and retrains
        once, so it gives a single placebo score.
    keep_only: take away every column except this group instead.
    block: shuffle in runs of `block` rows (only for how="shuffle").
    """

    columns: Sequence[Any]
    group: str | None = None
    how: str = "shuffle"
    keep_only: bool = False
    block: int | None = None
    name: str | None = None

    needs_refit: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.columns, (str, int, np.integer)):
            self.columns = [self.columns]  # one column, such as "temp", 3 or np.int64(3)
        self.columns = list(self.columns)
        if not self.columns:
            raise ValueError("Ablate needs at least one column")
        if self.how not in ("shuffle", "drop"):
            raise ValueError(f"how must be 'shuffle' or 'drop', got {self.how!r}")
        if self.how == "drop" and self.block is not None:
            raise ValueError("block only applies to how='shuffle'")

    def default_label(self) -> str:
        group = self.group or ",".join(str(c) for c in self.columns)
        text = f"all but {group}" if self.keep_only else group
        if self.how == "drop":
            text += ", drop"
        return f"Ablate({text})"

    def placebo_scores(self, ctx, B, rng):
        positions = column_positions(ctx.X_train, self.columns)
        column_positions(ctx.X_test, self.columns)  # same check on the test rows

        if self.how == "drop":
            X_train = drop_columns(ctx.X_train, positions, self.keep_only)
            X_test = drop_columns(ctx.X_test, positions, self.keep_only)
            score = ctx.fit_and_score(X_train, ctx.y_train, X_test)
            return np.array([score]), {"how": "drop"}

        if self.keep_only:
            width = ctx.X_train.shape[1]
            targets = [i for i in range(width) if i not in positions]
            if not targets:
                raise ValueError("keep_only=True with every column in the group removes nothing")
        else:
            targets = positions
        n_train = n_rows(ctx.X_train)
        n_test = n_rows(ctx.X_test)
        scores = np.empty(B)
        for b in range(B):
            train_order = block_permutation(n_train, self.block, rng)
            test_order = block_permutation(n_test, self.block, rng)
            X_train = shuffle_columns(ctx.X_train, targets, train_order)
            X_test = shuffle_columns(ctx.X_test, targets, test_order)
            scores[b] = ctx.fit_and_score(X_train, ctx.y_train, X_test)
        return scores, {"how": "shuffle"}


# ---------------------------------------------------------------------- prediction controls


@dataclass
class PermutedWeights(Control):
    """The same test predictions, shuffled across rows.

    "Weights" means whatever your model outputs per row: a prediction, a score, a size,
    an allocation. The placebo keeps their exact values, so their mean, spread and shape
    are the same. Only their link to the right row is gone. Use `block` when the rows are
    in time order and the predictions drift slowly.
    """

    block: int | None = None
    name: str | None = None

    def default_label(self) -> str:
        return _block_label("PermutedWeights", self.block)

    def placebo_scores(self, ctx, B, rng):
        predictions = ctx.predictions
        scores = np.empty(B)
        for b in range(B):
            order = block_permutation(len(predictions), self.block, rng)
            scores[b] = ctx.score_predictions(predictions[order])
        return scores, {}


@dataclass
class RandomWeights(Control):
    """Random test predictions with the same mean and spread as the real ones.

    By default the values are drawn from a normal distribution, then rescaled so their
    mean and standard deviation match the real predictions exactly. With
    nonnegative=True (for budgets, sizes or rates that cannot go below 0) they come from a
    gamma distribution with the same mean and spread, then rescaled so the mean matches
    exactly.
    """

    nonnegative: bool = False
    name: str | None = None

    def default_label(self) -> str:
        return "RandomWeights(nonnegative)" if self.nonnegative else "RandomWeights"

    def placebo_scores(self, ctx, B, rng):
        predictions = _one_number_per_row(ctx.predictions, self.label).astype(float)
        if self.nonnegative and predictions.min() < 0:
            raise ValueError("nonnegative=True, but some real predictions are below 0")
        mean = float(predictions.mean())
        spread = float(predictions.std())
        scores = np.empty(B)
        for b in range(B):
            weights = self._draw(len(predictions), mean, spread, rng)
            scores[b] = ctx.score_predictions(weights)
        return scores, {"mean": mean, "std": spread}

    def _draw(self, n: int, mean: float, spread: float, rng: np.random.Generator) -> np.ndarray:
        if spread == 0:
            return np.full(n, mean)
        if not self.nonnegative:
            z = rng.standard_normal(n)
            z = (z - z.mean()) / z.std()
            return mean + spread * z
        # Gamma with shape k and scale t has mean k*t and variance k*t^2.
        # Solving for the real mean and variance gives the two numbers below.
        shape = (mean / spread) ** 2
        scale = spread**2 / mean
        weights = rng.gamma(shape, scale, size=n)
        return weights * (mean / weights.mean())


@dataclass
class SignFlip(Control):
    """The same test predictions with random signs: sizes kept, directions made random.

    Each prediction p becomes center + s * (p - center), where s is +1 or -1 at random.
    Use center=0.5 for probabilities. If the score survives random signs, the model's
    lift comes from how big its calls are (or when it makes them), not from their
    direction. The report also gives the score with every sign flipped
    (details["full_flip_score"]).
    """

    block: int | None = None
    center: float = 0.0
    name: str | None = None

    def default_label(self) -> str:
        return _block_label("SignFlip", self.block)

    def placebo_scores(self, ctx, B, rng):
        predictions = _one_number_per_row(ctx.predictions, self.label)
        offset = predictions - self.center
        scores = np.empty(B)
        for b in range(B):
            signs = random_signs(len(predictions), self.block, rng)
            scores[b] = ctx.score_predictions(self.center + signs * offset)
        full_flip = ctx.score_predictions(self.center - offset)
        return scores, {"full_flip_score": full_flip}


@dataclass
class RandomEntries(Control):
    """For event-style predictions: the same events, placed on random rows.

    An event is a non-zero prediction, such as an alert, an offer sent, or a +1/-1 call.
    Each placebo keeps the number of events and their values, and puts them on random
    rows (without repeats). Every other row gets 0.

    eligible: an optional boolean mask over the test rows. Placebo events only land on
        rows where it is True (for example, hours when the machine was running). Every
        real event must be on an eligible row too, or the placebos could never match it.
    """

    eligible: Any = field(default=None, repr=False)
    name: str | None = None

    def placebo_scores(self, ctx, B, rng):
        predictions = _one_number_per_row(ctx.predictions, self.label)
        n = len(predictions)
        event_rows = np.flatnonzero(predictions != 0)
        if event_rows.size == 0:
            raise ValueError("RandomEntries found no events (non-zero predictions) to move")
        values = predictions[event_rows]
        if self.eligible is None:
            pool = np.arange(n)
        else:
            mask = np.asarray(self.eligible, dtype=bool)
            if mask.shape != (n,):
                raise ValueError(f"eligible must be a boolean mask of length {n}")
            pool = np.flatnonzero(mask)
        if event_rows.size > pool.size:
            raise ValueError(
                f"{event_rows.size} events but only {pool.size} eligible rows to put them on"
            )
        outside = np.setdiff1d(event_rows, pool)
        if outside.size:
            raise ValueError(
                f"{outside.size} real events are on rows that eligible marks False; "
                "the placebos can never put an event there, so the test would be unfair"
            )
        scores = np.empty(B)
        for b in range(B):
            placebo = np.zeros_like(predictions)
            rows = rng.choice(pool, size=event_rows.size, replace=False)
            placebo[rows] = rng.permutation(values)
            scores[b] = ctx.score_predictions(placebo)
        return scores, {"events": int(event_rows.size), "eligible_rows": int(pool.size)}


@dataclass
class LagShift(Control):
    """Move the test predictions in time against the outcomes, then score again.

    The shift wraps around the end (the last rows move to the front), so every
    prediction is kept and their order is kept. This keeps the predictions' own
    patterns over time, so it is a fair placebo for time series.

    lags: shift by exactly these numbers of rows (B is then ignored). Negative lags move
        predictions earlier. Without lags, B different shifts are drawn at random between
        min_lag and n - min_lag rows. If there are no more than B such shifts, every one
        of them is used once, so the control can have fewer than B runs.
    min_lag: the smallest random shift. Set it above the time over which outcomes stay
        linked, so a placebo cannot score well just because nearby rows look alike.

    The shifts are drawn without repeats. With min_lag=1 and rows that are exchangeable
    when there is no skill, that makes the p-value exact (Phipson and Smyth, 2010,
    section 5). Drawing with repeats would make it a little too small on short test sets.
    """

    lags: Sequence[int] | None = None
    min_lag: int = 1
    name: str | None = None

    def __post_init__(self) -> None:
        if self.lags is not None:
            self.lags = [int(lag) for lag in self.lags]
            if not self.lags:
                raise ValueError("lags is empty")
        if self.min_lag < 1:
            raise ValueError("min_lag must be at least 1")

    def default_label(self) -> str:
        if self.lags is None:
            return "LagShift" if self.min_lag == 1 else f"LagShift(min_lag={self.min_lag})"
        if len(self.lags) <= 4:
            return "LagShift(lags=" + ",".join(str(lag) for lag in self.lags) + ")"
        return f"LagShift({len(self.lags)} lags)"

    def placebo_scores(self, ctx, B, rng):
        predictions = ctx.predictions
        n = len(predictions)
        if self.lags is None:
            allowed = np.arange(self.min_lag, n - self.min_lag + 1)
            if allowed.size == 0:
                raise ValueError(f"min_lag={self.min_lag} is too large for {n} test rows")
            if B >= allowed.size:
                lags = allowed  # every allowed shift, once each
            else:
                lags = rng.choice(allowed, size=B, replace=False)
        else:
            lags = np.asarray(self.lags, dtype=np.int64)
            if np.any(lags % n == 0):
                raise ValueError(f"a lag of 0 (or a multiple of {n}) would not move anything")
        scores = np.empty(len(lags))
        for i, lag in enumerate(lags):
            scores[i] = ctx.score_predictions(np.roll(predictions, int(lag), axis=0))
        return scores, {"lags": [int(lag) for lag in lags]}
