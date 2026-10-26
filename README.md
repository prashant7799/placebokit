# placebokit

[![CI](https://github.com/prashant7799/placebokit/actions/workflows/ci.yml/badge.svg)](https://github.com/prashant7799/placebokit/actions/workflows/ci.yml)

placebokit runs negative controls on a model result. A negative control is a placebo: a version of your result that cannot know the answer. It shuffles the training labels and retrains, hides a group of features, shuffles the model's predictions across rows, gives them random signs, or moves predicted events to random times. If a placebo scores as well as your model, the score is no evidence of real skill. Each control returns its placebo scores and an empirical p-value, (k + 1) / (B + 1), that is never 0. It also checks train/serve parity: features computed in one batch for training must match the same features computed live. You bring a fit function and a score function, so it works with any model. It has pytest helpers that fail a CI build when a model stops beating its placebos. It is for data scientists and ML engineers who evaluate models on tabular or time-series data, MLOps teams who gate retrains in CI, and anyone who has to answer "is this gain real?" before trusting it.

## What it does NOT do

- **It does not prove a model is good.** Beating the placebos is necessary, not sufficient. A model can beat every placebo and still be useless in practice.
- **It does not find leaks in your features by itself.** A feature that contains the answer helps the real model, so the real model beats every placebo. `check_causal` catches one kind of leak: feature code that reads later rows. `Ablate` shows which group of features carries the score, but you must judge whether that is plausible.
- **It does not say how big the gain is.** It asks whether a placebo can match the real score. It gives no confidence interval for the gap.
- **It cannot see steps fitted outside your fit function.** If a scaler, an encoder or a feature selection step was fitted on all rows before the split, that leak is invisible to it. Put every learned step inside `fit`.
- **It does not look inside the model.** "Weights" here means the model's output for each row (a prediction, a size, an allocation), not its parameters.
- **It is not a drift monitor.** It does not watch live data change over time.
- **It runs one train/test split per call.** It does not run cross-validation for you. Run it once per fold if you need that.
- **It does not correct for many tests.** If you try 50 models and keep the one that beats its placebos, its p-value no longer means what it says. The same holds for trying many controls, seeds or splits and reporting the best one.
- **It does not choose B, block sizes or lags for you.** The defaults are a starting point. The concepts below explain how to pick them.
- **It does not train anything itself.** It only needs numpy and pandas. Your model can come from any library.

## Install

```bash
pip install git+https://github.com/prashant7799/placebokit
```

A PyPI release will come later. You need Python 3.10 or newer. The only dependencies are numpy and pandas.

## Quickstart (60 seconds)

This builds 2,000 hours of synthetic machine data. The target depends on one vibration sensor and one temperature sensor. A least-squares model is checked against five placebos.

<!-- code: examples/quickstart.py -->
```python
import numpy as np

from placebokit import Ablate, PermutedWeights, ShuffledLabels, SignFlip, run_controls

# 2,000 hours of synthetic machine data: 3 vibration and 3 temperature sensors.
rng = np.random.default_rng(0)
X = rng.normal(size=(2000, 6))
# The next-hour load change depends on one vibration and one temperature sensor.
y = 0.6 * X[:, 0] + 0.3 * X[:, 3] + rng.normal(size=2000)
train, test = np.arange(1500), np.arange(1500, 2000)  # older hours train, newer hours test


def fit(X_train, y_train):
    weights, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)  # least squares
    return lambda X_new: X_new @ weights  # fit returns a predict function


def score(y_true, y_pred):
    return 1 - np.mean((y_true - y_pred) ** 2) / np.var(y_true)  # R squared: higher is better


report = run_controls(
    X, y, fit=fit, score=score, train_idx=train, test_idx=test,
    controls=[
        ShuffledLabels(block=24),              # retrain on shuffled answers, in 24-hour runs
        Ablate([0, 1, 2], group="vibration"),  # retrain with the vibration sensors shuffled
        Ablate([4, 5], group="unused"),        # two sensors the target never used
        PermutedWeights(),                     # the real predictions, on the wrong hours
        SignFlip(),                            # the real predictions, random directions
    ],
    B=99, seed=0,
)
print(report)
```

Output:

<!-- output: examples/quickstart.py -->
```text
placebokit report (B=99, seed=0, higher is better)
real score: 0.278

control                   runs    mean    best  p-value
ShuffledLabels(block=24)    99  -0.001   0.064    0.010  beaten
Ablate(vibration)           99   0.047   0.054    0.010  beaten
Ablate(unused)              99   0.278   0.286    0.470  NOT beaten
PermutedWeights             99  -0.255  -0.145    0.010  beaten
SignFlip                    99  -0.257  -0.119    0.010  beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05
```

How to read it:

- The real model explains 27.8% of the variation (R squared 0.278).
- Models trained on shuffled answers explain nothing (mean -0.001). None of the 99 reached 0.278, so k = 0 and p = 1/100 = 0.01. That is the smallest p-value 99 runs can give.
- Shuffling the vibration sensors drops the score to about 0.05. The model really uses them.
- Shuffling the two unused sensors changes nothing (0.278), so that placebo is not beaten. Here that is the expected answer: those sensors carry no signal. A large p-value on an ablation means "this group adds nothing".
- The real predictions on the wrong hours, or with random directions, score far below 0. The model's skill is in matching the right hour and the right direction.

To fail a CI build when the model stops beating its placebos, see [pytest and CI](#pytest-and-ci).

## Concepts

### The idea

A negative control is a version of the experiment where the answer is known to be "no effect". Medicine has the sugar pill. Lab work has the sample with no active ingredient. For a model, a placebo is a version that cannot know the answer: it was trained on shuffled labels, it lost a group of features, or its predictions were moved to the wrong rows.

If a placebo scores like your model, then something other than real skill made the score. Common causes are a quirk of the metric (such as a high base rate), a pattern that comes for free (such as knowing *when* big outcomes happen but not *which way*), or rows that are linked in time and line up by chance.

```text
                       training rows                          test rows
  X, y ──────► fit(X_train, y_train) ──► predict(X_test) ──► score(y_test, pred) = REAL SCORE
                     ▲                                │
   refit controls    │                                │   prediction controls
   (retrain B times) │                                │   (no retraining)
                     │                                ▼
   ShuffledLabels: shuffle y_train          PermutedWeights: shuffle pred across rows
   Ablate: shuffle or drop a column group   RandomWeights:   random pred, same mean and spread
                                            SignFlip:        random signs, same sizes
                                            RandomEntries:   same events, random rows
                                            LagShift:        pred moved in time

  Every placebo is scored against the REAL test labels.
  p-value = (k + 1) / (B + 1), where k = placebo runs that scored at least as well.
```

### The p-value

For each control, placebokit counts k, the placebo runs that scored at least as well as the real model. The p-value is (k + 1) / (B + 1). The +1 on top and bottom counts the real run as one of the possible outcomes (Phipson and Smyth, 2010). So:

- The p-value is never 0. A report that prints "p = 0.000" after a finite number of shuffles is wrong. placebokit's text report adds decimals as B grows, so 1 / (B + 1) always shows: with B = 9,999 it prints 0.0001.
- With B runs, the smallest p-value is 1 / (B + 1):

  | B | smallest p-value |
  |---|---|
  | 19 | 0.05 |
  | 99 | 0.01 |
  | 999 | 0.001 |

- Ties count against the model. A placebo run that returns NaN (for example, a correlation of constant predictions) also counts against the model.
- If the placebo runs and the real run are exchangeable when there is no skill, the p-value is valid: P(p <= j / (B + 1)) <= j / (B + 1), with equality when there are no ties.

A small p-value says the real score is hard to explain with that placebo. A large one says the placebo can match it.

### Which control answers which question

| Control | What the placebo keeps | What it breaks | Question it answers |
|---|---|---|---|
| `ShuffledLabels()` | features, test labels, the fitting code | the link between features and training labels | Could a model trained on noise score this well? |
| `Ablate(cols)` | everything else | the group's link to the rest of the row | Does the model need this group of features? |
| `PermutedWeights()` | the exact set of predictions | which row each prediction belongs to | Is it the predictions' distribution, or their placement? |
| `RandomWeights()` | the mean and spread of the predictions | everything else | Is it the average size of the outputs (a budget, a dose), or where they go? |
| `SignFlip()` | the size of each prediction | its direction | Is the skill in the direction, or only in the size or timing? |
| `RandomEntries()` | the number of events and their values | when they happen | Is the timing of the events real? |
| `LagShift()` | the predictions and their pattern over time | their alignment with the outcomes | Does it work only when aligned with the right moment? |

Refit controls (`ShuffledLabels`, `Ablate`) call your `fit` once per run. Prediction controls never retrain, so they are cheap even with a large B.

### Rows close in time are linked

In a time series, each row looks like its neighbours. A plain shuffle breaks that, and it can make a placebo far too easy to beat. The [serial predictions example](#examples) shows this. Two unrelated slow-moving series line up by chance. A plain shuffle calls the skill-free forecast "skilled" in 32% of trials. A block shuffle or a lag shift keeps the slow movement and brings that close to the nominal 5%.

- `block=` (on `ShuffledLabels`, `Ablate`, `PermutedWeights` and `SignFlip`) cuts the rows into runs of `block` rows. Only the order of the runs changes. The order inside each run is kept. Pick a block longer than the time over which rows stay linked.
- `LagShift(min_lag=...)` slides the predictions in time. It wraps around the end, so nothing is lost. Pick `min_lag` longer than the time over which outcomes stay linked.
- For `ShuffledLabels`, `block` keeps each shuffled training set's runs and local base rates like the real one's. This matters most for models that learn from label order or label runs. It does not fix overlap between training and test rows. Leave a gap between them for that.

### Scores that ignore size

Correlation and AUC ignore the overall scale of the predictions. Only their ranking or direction matters. A model trained on shuffled labels still picks some random direction, and with only a few features, a random direction can line up with the signal by luck. Then `ShuffledLabels` is a weak control. A score that also rewards the right size, such as R squared or a loss, gives a stronger one. The quickstart uses R squared for this reason.

### Event-style predictions

Some models do not predict a number for every row. They raise an alert, send an offer, or call "+1" or "-1" on a few rows. Give those predictions as an array with 0 for "no event". `RandomEntries` keeps the events and moves them to random rows. `SignFlip` keeps the events and their timing but gives them random sides. Together they separate timing skill from direction skill. The [alarm example](#examples) shows a model that knows *when* big moves come but not *which way*.

### Train/serve parity

In training, features are usually computed in one batch over the whole history. When the model is served, the same features are computed live, from only the rows that exist at that moment. The two must give the same numbers. A feature function takes a DataFrame (or Series) of raw rows in time order and returns a DataFrame (or Series) of features with the same row labels. It may drop rows at the start.

```text
  training:  feature_fn(all rows)  ──────────────►  value at row t  ─┐
                                                                     ├── must match
  serving:   feature_fn(last `window` rows up to t) ──► value at t  ─┘
```

- `check_repeatable` runs the feature function twice on fresh copies. The results must match, and the input must not change. Unseeded randomness and functions that add columns to their input fail.
- `check_causal` changes every number after a cut, each one up or down at random, and checks that no feature value at or before the cut moves. The cuts start at rows 0, 1, 2, 4, 8, and so on, so some cut lands inside the warm-up period of almost any feature. That is where back-filling from later rows hides.
- `check_warmup` computes the features from only the last `window` rows and compares the value at the last row with the full-history value. It finds how much history a live server must keep.
- `check_causal` and `check_warmup` match feature rows to raw rows by label. If the feature function drops the input's row labels (for example with `reset_index`), they raise an error instead of comparing nothing or the wrong rows. This includes data with the default labels 0, 1, 2 and so on, where the new labels look like the old ones: both checks also run the feature function once on the data without its first row, whose labels start at 1.
- `check_parity` compares any two feature tables, for example the output of your training pipeline and your serving pipeline.

### Seeds

`run_controls(..., seed=0)` makes every run repeatable. Each control gets its own random stream from that seed, so adding a control at the end of the list does not change the others.

## API reference

### `run_controls`

```python
run_controls(X, y, *, fit, score, train_idx, test_idx, controls, B=99, seed=0,
             higher_is_better=True) -> ControlReport
```

Trains the real model once, scores it, then runs every control.

- `X`: features. A 2-D numpy array, a pandas DataFrame, or a list (for example texts).
- `y`: labels, one per row of `X`.
- `fit(X_train, y_train)`: trains a model. It returns either a predict function (`X -> predictions`) or an object with a `.predict` method. `X_train` has the same type as `X`. `y_train` is a numpy array.
- `score(y_true, y_pred) -> float`: both arguments are numpy arrays.
- `train_idx`, `test_idx`: row positions or boolean masks. They must not overlap. For time series, give them in time order.
- `controls`: a list of controls. Names must be unique; pass `name=` to tell two apart.
- `B`: placebo runs per control.
- `seed`: an integer for repeatable runs, or None.
- `higher_is_better`: False for losses such as mean squared error.

### `run_prediction_controls`

```python
run_prediction_controls(y_true, y_pred, *, score, controls, B=99, seed=0,
                        higher_is_better=True) -> ControlReport
```

The same, when you only have test labels and predictions (for example from a saved predictions file). Only prediction controls are allowed.

### Controls

Every control takes an optional `name=` that replaces its label in the report.

| Control | Arguments |
|---|---|
| `ShuffledLabels(block=None)` | `block`: shuffle the training labels in runs of this many rows. |
| `Ablate(columns, group=None, how="shuffle", keep_only=False, block=None)` | `columns`: names (DataFrame) or positions (array), or a single one. `group`: a short name for the report. `how="shuffle"` shuffles the group's rows (all its columns together) in both the training and the test rows, B times. `how="drop"` removes the columns and retrains once, so it gives one placebo score and a p-value of 0.5 or 1.0. `keep_only=True` takes away every *other* column instead. |
| `PermutedWeights(block=None)` | `block`: shuffle the predictions in runs of this many rows. Works for 2-D predictions too (rows move as a whole). |
| `RandomWeights(nonnegative=False)` | Normal values rescaled to the real mean and standard deviation exactly. `nonnegative=True` draws from a gamma distribution with the same mean and spread instead (for budgets, sizes or rates), rescaled so the mean matches exactly. |
| `SignFlip(block=None, center=0.0)` | Each prediction p becomes `center + s * (p - center)` with a random sign s. Use `center=0.5` for probabilities. `block` gives each run of rows one sign. `details["full_flip_score"]` is the score with every sign flipped. |
| `RandomEntries(eligible=None)` | Non-zero predictions are events. Each placebo puts the same events (same values) on random rows, without repeats. `eligible` is an optional boolean mask: events only land where it is True. Every real event must be on an eligible row, or it raises an error. |
| `LagShift(lags=None, min_lag=1)` | `np.roll` of the predictions. `lags=[...]` uses exactly those shifts (B is then ignored). Otherwise B different shifts are drawn between `min_lag` and `n - min_lag`. If there are no more than B such shifts, each is used once, so there are fewer than B runs. `details["lags"]` lists them. |

### `ControlReport` and `ControlResult`

- `report.real_score`, `report.B`, `report.seed`, `report.higher_is_better`
- `report["SignFlip"]` returns a `ControlResult`. `report.names` lists the names. You can loop over a report.
- `report.passed(alpha=0.05)`: True when every control has p-value <= alpha.
- `report.to_frame(alpha=0.05)`: a DataFrame with `control, runs, mean, std, best, p_value, beaten`.
- `report.to_text(alpha=0.05, digits=3)`: the text table, with `digits` decimals for the scores. p-values get at least 3 decimals and never fewer than `digits`, and more from 1,000 runs on, so the smallest one never prints as 0. `print(report)` uses the defaults.
- `result.scores` (numpy array), `result.p_value`, `result.runs`, `result.mean`, `result.std`, `result.best`, `result.n_failed` (NaN runs), `result.min_p_value`, `result.beaten(alpha)`, `result.details` (a dict).

### Statistics and shuffles

- `empirical_p_value(real_score, placebo_scores, higher_is_better=True)`: (k + 1) / (B + 1), with ties and NaN placebo scores counted against the model.
- `block_permutation(n, block, rng)`: a random order of `0..n-1`. With a block size, whole blocks move.
- `random_signs(n, block, rng)`: n random +1/-1 values. With a block size, each block shares one sign.

### Parity checks

All of these return a `ParityReport` with `ok`, `n_compared`, `n_mismatched`, `max_abs_diff`, `first_mismatch`, `notes` and (for `check_warmup`) `min_window`. `print(report)` explains the result. Numbers match within `atol + rtol * |b|` (defaults `rtol=1e-9`, `atol=1e-12`). A missing value matches only another missing value.

- `check_parity(a, b, *, rtol, atol)`: compare two tables (DataFrame, Series or array), rows by label and columns by name.
- `check_repeatable(feature_fn, data, *, rtol, atol)`
- `check_causal(feature_fn, data, *, cuts=None, perturb=None, seed=0, rtol, atol)`: `cuts` are row positions. `perturb(future_rows, rng)` returns changed rows with the same labels. By default `perturb_numbers` moves every float by 1 to 2 times the column's spread and every integer by 1 to 9, each up or down at random (unsigned integers only move up), and flips booleans. Text and date columns are left alone. The check fails if it compared no values at all.
- `check_warmup(feature_fn, data, windows, *, rows=None, rtol, atol)`: `windows` is one window (to assert that a serving window is enough) or a list to search from the smallest up. `rows` are the row positions to check; by default five rows spread over the second half, including the last row. The counts in the report are for the last window tried.

### pytest and CI

`placebokit.testing` has two assertions. They raise `AssertionError`, so pytest shows them as normal test failures, and they do not need pytest.

- `assert_beats_placebos(report, alpha=0.05, controls=None)`: fails unless every listed control (a list of names or one name; default: all) has p-value <= alpha. It raises `ValueError` instead when a control has too few runs to ever reach alpha (with alpha = 0.05 you need B >= 19), because a test that can never pass is a broken test. An empty `controls` list also raises, because it would pass without testing anything.
- `assert_parity(parity_report)`: fails when a parity check did not pass.

```python
from placebokit import ShuffledLabels, SignFlip, assert_beats_placebos, run_controls


def test_model_beats_placebos():
    report = run_controls(X, y, fit=fit, score=score, train_idx=train, test_idx=test,
                          controls=[ShuffledLabels(block=24), SignFlip()], B=99, seed=0)
    assert_beats_placebos(report)
```

An optional pytest plugin lets CI choose B from the command line, for a quick check on every pull request and a larger one every night. Turn it on in your top-level `conftest.py` with `pytest_plugins = ["placebokit.pytest_plugin"]`, or run `pytest -p placebokit.pytest_plugin`. Then ask for the fixtures:

```python
def test_model_beats_placebos(placebo_b, placebo_seed):
    report = run_controls(..., B=placebo_b, seed=placebo_seed)
    assert_beats_placebos(report)
```

```bash
pytest                    # B = 99, seed = 0
pytest --placebo-b=999    # nightly
```

### Writing your own control

Subclass `Control` and implement `placebo_scores(ctx, B, rng)`. It returns an array of placebo scores and a dict of details. `ctx` is a `Context` with `y_test`, `predictions`, `real_score`, `score_predictions(pred)`, and, inside `run_controls`, `X_train`, `y_train`, `X_test` and `fit_and_score(X_train, y_train, X_test)`. Set the class attribute `needs_refit = True` if your control retrains.

## Examples

All examples use synthetic data and run in a few seconds. Run them from the repository root, for example `python examples/alarm_direction.py`.

| File | What it shows |
|---|---|
| [`quickstart.py`](examples/quickstart.py) | The quickstart above. |
| [`alarm_direction.py`](examples/alarm_direction.py) | A model that knows *when* big sensor moves come but not *which way*. `SignFlip` catches it; `RandomEntries` shows the timing is real. |
| [`imbalanced_accuracy.py`](examples/imbalanced_accuracy.py) | 90.2% accuracy when 89.5% of the answers are "no". `ShuffledLabels` shows the real floor: the model beats it (p = 0.005), but only by 0.7 points. |
| [`budget_weights.py`](examples/budget_weights.py) | Two coupon-budget allocations. The one that earns more in total (2,086 vs 1,894) only spends twice as much: it does not beat random allocations of the same budget. The targeted one does. |
| [`serial_predictions.py`](examples/serial_predictions.py) | Why time series need block shuffles or lag shifts. |
| [`feature_parity.py`](examples/feature_parity.py) | Six feature functions through the three parity checks. |
| [`test_ci_gate.py`](examples/test_ci_gate.py) | A pytest file to copy: the model must beat its placebos, the features must not look ahead, and the serving window must be enough. Run it with `pytest examples/test_ci_gate.py`. |

The imbalanced accuracy example: a model trained on shuffled answers learns to say "no" every time. That alone is right 89.5% of the time, so the real model's 90.2% is only 0.7 points of skill.

<!-- output: examples/imbalanced_accuracy.py -->
```text
placebokit report (B=199, seed=0, higher is better)
real score: 0.902

control         runs   mean   best  p-value
ShuffledLabels   199  0.895  0.895    0.005  beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05

share of 'no' answers in the test rows: 0.895
lift over the floor: +0.007
```

The budget example: Model B earns more in total only because it spends about twice as much. Random allocations of the same budget earn as much as it does. Model A spends half as much and beats both placebos.

<!-- output: examples/budget_weights.py -->
```text
== Model A (targeted): mean weight 1.00
placebokit report (B=199, seed=2, higher is better)
real score: 1894

control                     runs  mean  best  p-value
RandomWeights(nonnegative)   199  1045  1238    0.005  beaten
PermutedWeights              199  1045  1193    0.005  beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05

== Model B (spends more): mean weight 1.98
placebokit report (B=199, seed=2, higher is better)
real score: 2086

control                     runs  mean  best  p-value
RandomWeights(nonnegative)   199  2067  2178    0.270  NOT beaten
PermutedWeights              199  2071  2173    0.355  NOT beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05
```

The alarm example: Model A raises alarms in noisy hours and picks the side from a useless hint. Its hit rate (0.228) is far above random alarms (0.125). But random sides do just as well (0.230), so its side calls carry no skill. Model B has a weakly useful hint and beats both placebos.

<!-- output: examples/alarm_direction.py -->
```text
== Model A (useless hint): 3264 alarms
placebokit report (B=199, seed=1, higher is better)
real score: 0.228

control        runs   mean   best  p-value
SignFlip        199  0.230  0.244    0.700  NOT beaten
RandomEntries   199  0.125  0.138    0.005  beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05
every side flipped: 0.233

== Model B (useful hint): 3264 alarms
placebokit report (B=199, seed=1, higher is better)
real score: 0.335

control        runs   mean   best  p-value
SignFlip        199  0.230  0.248    0.005  beaten
RandomEntries   199  0.126  0.140    0.005  beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05
every side flipped: 0.125
```

The serial predictions example: a forecast and a reading that drift slowly, generated independently. The plain shuffle says "beaten"; the block shuffle and the lag shift do not. Over 200 such pairs, the plain shuffle calls the skill-free forecast skilled 32% of the time. The block shuffle and the lag shift stay near 5%.

<!-- output: examples/serial_predictions.py -->
```text
placebokit report (B=199, seed=0, higher is better)
real score: 0.163

control                     runs   mean   best  p-value
PermutedWeights              199  0.001  0.127    0.005  beaten
PermutedWeights(block=100)   199  0.006  0.283    0.140  NOT beaten
LagShift(min_lag=100)        199  0.009  0.282    0.205  NOT beaten

p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well
beaten = p-value <= 0.05

Skill-free forecasts called skilled at p <= 0.05, out of 200:
  PermutedWeights             64  (32%)
  PermutedWeights(block=100)   5  (2%)
  LagShift(min_lag=100)       10  (5%)
A fair test should be near 5%.
```

The feature parity example: a centred window, a whole-sample z-score and a back-filled warm-up all look ahead. Exponential smoothing is causal, but a server needs 229 rows of history to reproduce it within the default tolerance (about nine significant digits). A function that adds a column to its input is not repeatable.

<!-- output: examples/feature_parity.py -->
```text
feature function     repeatable  causal   warm-up needed
rolling_features     ok          ok       24 rows
centred_mean         ok          FAILED   never matches
backfilled_mean      ok          FAILED   24 rows
whole_sample_zscore  ok          FAILED   never matches
smoothed             ok          ok       229 rows
adds_column          FAILED      ok       24 rows

Details for backfilled_mean:
causal check: FAILED
  compared 3,521 values, 37 differ (largest difference 0.8368)
  first difference: row 2026-01-01 00:00:00, column 'mean_24h': 19.8664 vs 20.7032
  - future changed after rows: 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 499, 999, 1499
  - after row 0: 1 of 1 past values moved
  - after row 1: 2 of 2 past values moved
  - after row 2: 3 of 3 past values moved
  - after row 4: 5 of 5 past values moved
  - after row 8: 9 of 9 past values moved
  - after row 16: 17 of 17 past values moved
```

## How correctness is tested

The tests use synthetic data only and never touch the network. Run them with `pytest -q`.

- **Golden values from the paper** (Phipson and Smyth, 2010, Table 1). With 100 random permutations and b = 0 to 7 of them at least as good, the reported p-values are 0.0099, 0.0198, ..., 0.0792, as in the table's upper-bound column. The test also recomputes the table's exact column from the paper's equation (1), to nine decimals, and checks that each reported p-value is at or above it. The paper shows this is the safe direction.
- **Hand-counted values.** A real score above all 99 placebos gives exactly 0.01, never 0. A real score below all of them gives 1.0. Real 0.75 against placebos 0.1 to 0.9 gives 3/10. Ties and NaN placebos count against the model. The text report prints 1/2,001 as 0.0005 and 1/10,000 as 0.0001, never as 0.000.
- **Exact level.** When the real score is exchangeable with the placebos, every rank is equally likely. Going through every rank for B = 9, 19 and 99 gives each p-value 1/(B+1), ..., 1 exactly once, so P(p <= j/(B+1)) = j/(B+1). A simulation of 20,000 exchangeable trials with B = 19 rejects at 4.9%, within four standard errors of 5%.
- **Null calibration of each control.** With no real skill, how often is a control "beaten" at alpha = 0.05 with B = 19? `ShuffledLabels` with labels that ignore the features: 4.8% of 1,000 trials. `PermutedWeights`, `RandomWeights`, `SignFlip` and `LagShift` on unrelated normal predictions: 4.85%, 4.55%, 5.0% and 5.1% of 2,000 trials. `RandomEntries` with events on random rows: 5.45% of 2,000 trials. Each rate must be within three standard errors of 5%, so a control that is too strict fails, as well as one that is too loose.
- **Lag shifts are exact.** `LagShift` draws different shifts, never the same one twice. Take all 12 circular shifts of one 12-row forecast: whichever is the real one, the p-values are 1/12, 2/12, ..., 12/12, once each. On 30-row test sets it calls a skill-free forecast skilled 4.6% of the time over 3,000 trials. Drawing shifts with repeats instead gives 6.4% on the same trials.
- **Serial data.** For two unrelated AR(1) series (phi = 0.95, 600 rows, 500 trials), a plain shuffle is "beaten" 32.6% of the time, a 60-row block shuffle 5.6%, and a lag shift of at least 60 rows 8.0%. The test asserts that the plain shuffle is badly off and that the other two stay at or below 10%. They are approximations here, not exact tests.
- **Planted signals.** A real linear signal beats `ShuffledLabels` at the smallest possible p-value. Ablating its column hurts; ablating noise columns does not. A model that only knows *when* outcomes are big does not beat `SignFlip` but does beat `RandomEntries`; a model with direction skill beats both.
- **Mechanics.** Spies on `fit` and `score` confirm that only training labels move, that test labels are always the real ones, that blocks stay whole and in order, that ablated columns move together and keep their types (a categorical column stays categorical), that `RandomWeights` match the mean and spread exactly and `RandomEntries` keep the events.
- **Planted leaks.** `check_causal` catches a centred window, a next-row shift, a whole-sample mean, a whole-sample median, a back-filled warm-up, a back-filled first difference, and a distance from the whole-sample minimum even when the cut comes after the minimum. A trailing window passes. A test shows that only the early cuts catch the back-filled warm-up. Feature output that lost the data's row labels raises an error instead of passing or comparing the wrong rows, both for dated rows and for rows with the default labels 0, 1, 2 and so on. The same feature with its labels kept passes. A causal check that compared nothing fails. A gap of missing values filled from later rows (interpolation or back-filling) is missed by the default cuts and caught by a cut inside the gap.
- **Warm-up.** A 10-row rolling mean needs exactly 10 rows, and the report counts only that window. An expanding mean never matches a window. Exponential smoothing needs fewer rows at a looser tolerance.
- **Property tests** (Hypothesis). The p-value equals a plain count, is the exact mirror image for lower-is-better scores, and depends only on ranks: shuffling the placebo runs or passing every score through the same increasing function changes nothing. Every prediction control gives the same p-values when the score goes through an increasing function. Block permutations are always permutations with whole blocks, every block order is possible, and block signs are shared within each block. Any table matches itself.
- **Bad input is refused.** Row positions that are not whole numbers, repeated column names, real events outside the eligible rows, and an empty list of controls to assert all raise errors.
- **Repeatability.** The same seed gives the same scores, a new seed gives new scores, and adding a control at the end does not change the others.
- **Documentation.** A test runs every example, checks that the quickstart above is the same code as `examples/quickstart.py`, and checks that every output block in this README matches what the code prints.

## Limitations and assumptions

- **Exchangeability.** The p-values are exact only when, without real skill, the real run and the placebo runs are exchangeable. `ShuffledLabels`, `PermutedWeights`, `RandomEntries` and `LagShift` (with `min_lag=1`) are exact when rows are independent. `Ablate` with `how="shuffle"` is exact when the group is also independent of the other columns. `SignFlip` also needs the predictions to be symmetric around `center`. `RandomWeights` also needs the real predictions to look like its random draw (normal, or gamma with `nonnegative=True`); otherwise read it as a benchmark, not an exact test. Block shuffles, and lag shifts with a larger `min_lag`, are approximations for linked rows. They work well when the block or lag is longer than the time over which rows stay linked.
- **One split.** Each call uses one training set and one test set. A lucky test set can still flatter a model; more test data or more folds help.
- **Cost.** Each refit control trains the model B times. Use prediction controls, a smaller B, or a smaller sample when training is slow.
- **Ablation.** Shuffling a group keeps its own values but breaks its link to the rest of the row. If other features carry the same information, the ablation shows little loss. Read the result as "what this group adds beyond the rest".
- **Drop-mode ablation** gives one placebo score, so its p-value is 0.5 or 1.0. Use `how="shuffle"` when you need a p-value.
- **Random weights.** The default normal draw can go below zero. Use `nonnegative=True` or `PermutedWeights` for budgets and sizes.
- **Sign flips and base rates.** If outcomes lean one way, a model that always calls that side beats `SignFlip`. That is direction skill, but a cheap kind. Compare with a constant prediction too.
- **Events** are the non-zero predictions. Encode "no event" as 0.
- **Lag shifts wrap around.** The end of the series is joined to the start. For long series this edge effect is small. A short test set allows only a few shifts, which limits the smallest p-value.
- **Many controls.** Each p-value is for one control. Requiring every control to be beaten is stricter than any one of them.
- **Parity checks** need a feature function that takes a DataFrame or Series in time order and returns features with the same row labels. `check_causal` changes numbers, not text, dates or the number of rows, so a leak through those is not seen. Missing values stay missing too, so a feature that looks ahead only at *which* rows are missing is not caught. It also tries only a few cuts, and some leaks show only at a cut in the right place. Gaps of missing values (NaN) are the usual case: a feature that fills a gap from later rows, by interpolation or by back-filling across the gap, looks ahead only at cuts inside the gap, so the default cuts will usually miss it. If your data has gaps, pass `cuts=` with row positions inside them. `check_warmup` checks a few rows, not every row.

## Related approaches

These are general kinds of tools people use for nearby questions. placebokit makes the choices noted after each.

- **Label-permutation tests for classifiers.** The classic test (Ojala and Garriga, 2010) shuffles all the labels and repeats the whole evaluation, test labels included. placebokit shuffles only the training labels and keeps the test labels real, offers block shuffles for time series, and takes any fit function.
- **Permutation feature importance.** This shuffles a column in the test data without retraining, to see how much a fitted model relies on it. `Ablate` retrains, to see whether the model needs the column at all.
- **Refutation tests for causal effect estimates.** These add placebo treatments or random confounders to check an estimated effect. placebokit checks predictive scores, not causal effects.
- **Look-ahead checkers for feature code.** Some tools focus on testing feature code for future leaks. placebokit keeps this part small (`check_causal`, `check_warmup`) and puts it next to the controls.
- **Drift monitors.** These watch live data and predictions change over time. That is out of scope here.
- **Block bootstrap methods.** These resample blocks of rows to get confidence intervals for dependent data (Künsch, 1989). placebokit uses blocks for the same reason, to respect linked rows, but for placebo tests rather than intervals.

## References and citation

- Phipson, B. and Smyth, G. K. (2010). Permutation P-values should never be zero: calculating exact P-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. doi:10.2202/1544-6115.1585
- Ojala, M. and Garriga, G. C. (2010). Permutation tests for studying classifier performance. *Journal of Machine Learning Research*, 11, 1833-1863.
- Lipsitch, M., Tchetgen Tchetgen, E. and Cohen, T. (2010). Negative controls: a tool for detecting confounding and bias in observational studies. *Epidemiology*, 21(3), 383-388. doi:10.1097/EDE.0b013e3181d61eeb
- Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), Article 15. doi:10.1145/2382577.2382579
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217-1241. doi:10.1214/aos/1176347265
- Breck, E., Cai, S., Nielsen, E., Salib, M. and Sculley, D. (2017). The ML test score: a rubric for ML production readiness and technical debt reduction. *2017 IEEE International Conference on Big Data*, 1123-1132.

If you use placebokit in published work, please cite Phipson and Smyth (2010) for the p-value and, if you like, this software. [`CITATION.cff`](CITATION.cff) has the details.

## Licence

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Prashant Lohar.
