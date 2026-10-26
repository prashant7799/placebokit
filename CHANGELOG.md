# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-23

First release.

### Added

- `run_controls(X, y, fit=..., score=..., train_idx=..., test_idx=..., controls=...)` trains
  the real model once, then runs each control and returns a `ControlReport`. Any model works:
  `fit` returns a predict function or an object with `.predict`.
- `run_prediction_controls(y_true, y_pred, ...)` for when only test labels and predictions
  exist.
- Refit controls: `ShuffledLabels` (training labels only, with an optional block shuffle for
  time series) and `Ablate` (shuffle or drop a group of feature columns, or keep only that
  group; one column can be given as a name, a Python integer or a numpy integer).
- Prediction controls: `PermutedWeights` (optional blocks), `RandomWeights` (mean and spread
  matched exactly; a non-negative gamma version for budgets and sizes), `SignFlip` (random
  signs around a centre, optional blocks, plus the fully flipped score), `RandomEntries`
  (same events on random rows, with an optional eligible mask that the real events must
  respect) and `LagShift` (fixed circular shifts, or random ones drawn without repeats so
  the p-value is exact for exchangeable rows).
- `empirical_p_value`: (k + 1) / (B + 1), never 0, with ties and NaN placebo scores counted
  against the model.
- A text report (`print(report)`, `report.to_text`) that prints p-values with enough
  decimals for the number of runs, never fewer than 3 or than `digits`, so the smallest
  p-value never shows as 0.000 (with B = 9,999 it shows as 0.0001). Numbers that round to
  zero print as 0.000, not -0.000.
- Each control gets its own random stream from one seed, so results repeat and adding a
  control does not change the others.
- Train/serve parity checks: `check_parity`, `check_repeatable`, `check_causal` (moves every
  future number up or down at random, with early cuts at rows 0, 1, 2, 4, ... that land
  inside warm-up periods) and `check_warmup` (finds the smallest serving window), all
  returning a `ParityReport`. Feature output that lost the data's row labels raises an
  error, also for data with the default labels 0, 1, 2, ... (both checks run the feature
  function once more on the data without its first row), and a causal check that compared
  nothing fails.
- `placebokit.testing.assert_beats_placebos` (all controls, a list of names or one name)
  and `assert_parity` for CI, and an optional pytest plugin with `--placebo-b` and
  `--placebo-seed`. When B is too small to ever reach alpha, the error gives the exact
  smallest p-value, for example 1/99 = 0.0101.
- Tests: golden values from Table 1 of Phipson and Smyth (2010), exact level under
  exchangeability, two-sided null calibration of every prediction control and of
  `ShuffledLabels`, an exact rank test for `LagShift`, a serial-data demonstration that
  plain shuffles are too easy to beat, planted signals and planted look-ahead leaks,
  property tests, refusal of bad input, and a check that the README shows the real code
  and output of the examples.
- Examples on synthetic data: sensor alarms, imbalanced accuracy, coupon budgets, slow-moving
  forecasts, feature parity and a copyable CI test file.
- Input checks: row positions must be whole numbers, DataFrame column names must be unique
  for `Ablate`, and `Ablate` keeps column types such as categorical when it shuffles.

### Planned

- Paired block bootstrap confidence intervals for the gap between the real score and a
  placebo.
- A cross-validation helper that runs the controls on each fold and combines the results.
- A `feature_fn` hook so features are rebuilt inside each control.
- A pitfalls guide with runnable demos (gap-free walk-forward splits, an embargo that is
  computed but not used, shuffled k-fold on time series, tests that pass on empty training
  sets).
- An optional drift extra (population stability index with open-ended outer bins).
- Release on PyPI.
