"""Train/serve parity checks for feature code.

In training, features are usually computed in one batch over the whole history. When a
model is served, the same features are computed live, from only the rows that exist at
that moment. The two must give the same numbers. These checks look for the usual ways
they drift apart:

- check_parity: compare two feature tables that should be equal.
- check_repeatable: the same features computed twice must match, and the input must not
  change.
- check_causal: changing future rows must not change past feature values.
- check_warmup: features computed from only the last `window` rows must match the
  full-history values. This finds how much history a server needs.

A feature function takes a pandas DataFrame (or Series) of raw rows in time order and
returns a DataFrame (or Series) of features with the same row labels. It may drop rows at
the start, for example while a rolling window fills up.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_unsigned_integer_dtype,
)

Table = pd.DataFrame | pd.Series
FeatureFn = Callable[[Any], Any]

DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


@dataclass
class ParityReport:
    """The result of a parity check. `ok` is the verdict; str(report) explains it."""

    check: str
    ok: bool
    n_compared: int = 0
    n_mismatched: int = 0
    max_abs_diff: float = 0.0
    first_mismatch: str | None = None
    notes: list[str] = field(default_factory=list)
    min_window: int | None = None

    def __str__(self) -> str:
        lines = [f"{self.check} check: {'passed' if self.ok else 'FAILED'}"]
        line = f"  compared {self.n_compared:,} values, {self.n_mismatched:,} differ"
        if self.n_mismatched and self.max_abs_diff > 0:
            line += f" (largest difference {self.max_abs_diff:.4g})"
        lines.append(line)
        if self.min_window is not None:
            lines.append(f"  smallest window that matches: {self.min_window} rows")
        if self.first_mismatch:
            lines.append(f"  first difference: {self.first_mismatch}")
        for note in self.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- check_parity


def check_parity(
    a: Any,
    b: Any,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    check: str = "parity",
) -> ParityReport:
    """Compare two feature tables cell by cell, matching rows by label and columns by name.

    a, b: DataFrames, Series or numpy arrays. Numbers match when they are within
        atol + rtol * |b| of each other (numpy.isclose). A missing value matches only
        another missing value. Other values must be equal.
    The check fails if a row or a column is in one table only.
    """
    first = _as_frame(a, "the first table")
    second = _as_frame(b, "the second table")
    for table, what in ((first, "first"), (second, "second")):
        if not table.index.is_unique:
            raise ValueError(f"the {what} table has repeated row labels")
        if not table.columns.is_unique:
            raise ValueError(f"the {what} table has repeated column names")

    report = ParityReport(check=check, ok=True)

    only_first = [c for c in first.columns if c not in second.columns]
    only_second = [c for c in second.columns if c not in first.columns]
    if only_first:
        report.notes.append(f"columns only in the first table: {only_first}")
    if only_second:
        report.notes.append(f"columns only in the second table: {only_second}")
    rows_only_first = first.index[~first.index.isin(second.index)]
    rows_only_second = second.index[~second.index.isin(first.index)]
    if len(rows_only_first):
        report.notes.append(
            f"{len(rows_only_first)} rows only in the first table, "
            f"starting at {_fmt(rows_only_first[0])}"
        )
    if len(rows_only_second):
        report.notes.append(
            f"{len(rows_only_second)} rows only in the second table, "
            f"starting at {_fmt(rows_only_second[0])}"
        )
    if report.notes:
        report.ok = False

    columns = [c for c in first.columns if c in second.columns]
    rows = first.index[first.index.isin(second.index)]
    left = first.loc[rows, columns]
    right = second.loc[rows, columns]
    report.n_compared = int(left.size)

    first_bad: tuple[int, Any] | None = None  # (row position, column)
    for col in columns:
        same, diff = _compare_column(left[col], right[col], rtol, atol)
        bad = np.flatnonzero(~same)
        if bad.size == 0:
            continue
        report.ok = False
        report.n_mismatched += int(bad.size)
        if diff is not None:
            finite = diff[bad][np.isfinite(diff[bad])]
            if finite.size:
                report.max_abs_diff = max(report.max_abs_diff, float(finite.max()))
        if first_bad is None or bad[0] < first_bad[0]:
            first_bad = (int(bad[0]), col)

    if first_bad is not None:
        pos, col = first_bad
        report.first_mismatch = (
            f"row {_fmt(rows[pos])}, column {col!r}: "
            f"{_fmt(left[col].iloc[pos])} vs {_fmt(right[col].iloc[pos])}"
        )
    return report


def _compare_column(
    x: pd.Series, y: pd.Series, rtol: float, atol: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Element-wise match of two aligned columns, plus absolute differences for numbers."""
    if is_numeric_dtype(x) and is_numeric_dtype(y):
        xv = x.to_numpy(dtype=float, na_value=np.nan)
        yv = y.to_numpy(dtype=float, na_value=np.nan)
        same = np.isclose(xv, yv, rtol=rtol, atol=atol, equal_nan=True)
        with np.errstate(invalid="ignore"):
            diff = np.abs(xv - yv)
        return same, diff
    xs = pd.Series(x.to_numpy(dtype=object))
    ys = pd.Series(y.to_numpy(dtype=object))
    both_missing = (xs.isna() & ys.isna()).to_numpy()
    equal = np.array([_equal(p, q) for p, q in zip(xs, ys, strict=True)], dtype=bool)
    return equal | both_missing, None


def _equal(p: Any, q: Any) -> bool:
    try:
        return bool(p == q)
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------- check_repeatable


def check_repeatable(
    feature_fn: FeatureFn,
    data: Table,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityReport:
    """Compute the features twice, on two fresh copies of the data. They must match.

    It also fails if feature_fn changed the data it was given (for example by adding a
    column to it), because then the second call in production would see different input.
    """
    _require_table(data)
    original = data.copy(deep=True)
    working = data.copy(deep=True)
    first = feature_fn(working)
    changed_input = not working.equals(original)
    second = feature_fn(original.copy(deep=True))
    report = check_parity(first, second, rtol=rtol, atol=atol, check="repeatable")
    if changed_input:
        report.ok = False
        report.notes.insert(0, "feature_fn changed its input data in place")
    return report


# -------------------------------------------------------------------------- check_causal


def perturb_numbers(rows: Table, rng: np.random.Generator) -> Table:
    """Move every number in these rows by a random amount. Other columns are left alone.

    Floats move by 1 to 2 times the column's spread, integers by 1 to 9, and booleans
    flip. Each row moves up or down at random (unsigned integers only move up). So every
    number changes, and a feature built on a whole-sample minimum or maximum moves too.
    """
    if isinstance(rows, pd.Series):
        moved = _perturb_column(rows, rng)
        if moved is None:
            raise ValueError("the data holds no numbers to change; pass perturb=")
        return moved
    out = rows.copy()
    changed_any = False
    for pos in range(out.shape[1]):
        moved = _perturb_column(out.iloc[:, pos], rng)
        if moved is not None:
            out.isetitem(pos, moved)
            changed_any = True
    if not changed_any:
        raise ValueError("the data has no numeric columns to change; pass perturb=")
    return out


def _perturb_column(col: pd.Series, rng: np.random.Generator) -> pd.Series | None:
    """The column with every value moved, or None for columns that hold no numbers."""
    if is_bool_dtype(col):
        return ~col
    if not (is_integer_dtype(col) or is_float_dtype(col)):
        return None  # text, dates and other columns are left alone
    # +1 or -1 for each row. Moving only up would miss, for example, a feature that
    # subtracts the whole-sample minimum when that minimum lies in the past.
    direction = rng.choice(np.array([-1, 1]), size=len(col))
    if is_integer_dtype(col):
        steps = rng.integers(1, 10, size=len(col))
        if not is_unsigned_integer_dtype(col):
            steps = steps * direction
        return col + steps
    values = col.to_numpy(dtype=float, na_value=np.nan)
    finite = values[np.isfinite(values)]
    scale = float(finite.std()) if finite.size > 1 else 0.0
    if not scale > 0:
        scale = 1.0
    moved = values + direction * scale * rng.uniform(1.0, 2.0, size=len(values))
    return pd.Series(moved, index=col.index, name=col.name).astype(col.dtype)


def default_cuts(n: int) -> list[int]:
    """Cut points: rows 0, 1, 2, 4, 8, ... up to a quarter of the data, then 25%, 50%, 75%.

    The early cuts fall inside the warm-up period of almost any feature, where it has too
    little history. That is where back-filling from later rows tends to hide. Doubling
    the cut each time means that, whatever the warm-up length, some cut lands inside it.
    """
    cuts = {0}
    cut = 1
    while cut < n // 4:
        cuts.add(cut)
        cut *= 2
    for fraction in (0.25, 0.5, 0.75):
        cuts.add(int(fraction * (n - 1)))
    return sorted(c for c in cuts if 0 <= c <= n - 2)


def check_causal(
    feature_fn: FeatureFn,
    data: Table,
    *,
    cuts: Sequence[int] | None = None,
    perturb: Callable[[Table, np.random.Generator], Table] | None = None,
    seed: int | None = 0,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityReport:
    """Change the future and check that no past feature value moves.

    For each cut (a row position), every row after the cut is changed, the features are
    computed again, and the feature rows up to and including the cut must match the
    original ones. A feature that moves has looked at the future.

    cuts: row positions. By default rows 0, 1, 2, 4, 8, ... and then 25%, 50% and 75%
        of the way through the data (see default_cuts).
    perturb: perturb(future_rows, rng) -> changed rows with the same row labels. By
        default every number moves (see perturb_numbers).

    feature_fn must return a DataFrame or Series whose row labels come from the data, so
    each feature row can be matched to its raw row. It raises ValueError when they do
    not, for example after reset_index, even when the data has the default labels 0, 1,
    2 and so on. The check fails if no past value was compared at all, because a check
    that compared nothing proves nothing.
    """
    _require_table(data)
    if not data.index.is_unique:
        raise ValueError("data has repeated row labels")
    n = len(data)
    if n < 4:
        raise ValueError("need at least 4 rows of data")
    cuts = default_cuts(n) if cuts is None else sorted({int(c) for c in cuts})
    for cut in cuts:
        if not 0 <= cut <= n - 2:
            raise ValueError(f"cut {cut} must be between 0 and {n - 2}")
    change = perturb if perturb is not None else perturb_numbers
    rng = np.random.default_rng(seed)

    full = _labelled_output(feature_fn(data.copy(deep=True)), data)
    _require_kept_labels(feature_fn, data)
    report = ParityReport(check="causal", ok=True)
    report.notes.append("future changed after rows: " + ", ".join(str(c) for c in cuts))
    for cut in cuts:
        past = data.iloc[: cut + 1]
        future = data.iloc[cut + 1 :].copy(deep=True)
        new_future = change(future, rng)
        if not new_future.index.equals(future.index):
            raise ValueError("perturb must return the same rows, with the same row labels")
        changed = pd.concat([past, new_future])
        out = _labelled_output(feature_fn(changed), changed)

        past_labels = data.index[: cut + 1]
        before = full[full.index.isin(past_labels)]
        after = out[out.index.isin(past_labels)]
        result = check_parity(before, after, rtol=rtol, atol=atol)

        report.n_compared += result.n_compared
        report.n_mismatched += result.n_mismatched
        report.max_abs_diff = max(report.max_abs_diff, result.max_abs_diff)
        if result.ok:
            continue
        report.ok = False
        if report.first_mismatch is None:
            report.first_mismatch = result.first_mismatch
        report.notes.append(
            f"after row {cut}: {result.n_mismatched} of {result.n_compared} past values moved"
        )
        report.notes.extend(f"  {note}" for note in result.notes)
    if report.n_compared == 0:
        report.ok = False
        report.notes.append(
            "no past feature value was compared (do all feature rows come after the last cut?)"
        )
    return report


# -------------------------------------------------------------------------- check_warmup


def check_warmup(
    feature_fn: FeatureFn,
    data: Table,
    windows: int | Sequence[int],
    *,
    rows: Sequence[int] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ParityReport:
    """Find how many rows of history a server needs to reproduce the training features.

    For each check row t, the features are computed from only the `window` rows that end
    at t, and the value at t must match the value computed from the full history.
    Windows are tried from the smallest up, and the check stops at the first window that
    matches at every check row. report.min_window holds it (None if no window matched).
    The counts in the report are for the last window tried.

    windows: one window (to assert that a serving window is enough) or a list to search.
    rows: the row positions to check. By default five rows spread over the second half.

    Like check_causal, it raises ValueError when feature_fn does not keep the data's row
    labels.
    """
    _require_table(data)
    if not data.index.is_unique:
        raise ValueError("data has repeated row labels")
    n = len(data)
    if isinstance(windows, (int, np.integer)):
        windows = [int(windows)]
    windows = sorted({int(w) for w in windows})
    if not windows or windows[0] < 1:
        raise ValueError("windows must hold positive row counts")
    if rows is None:
        rows = np.unique(np.linspace(n // 2, n - 1, 5).astype(int)).tolist()
    rows = sorted({int(r) for r in rows})
    if rows[0] < 0 or rows[-1] > n - 1:
        raise ValueError(f"rows must be between 0 and {n - 1}")
    if windows[-1] > rows[0] + 1:
        raise ValueError(
            f"a window of {windows[-1]} rows does not fit before row {rows[0]}; "
            "use smaller windows or later rows"
        )

    full = _labelled_output(feature_fn(data.copy(deep=True)), data)
    _require_kept_labels(feature_fn, data)
    labels = [data.index[t] for t in rows]

    # A row with no feature value even with the full history cannot be served live.
    missing = [label for label in labels if label not in full.index]
    if missing:
        report = ParityReport(check="warmup", ok=False)
        report.first_mismatch = f"no output for row {_fmt(missing[0])}, even with the full history"
        report.notes.append(
            f"{len(missing)} of {len(rows)} check rows have no feature row at all; "
            "a live server could not produce them (does the feature need future rows?)"
        )
        return report

    tried = []
    for window in windows:
        tried.append(window)
        report = _compare_window(feature_fn, data, full, rows, labels, window, rtol, atol)
        if report.ok:
            report.min_window = window
            break

    # The counts and the first difference describe the last window tried: the one that
    # matched, or the largest one if none did.
    report.notes.insert(0, f"tried {len(tried)} window sizes, from {tried[0]} to {tried[-1]} rows")
    if not report.ok:
        report.notes.append(
            f"no window up to {windows[-1]} rows reproduces the full-history values"
        )
    return report


def _compare_window(
    feature_fn: FeatureFn,
    data: Table,
    full: pd.DataFrame,
    rows: list[int],
    labels: list[Any],
    window: int,
    rtol: float,
    atol: float,
) -> ParityReport:
    """Compute the features from the last `window` rows up to each check row, and compare."""
    report = ParityReport(check="warmup", ok=False)
    matched = 0
    for t, label in zip(rows, labels, strict=True):
        recent = data.iloc[t - window + 1 : t + 1].copy(deep=True)
        out = _labelled_output(feature_fn(recent), recent)
        expected = full.loc[[label]]
        if label not in out.index:
            report.n_compared += expected.size
            report.n_mismatched += expected.size
            if report.first_mismatch is None:
                report.first_mismatch = f"window {window}: no output for row {_fmt(label)}"
            continue
        result = check_parity(expected, out.loc[[label]], rtol=rtol, atol=atol)
        report.n_compared += result.n_compared
        report.n_mismatched += result.n_mismatched
        report.max_abs_diff = max(report.max_abs_diff, result.max_abs_diff)
        if result.ok:
            matched += 1
        elif report.first_mismatch is None:
            report.first_mismatch = f"window {window}: {result.first_mismatch}"
    report.ok = matched == len(rows)
    report.notes.append(f"window {window}: {matched} of {len(rows)} check rows match")
    return report


# ------------------------------------------------------------------------------- helpers


def _require_table(data: Any) -> None:
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError(f"data must be a pandas DataFrame or Series, got {type(data).__name__}")


def _labelled_output(obj: Any, data: Table) -> pd.DataFrame:
    """feature_fn output as a DataFrame, after checking that its rows are labelled like data.

    The checks match feature rows to raw rows by label. Output with new labels (for
    example after reset_index, or a plain numpy array) would match nothing, or the wrong
    rows. New labels that the input also has get past this; see _require_kept_labels.
    """
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError(
            "feature_fn must return a DataFrame or Series with the data's row labels, "
            f"got {type(obj).__name__}"
        )
    frame = _as_frame(obj, "feature_fn output")
    unknown = frame.index[~frame.index.isin(data.index)]
    if len(unknown):
        raise ValueError(
            f"feature_fn returned {len(unknown)} rows whose labels are not in its input "
            f"(first: {_fmt(unknown[0])}; the input's labels run from {_fmt(data.index[0])} "
            f"to {_fmt(data.index[-1])}); keep the input's row labels (reset_index, for "
            "example, replaces them)"
        )
    return frame


def _require_kept_labels(feature_fn: FeatureFn, data: Table) -> None:
    """Raise when feature_fn relabels its rows, even with labels that data also has.

    _labelled_output only catches labels that are not in the input. With the pandas
    default labels 0, 1, 2, ..., output after reset_index is labelled 0, 1, 2, ... too,
    so every label is in the input, but each one points at the wrong row and a causal
    feature looks leaky. Given the data without its first row, the input's labels start
    at 1, and relabelled output, which starts at 0, is caught.
    """
    if len(data) < 2:
        return
    rest = data.iloc[1:].copy(deep=True)
    _labelled_output(feature_fn(rest), rest)


def _as_frame(obj: Any, what: str) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name if obj.name is not None else 0)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return pd.DataFrame({0: obj})
        if obj.ndim == 2:
            return pd.DataFrame(obj)
    raise TypeError(
        f"{what} must be a DataFrame, a Series or a numpy array, got {type(obj).__name__}"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)
