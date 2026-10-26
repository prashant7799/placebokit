"""Small helpers that let the controls work on numpy arrays, pandas tables and plain lists."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def n_rows(X: Any) -> int:
    """Number of rows in X."""
    if isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
        return int(X.shape[0])
    return len(X)


def take_rows(X: Any, rows: np.ndarray) -> Any:
    """Return the given row positions of X, keeping X's type.

    pandas objects keep their index and column names, so a model that reads columns by
    name still works. Plain lists (for example a list of texts) become lists again.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[rows]
    if isinstance(X, np.ndarray):
        return X[rows]
    return [X[i] for i in rows]


def as_row_positions(rows: Any, total: int, what: str) -> np.ndarray:
    """Turn row positions or a boolean mask into an int array. The order is kept."""
    arr = np.asarray(rows)
    if arr.dtype == bool:
        if arr.shape != (total,):
            raise ValueError(f"{what} is a boolean mask of length {arr.size}, expected {total}")
        arr = np.flatnonzero(arr)
    if arr.size == 0:
        raise ValueError(f"{what} is empty")
    if not np.issubdtype(arr.dtype, np.integer):
        # Floats such as 3.7 would be cut down to 3 without a word, so refuse them.
        raise ValueError(f"{what} must hold integer row positions or a boolean mask")
    arr = arr.astype(np.int64, copy=False).ravel()
    if arr.min() < 0 or arr.max() >= total:
        raise ValueError(f"{what} holds positions outside 0..{total - 1}")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{what} repeats some row positions")
    return arr


def column_positions(X: Any, columns: Sequence[Any]) -> list[int]:
    """Check that the columns exist in X and return their integer positions."""
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("X has repeated column names, so a column name is ambiguous")
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise KeyError(f"columns not found in X: {missing}")
        return [X.columns.get_loc(c) for c in columns]
    if isinstance(X, np.ndarray) and X.ndim == 2:
        width = X.shape[1]
        positions = []
        for c in columns:
            if isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer)):
                raise KeyError(
                    f"X is a numpy array, so columns must be integer positions, got {c!r}"
                )
            if not 0 <= int(c) < width:
                raise KeyError(f"column {c} is outside 0..{width - 1}")
            positions.append(int(c))
        return positions
    raise TypeError("feature ablation needs X as a 2-D numpy array or a pandas DataFrame")


def drop_columns(X: Any, positions: list[int], keep_only: bool) -> Any:
    """Remove the given columns, or remove every other column when keep_only is True."""
    width = X.shape[1]
    if keep_only:
        keep = positions
    else:
        keep = [i for i in range(width) if i not in positions]
    if not keep:
        raise ValueError("ablation would remove every column")
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, keep]
    return X[:, keep]


def shuffle_columns(X: Any, positions: list[int], order: np.ndarray) -> Any:
    """Return a copy of X where the given columns are reordered by `order`, all together.

    The columns move as one group, so the links between them survive. Only their link
    to the rest of the row (and to the label) is broken.
    """
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        for pos in positions:
            # Reorder the column as a Series so its dtype (for example categorical) is kept,
            # then put back the original row labels so nothing is realigned.
            moved = X.iloc[order, pos].set_axis(X.index)
            out.isetitem(pos, moved)
        return out
    out = X.copy()
    out[:, positions] = X[order][:, positions]
    return out
