"""The empirical p-value used by every control."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def empirical_p_value(
    real_score: float,
    placebo_scores: Sequence[float] | np.ndarray,
    higher_is_better: bool = True,
) -> float:
    """How often a placebo scored at least as well as the real model, as (k + 1) / (B + 1).

    k counts the placebo runs that matched or beat the real score, and B is the number of
    placebo runs. The +1 on top and bottom counts the real run as one of the possible
    outcomes. So the p-value is never 0, and with B runs the smallest possible value is
    1 / (B + 1) (Phipson and Smyth, 2010).

    Two choices keep the p-value on the safe side:

    - A tie counts against the real model.
    - A placebo run whose score is NaN (for example a score that is undefined for constant
      predictions) also counts against the real model.
    """
    placebo = np.asarray(placebo_scores, dtype=float).ravel()
    if placebo.size == 0:
        raise ValueError("need at least one placebo score")
    real = float(real_score)
    if np.isnan(real):
        raise ValueError("the real score is NaN, so there is nothing to compare against")
    if higher_is_better:
        at_least_as_good = placebo >= real
    else:
        at_least_as_good = placebo <= real
    at_least_as_good = at_least_as_good | np.isnan(placebo)
    k = int(at_least_as_good.sum())
    return (k + 1) / (placebo.size + 1)
