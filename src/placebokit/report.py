"""Result types: one ControlResult per control, gathered in a ControlReport."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ControlResult:
    """The placebo scores of one control, compared with the real score."""

    name: str
    real_score: float
    scores: np.ndarray
    p_value: float
    higher_is_better: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def runs(self) -> int:
        """Number of placebo runs."""
        return int(self.scores.size)

    @property
    def n_failed(self) -> int:
        """Placebo runs whose score was NaN. They count against the real model."""
        return int(np.isnan(self.scores).sum())

    @property
    def mean(self) -> float:
        """Average placebo score (NaN runs left out)."""
        return _nan_stat(np.mean, self.scores)

    @property
    def std(self) -> float:
        """Spread of the placebo scores (NaN runs left out)."""
        return _nan_stat(np.std, self.scores)

    @property
    def best(self) -> float:
        """The best placebo score: the highest, or the lowest when lower is better."""
        if self.higher_is_better:
            return _nan_stat(np.max, self.scores)
        return _nan_stat(np.min, self.scores)

    @property
    def min_p_value(self) -> float:
        """The smallest p-value this many runs can give: 1 / (runs + 1)."""
        return 1.0 / (self.runs + 1)

    def beaten(self, alpha: float = 0.05) -> bool:
        """True when the real model beat this placebo: p-value <= alpha."""
        return self.p_value <= alpha


@dataclass
class ControlReport:
    """Everything one call to run_controls found."""

    real_score: float
    results: list[ControlResult]
    B: int
    seed: int | None
    higher_is_better: bool = True

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.results]

    def __getitem__(self, name: str) -> ControlResult:
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(f"no control named {name!r}; the controls are {self.names}")

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def passed(self, alpha: float = 0.05) -> bool:
        """True when the real model beat every control at level alpha."""
        return all(r.beaten(alpha) for r in self.results)

    def to_frame(self, alpha: float = 0.05) -> pd.DataFrame:
        """One row per control: runs, placebo mean, spread, best, p-value and the verdict."""
        rows = []
        for r in self.results:
            rows.append(
                {
                    "control": r.name,
                    "runs": r.runs,
                    "mean": r.mean,
                    "std": r.std,
                    "best": r.best,
                    "p_value": r.p_value,
                    "beaten": r.beaten(alpha),
                }
            )
        return pd.DataFrame(rows)

    def to_text(self, alpha: float = 0.05, digits: int = 3) -> str:
        """A plain-text table. str(report) calls this with the defaults.

        digits: decimals for the scores. p-values get at least as many and at least 3,
        and more when a control has 1,000 runs or more, so the smallest p-value,
        1 / (runs + 1), never prints as 0.
        """
        direction = "higher is better" if self.higher_is_better else "lower is better"
        lines = [
            f"placebokit report (B={self.B}, seed={self.seed}, {direction})",
            f"real score: {_fixed(self.real_score, digits)}",
            "",
        ]
        # One number of decimals for the whole p-value column, so the column lines up.
        p_places = _p_places(max((r.runs for r in self.results), default=0), digits)
        header = ["control", "runs", "mean", "best", "p-value", ""]
        table = [header]
        for r in self.results:
            verdict = "beaten" if r.beaten(alpha) else "NOT beaten"
            if r.n_failed:
                verdict += f" ({r.n_failed} NaN runs)"
            table.append(
                [
                    r.name,
                    str(r.runs),
                    _fixed(r.mean, digits),
                    _fixed(r.best, digits),
                    _fixed(r.p_value, p_places),
                    verdict,
                ]
            )
        widths = [max(len(row[i]) for row in table) for i in range(len(header))]
        for row in table:
            cells = [row[0].ljust(widths[0])]
            cells += [row[i].rjust(widths[i]) for i in range(1, 5)]
            cells.append(row[5])
            lines.append("  ".join(cells).rstrip())
        lines.append("")
        lines.append("p-value = (k + 1) / (runs + 1), k = placebo runs that did at least as well")
        lines.append(f"beaten = p-value <= {alpha}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _nan_stat(fn, values: np.ndarray) -> float:
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return float("nan")
    return float(fn(finite))


def _p_places(runs: int, digits: int = 3) -> int:
    """Decimals that show every p-value this many runs can give, down to 1 / (runs + 1).

    At least `digits` and at least 3. runs has len(str(runs)) digits, so runs + 1 is at
    most 10 ** len(str(runs)), and that many decimals keep 1 / (runs + 1) above zero:
    with 9,999 runs it prints as 0.0001, not 0.000.
    """
    return max(3, digits, len(str(runs)))


def _fixed(value: float, places: int) -> str:
    """value with `places` decimals. A value that rounds to zero prints without a minus."""
    text = f"{value:.{places}f}"
    if text.startswith("-") and not text.strip("-0."):
        return text[1:]  # "-0.000": a tiny negative, shown as 0.000
    return text
