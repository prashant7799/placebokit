"""Assertions for tests and CI: fail the build when a model does not beat its placebos.

These raise AssertionError, so pytest shows them as normal test failures. They do not
need pytest itself.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .parity import ParityReport
from .report import ControlReport, _fixed, _p_places


def assert_beats_placebos(
    report: ControlReport,
    alpha: float = 0.05,
    controls: str | Sequence[str] | None = None,
) -> None:
    """Fail unless the real model beat every listed control: p-value <= alpha.

    controls: the control names to check (as shown in the report), or one name. Default:
        all of them.

    It raises ValueError, not AssertionError, when a control has too few runs to ever
    reach alpha. With B runs the smallest p-value is 1 / (B + 1), so for alpha = 0.05
    you need B >= 19. A test that can never pass is a broken test, not a failed model.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    if controls is None:
        names = report.names
    elif isinstance(controls, str):
        names = [controls]  # one name, not a list of its letters
    else:
        names = list(controls)
    if not names:
        raise ValueError("no controls to check, so the assertion would pass without testing")
    results = [report[name] for name in names]

    for r in results:
        if r.min_p_value > alpha:
            needed = math.ceil(1 / alpha - 1)
            # Give the exact fraction: at 3 decimals 1/99 prints as 0.010, which looks
            # like it reaches alpha = 0.01.
            raise ValueError(
                f"{r.name} has {r.runs} runs, so its p-value is at least 1/{r.runs + 1} = "
                f"{r.min_p_value:.3g} and can never be <= {alpha}; it needs at least "
                f"{needed} runs"
            )

    failed = [r for r in results if r.p_value > alpha]
    if failed:
        lines = [
            f"the model (score {report.real_score:.4g}) did not beat "
            f"{len(failed)} of {len(results)} placebos at alpha={alpha}:"
        ]
        for r in failed:
            lines.append(
                f"  {r.name}: p-value {_fixed(r.p_value, _p_places(r.runs))}, "
                f"placebo mean {r.mean:.4g}, best placebo {r.best:.4g}"
            )
        raise AssertionError("\n".join(lines))


def assert_parity(report: ParityReport) -> None:
    """Fail when a parity check (check_parity, check_causal, ...) did not pass."""
    if not report.ok:
        raise AssertionError(str(report))
