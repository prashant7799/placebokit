"""Optional pytest plugin: choose the number of placebo runs from the command line.

Turn it on in your top-level conftest.py:

    pytest_plugins = ["placebokit.pytest_plugin"]

or run pytest with -p placebokit.pytest_plugin. Then a test can ask for the fixtures
placebo_b and placebo_seed, and CI can run a quick check on every pull request and a
larger one every night:

    pytest                     # B = 99
    pytest --placebo-b=999     # B = 999
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("placebokit")
    group.addoption(
        "--placebo-b",
        type=int,
        default=99,
        help="placebo runs per control for placebokit tests (default 99)",
    )
    group.addoption(
        "--placebo-seed",
        type=int,
        default=0,
        help="random seed for placebokit tests (default 0)",
    )


@pytest.fixture
def placebo_b(request: pytest.FixtureRequest) -> int:
    """Placebo runs per control, from --placebo-b."""
    value = int(request.config.getoption("--placebo-b"))
    if value < 1:
        raise pytest.UsageError("--placebo-b must be at least 1")
    return value


@pytest.fixture
def placebo_seed(request: pytest.FixtureRequest) -> int:
    """Random seed, from --placebo-seed."""
    return int(request.config.getoption("--placebo-seed"))
