"""The README must show real code and real output.

Blocks in README.md marked with <!-- code: path --> must equal that file, and blocks
marked with <!-- output: path --> must equal what that script prints.
"""

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MARKED_BLOCK = re.compile(r"<!-- (code|output): (\S+) -->\n```\w*\n(.*?)```", re.DOTALL)
EXAMPLES = sorted(p for p in (ROOT / "examples").glob("*.py") if not p.name.startswith("test_"))


@cache
def run_script(relative_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / relative_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def marked_blocks(kind):
    found = MARKED_BLOCK.findall(README)
    return [pytest.param(path, body, id=path) for k, path, body in found if k == kind]


def test_readme_has_marked_blocks():
    assert len(marked_blocks("code")) >= 1
    assert len(marked_blocks("output")) >= 6


@pytest.mark.parametrize("path, body", marked_blocks("code"))
def test_readme_code_matches_the_file(path, body):
    assert body == (ROOT / path).read_text(encoding="utf-8")


@pytest.mark.parametrize("path, body", marked_blocks("output"))
def test_readme_output_matches_the_script(path, body):
    assert run_script(path) == body


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_every_example_runs(script):
    assert run_script(str(script.relative_to(ROOT)))


def test_ci_gate_example_passes():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         str(ROOT / "examples" / "test_ci_gate.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
