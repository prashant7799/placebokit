import subprocess
import sys
from importlib import metadata
from pathlib import Path

import placebokit


def test_version_matches_package_metadata():
    assert placebokit.__version__ == metadata.version("placebokit")


def test_package_is_typed():
    assert (Path(placebokit.__file__).parent / "py.typed").exists()


def test_public_names_exist():
    for name in placebokit.__all__:
        assert hasattr(placebokit, name), name


def test_import_does_not_need_pytest():
    # The library must work without pytest; only the optional plugin imports it.
    code = "import sys, placebokit; assert 'pytest' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
