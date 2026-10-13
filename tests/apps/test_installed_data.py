# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for dataset resources in an installed distribution."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.apps

DATASETS = [
    ("Planted", 50000),
    ("TaceAs", 50000),
    ("PHat", 50000),
    ("Mutag0", 20000),
    ("Mutag1", 20000),
    ("Mutag2", 20000),
    ("Mutag3", 20000),
    ("Formic", 20000),
    ("Water", 5000),
    ("Pyrrole", 1000),
    ("MUTAG", 188),
    ("QM9Exact", 1100),
    ("QM9MC", 1100),
]

INSTALLED_PROBE = """
import importlib.abc
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

class NoPkgResources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pkg_resources" or fullname.startswith("pkg_resources."):
            raise ModuleNotFoundError("pkg_resources is unavailable", name=fullname)

sys.meta_path.insert(0, NoPkgResources())
installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))

import numpy as np
import strawberryfields as sf
from strawberryfields.apps import data

package_file = Path(sf.__file__).resolve()
assert package_file.is_relative_to(installed), package_file
assert distribution("StrawberryFields").locate_file("strawberryfields/__init__.py").resolve() == package_file
assert "pkg_resources" not in sys.modules
result = {"version": sf.__version__}
if len(sys.argv) > 2:
    name = sys.argv[2]
    factory = getattr(data, name)
    dataset = factory(0) if name in ("Water", "Pyrrole") else factory()
    first, last = np.asarray(dataset[0]), np.asarray(dataset[-1])
    assert first.size > 0 and first.shape == last.shape
    assert np.all(np.isfinite(first)) and np.all(np.isfinite(last))
    result["items"] = len(dataset)
print(json.dumps(result))
"""


def run_checked(command, cwd):
    """Include subprocess diagnostics in a failed assertion."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def installed_package(tmp_path_factory):
    """Build an sdist and its wheel without writing build files into the checkout."""
    root = Path(__file__).resolve().parents[2]
    work = tmp_path_factory.mktemp("installed package")
    source, wheels, installed, unrelated = [
        work / name for name in ("source", "wheels", "site", "cwd")
    ]
    source.mkdir()
    unrelated.mkdir()
    shutil.copytree(
        root / "strawberryfields",
        source / "strawberryfields",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("setup.py", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(root / name, source / name)

    run_checked([sys.executable, "setup.py", "sdist", "--dist-dir", str(work / "sdist")], source)
    sdists = list((work / "sdist").glob("*.tar.gz"))
    assert len(sdists) == 1
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    run_checked(
        pip
        + [
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-cache-dir",
            "-w",
            str(wheels),
            str(sdists[0]),
        ],
        unrelated,
    )
    artifacts = list(wheels.glob("*.whl"))
    assert len(artifacts) == 1
    run_checked(
        pip
        + ["install", "--no-deps", "--no-compile", "--target", str(installed), str(artifacts[0])],
        unrelated,
    )
    return installed, unrelated


def test_installed_import_without_pkg_resources(installed_package):
    """Import succeeds with the obsolete runtime module unavailable."""
    installed, unrelated = installed_package
    result = run_checked([sys.executable, "-I", "-c", INSTALLED_PROBE, str(installed)], unrelated)
    assert json.loads(result)["version"]


@pytest.mark.parametrize("name,expected_items", DATASETS)
def test_installed_dataset_without_pkg_resources(installed_package, name, expected_items):
    """Every bundled dataset remains readable from the installed wheel."""
    installed, unrelated = installed_package
    result = run_checked(
        [sys.executable, "-I", "-c", INSTALLED_PROBE, str(installed), name], unrelated
    )
    assert json.loads(result)["items"] == expected_items
