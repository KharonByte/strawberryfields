# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for dataset resources in an installed distribution."""

from copy import deepcopy
from email.parser import Parser
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import sysconfig

import pytest

# The maintenance script is not part of the installed distribution.
check_metadata = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "maintenance" / "check_metadata.py")
)["check_metadata"]

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


def run_checked(command, cwd, env=None):
    """Include subprocess diagnostics in a failed assertion."""
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
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


def test_installed_local_simulation_without_cloud_client(installed_package):
    """Wheel metadata and all three CPU simulators work without importing XCC."""
    installed, unrelated = installed_package
    probe = """
import contextlib
import importlib.abc
from importlib.metadata import distribution
import io
import json
from pathlib import Path
import sys
from packaging.requirements import Requirement

attempts = []
class NoCloudClient(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xcc" or fullname.startswith("xcc."):
            attempts.append(fullname)
            raise ModuleNotFoundError("cloud client is unavailable", name=fullname)

sys.meta_path.insert(0, NoCloudClient())
installed = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(installed))
import strawberryfields as sf
assert Path(sf.__file__).resolve().is_relative_to(installed)

requirements = [Requirement(r) for r in distribution("StrawberryFields").requires]
base = {r.name: r for r in requirements if not r.marker or r.marker.evaluate({"extra": ""})}
assert "xanadu-cloud-client" not in base
assert str(base["quantum-blackbird"].specifier) == "==0.5.0"
assert str(base["quantum-xir"].specifier) == "==0.2.2"
cloud = {r.name: r for r in requirements if r.marker and r.marker.evaluate({"extra": "cloud"})}
assert str(cloud["xanadu-cloud-client"].specifier) == "==0.3.2"

with contextlib.redirect_stdout(io.StringIO()):
    sf.about()
means = {}
for backend in ("gaussian", "fock", "bosonic"):
    program = sf.Program(1)
    with program.context as q:
        sf.ops.Dgate(0.2) | q[0]
        sf.ops.LossChannel(0.8) | q[0]
    state = sf.Engine(backend, backend_options={"cutoff_dim": 10}).run(program).state
    means[backend] = float(state.mean_photon(0)[0])
    assert abs(means[backend] - 0.032) < 1e-8, means
assert not attempts, attempts
assert "xcc" not in sys.modules
try:
    sf.RemoteEngine("X8")
except ImportError as error:
    assert ".[cloud]" in str(error), str(error)
else:
    raise AssertionError("RemoteEngine must explain how to install its extra")
assert attempts == ["xcc"], attempts
print(json.dumps(means))
"""
    result = run_checked([sys.executable, "-I", "-c", probe, str(installed)], unrelated)
    assert set(json.loads(result)) == {"gaussian", "fock", "bosonic"}


@pytest.mark.parametrize("launcher", ["console", "module"])
def test_pytest_launchers_collect_installed_package_tests(installed_package, launcher):
    """Both launchers collect tests without adding the checkout to PYTHONPATH."""
    installed, _ = installed_package
    root = Path(__file__).resolve().parents[2]
    if launcher == "console":
        executable = "pytest.exe" if sys.platform == "win32" else "pytest"
        command = [str(Path(sysconfig.get_path("scripts")) / executable)]
    else:
        command = [sys.executable, "-m", "pytest"]
    # Provide the installed wheel, not the source root, to the console launcher.
    env = dict(os.environ, PYTHONPATH=str(installed))
    output = run_checked(
        command + ["--collect-only", "-q", "-p", "no:cacheprovider", str(Path(__file__).resolve())],
        root,
        env=env,
    )
    assert "test_installed_metadata_matches_documentation" in output


def installed_metadata(installed):
    """Read wheel metadata after the source distribution has been built and installed."""
    paths = list(installed.glob("*.dist-info/METADATA"))
    assert len(paths) == 1
    return Parser().parsestr(paths[0].read_text(encoding="utf-8"))


def test_installed_metadata_matches_documentation(installed_package):
    """Installation constraints, attribution and documentation agree with the CI pins."""
    installed, unrelated = installed_package
    root = Path(__file__).resolve().parents[2]
    problems = check_metadata(root, installed_metadata(installed))
    assert not problems, "\n".join(problems)
    metadata_path = next(installed.glob("*.dist-info/METADATA"))
    run_checked(
        [sys.executable, str(root / "maintenance/check_metadata.py"), str(metadata_path)], unrelated
    )


@pytest.mark.parametrize(
    "field,value,diagnostic",
    [
        ("Requires-Python", None, "Requires-Python"),
        ("Requires-Python", ">=3.8", "Requires-Python"),
        ("Classifier", "Programming Language :: Python :: 3.10", "classifiers"),
        ("Maintainer", "Xanadu Inc.", "Maintainer"),
        ("Home-page", "https://github.com/XanaduAI/strawberryfields", "Home-page"),
        ("Project-URL", "Source, https://github.com/XanaduAI/strawberryfields", "Project-URL"),
    ],
)
def test_metadata_guard_rejects_stale_wheel_metadata(installed_package, field, value, diagnostic):
    """Reintroducing the old wheel declarations must fail the guard."""
    metadata = deepcopy(installed_metadata(installed_package[0]))
    del metadata[field]
    if value is not None:
        metadata[field] = value
    problems = check_metadata(Path(__file__).resolve().parents[2], metadata)
    assert any(diagnostic in problem for problem in problems), problems


@pytest.mark.parametrize(
    "path,before,after,diagnostic",
    [
        ("README.md", ">=3.12,<3.13", ">=3.8,<3.11", "README Python requirement"),
        ("README.md", "`3.12.3`", "`3.10.8`", "CI Python"),
        (
            "README.md",
            "python -m pip install --no-deps --no-build-isolation .",
            "pip install strawberryfields",
            "upstream PyPI",
        ),
        ("SUPPORT.md", "Python 3.12.3", "Python 3.10.8", "SUPPORT.md"),
        (".github/workflows/tests.yaml", "'3.12.3'", "'3.13.0'", "tests.yaml: CI Python"),
    ],
)
def test_metadata_guard_rejects_documentation_drift(
    installed_package, tmp_path, path, before, after, diagnostic
):
    """Changing documentation or a CI pin independently must be caught."""
    root = Path(__file__).resolve().parents[2]
    for name in ("README.md", "SUPPORT.md"):
        shutil.copy2(root / name, tmp_path / name)
    shutil.copytree(root / ".github/workflows", tmp_path / ".github/workflows")
    target = tmp_path / path
    original = target.read_text(encoding="utf-8")
    assert before in original
    target.write_text(original.replace(before, after), encoding="utf-8")
    problems = check_metadata(tmp_path, installed_metadata(installed_package[0]))
    assert any(diagnostic in problem for problem in problems), problems


def test_readme_example_from_installed_wheel(installed_package):
    """Run the published example outside the checkout against the installed artifact."""
    installed, unrelated = installed_package
    root = Path(__file__).resolve().parents[2]
    readme = (root / "README.md").read_text(encoding="utf-8")
    examples = re.findall(r"```python\n(.*?)\n```", readme, re.DOTALL)
    assert len(examples) == 1
    script = (
        "import sys\nfrom pathlib import Path\n"
        "installed = Path(sys.argv[1]).resolve()\nsys.path.insert(0, str(installed))\n"
        + examples[0]
        + "\nassert Path(sf.__file__).resolve().is_relative_to(installed)\n"
    )
    result = run_checked([sys.executable, "-I", "-c", script, str(installed)], unrelated)
    assert float(result.strip()) == pytest.approx(0.25)


@pytest.mark.parametrize("name,expected_items", DATASETS)
def test_installed_dataset_without_pkg_resources(installed_package, name, expected_items):
    """Every bundled dataset remains readable from the installed wheel."""
    installed, unrelated = installed_package
    result = run_checked(
        [sys.executable, "-I", "-c", INSTALLED_PROBE, str(installed), name], unrelated
    )
    assert json.loads(result)["items"] == expected_items
