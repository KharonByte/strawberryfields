# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Independent literal matrix controls for the correlated preparation guard."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

pytestmark = pytest.mark.frontend
spec = importlib.util.spec_from_file_location(
    "gaussian_preparation_checker",
    Path(__file__).resolve().parents[2] / "maintenance/check_required_tests.py",
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

PROFILES = {
    "all-backends": (("gaussian", "fock", "tf"), (True, False), None, 24),
    "frontend-apps-api-gaussian": (("gaussian",), (True, False), None, 8),
    "fock": (("fock",), (True, False), None, 8),
    "tensorflow": (("tf",), (True, False), None, 8),
    "tensorflow-batched-pure": (("tf",), (True,), 2, 4),
    "tensorflow-batched-mixed": (("tf",), (False,), 2, 4),
    "non-tensorflow": (("gaussian", "fock"), (True, False), None, 16),
    "local-without-cloud": (("gaussian", "fock"), (True, False), None, 16),
    "bosonic": ((), (), None, 0),
}
CLASSNAME = "integration.test_gaussian_state_decomposition"
FUNCTION = "test_correlated_gaussian_preparation"


@pytest.fixture(autouse=True)
def isolate_old_contracts(monkeypatch):
    for name in ("REQUIRED", "UNBATCHED_REQUIRED", "BATCH_CAPABLE_REQUIRED"):
        monkeypatch.setattr(checker, name, {})


def report(configuration):
    backends, initials, batch, _ = PROFILES[configuration]
    root = ET.Element("testsuite")
    for backend in backends:
        for target in ("pure", "mixed"):
            for initial in initials:
                for units in (1.0, 1.7):
                    suffix = "ordinary" if batch is None else "replicated2"
                    identity = f"{backend}-{target}-initial{initial}-hbar{units}-{suffix}"
                    case = ET.SubElement(
                        root, "testcase", classname=CLASSNAME, name=f"{FUNCTION}[{identity}]"
                    )
                    properties = ET.SubElement(case, "properties")
                    fields = {
                        "setup_hbar": units,
                        "requested_hbar": units,
                        "cutoff": 20,
                        "batch_size": batch,
                        "physical_target": target,
                        "requested_initial_pure": initial,
                        "observed_representation": (
                            "gaussian" if backend == "gaussian" else "density"
                        ),
                        "observed_state_pure": backend == "gaussian" and target == "pure",
                        "observed_dtype": {
                            "gaussian": "float64",
                            "fock": "complex128",
                            "tf": "complex64",
                        }[backend],
                        "observed_num_modes": 2,
                        "case_id": identity,
                    }
                    for name, value in fields.items():
                        ET.SubElement(properties, "property", name=name, value=str(value))
    return root


def verify(root, configuration, tmp_path):
    path = tmp_path / "gaussian.xml"
    ET.ElementTree(root).write(path)
    return checker.verify(path, configuration)


@pytest.mark.parametrize("configuration", PROFILES)
def test_complete_gaussian_matrix(configuration, tmp_path):
    assert verify(report(configuration), configuration, tmp_path) == PROFILES[configuration][3]


@pytest.mark.parametrize("configuration", [key for key in PROFILES if key != "bosonic"])
@pytest.mark.parametrize(
    "corruption", ["missing", "skipped", "failure", "error", "duplicate", "renamed", "unknown"]
)
def test_gaussian_matrix_rejects_bad_rows(configuration, corruption, tmp_path):
    root = report(configuration)
    case = root[0]
    if corruption == "missing":
        root.remove(case)
    elif corruption in ("skipped", "failure", "error"):
        ET.SubElement(case, corruption)
    elif corruption == "duplicate":
        root.append(deepcopy(case))
    elif corruption == "renamed":
        case.set("name", case.get("name").replace("hbar1.0", "hbar9.0"))
    else:
        extra = deepcopy(case)
        extra.set("name", extra.get("name") + "-unknown")
        root.append(extra)
    with pytest.raises(ValueError):
        verify(root, configuration, tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "setup_hbar",
        "requested_hbar",
        "cutoff",
        "batch_size",
        "physical_target",
        "requested_initial_pure",
        "observed_representation",
        "observed_state_pure",
        "observed_dtype",
        "observed_num_modes",
        "case_id",
    ],
)
@pytest.mark.parametrize("corruption", ["missing", "wrong", "duplicate"])
def test_gaussian_metadata_is_required(field, corruption, tmp_path):
    root = report("all-backends")
    properties = root[0].find("properties")
    prop = properties.find(f"property[@name='{field}']")
    if corruption == "missing":
        properties.remove(prop)
    elif corruption == "wrong":
        prop.set("value", "incorrect")
    else:
        properties.append(deepcopy(prop))
    with pytest.raises(ValueError, match=field):
        verify(root, "all-backends", tmp_path)


def test_initial_pure_finite_output_must_be_density(tmp_path):
    root = report("tensorflow-batched-pure")
    root[0].find("./properties/property[@name='observed_state_pure']").set("value", "True")
    with pytest.raises(ValueError, match="observed_state_pure"):
        verify(root, "tensorflow-batched-pure", tmp_path)


def test_unrelated_exclusion_is_preserved(tmp_path):
    root = report("all-backends")
    case = ET.SubElement(
        root, "testcase", classname="existing.unsupported", name="unbatched_contract"
    )
    ET.SubElement(case, "skipped")
    assert verify(root, "all-backends", tmp_path) == 24
