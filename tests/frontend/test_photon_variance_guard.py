# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""The distinct nonzero-variance matrix must execute in its required profiles."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

checker = Path(__file__).resolve().parents[2] / "maintenance" / "check_required_tests.py"
spec = importlib.util.spec_from_file_location("variance_required_checks", checker)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verify = module.verify
PHOTON_VARIANCE_REQUIREMENT = {"fock": ("A", "B"), "tf": ("A", "B", "distinct-batch")}
pytestmark = pytest.mark.frontend


@pytest.fixture
def photon_variance_report(tmp_path, monkeypatch):
    """An independent explicit matrix, not generated from the guard's catalog."""
    assert module.PHOTON_VARIANCE_REQUIRED == PHOTON_VARIANCE_REQUIREMENT
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {})
    monkeypatch.setattr(module, "NUMBER_STATE_REQUIRED", {})
    monkeypatch.setattr(module, "GAUSSIAN_PREPARATION_REQUIRED", {})

    def make(profile):
        backends = {
            "all-backends": ("fock", "tf"),
            "fock": ("fock",),
            "tensorflow": ("tf",),
            "non-tensorflow": ("fock",),
            "local-without-cloud": ("fock",),
            "tensorflow-batched-pure": ("tf",),
            "tensorflow-batched-mixed": ("tf",),
            "frontend-apps-api-gaussian": (),
            "bosonic": (),
        }[profile]
        states = (True, False)
        if profile.startswith("tensorflow-batched-"):
            states = (profile.endswith("pure"),)
        root = ET.Element("testsuite")
        for backend in backends:
            for pure in states:
                for unit in (1.0, 1.7):
                    for cutoff in (3, 4):
                        inputs = ("A", "B") if backend == "fock" else ("A", "B", "distinct-batch")
                        for label in inputs:
                            case = ET.SubElement(
                                root,
                                "testcase",
                                classname="integration.test_photon_variance",
                                name=f"test_distinct_photon_variance[{pure}-{unit}-{cutoff}-{backend}-{label}]",
                            )
                            props = ET.SubElement(case, "properties")
                            for name, value in (
                                ("effective_storage", "ket" if pure else "density"),
                                ("effective_hbar", str(unit)),
                                ("effective_cutoff", str(cutoff)),
                                (
                                    "effective_batch_size",
                                    "2" if label == "distinct-batch" else "None",
                                ),
                            ):
                                ET.SubElement(props, "property", name=name, value=value)
        path = tmp_path / "photon-variances.xml"

        def save():
            ET.ElementTree(root).write(path)
            return path

        return root, save

    return make


PHOTON_VARIANCE_PROFILES = (
    ("all-backends", 40),
    ("fock", 16),
    ("tensorflow", 24),
    ("non-tensorflow", 16),
    ("local-without-cloud", 16),
    ("tensorflow-batched-pure", 12),
    ("tensorflow-batched-mixed", 12),
    ("frontend-apps-api-gaussian", 0),
    ("bosonic", 0),
)


@pytest.mark.parametrize("profile,count", PHOTON_VARIANCE_PROFILES)
def test_photon_variance_guard_requires_complete_matrix(photon_variance_report, profile, count):
    _, save = photon_variance_report(profile)
    assert verify(save(), profile) == count


@pytest.mark.parametrize("profile,count", PHOTON_VARIANCE_PROFILES[:7])
@pytest.mark.parametrize(
    "corruption", ["missing", "skipped", "failure", "error", "duplicate", "replacement"]
)
def test_photon_variance_guard_rejects_every_missing_or_nonpassing_case(
    photon_variance_report, profile, count, corruption
):
    root, save = photon_variance_report(profile)
    assert verify(save(), profile) == count
    for index in range(len(root)):
        original = deepcopy(root)
        case = root[index]
        if corruption == "missing":
            root.remove(case)
        elif corruption in ("skipped", "failure", "error"):
            ET.SubElement(case, corruption)
        elif corruption == "duplicate":
            root.append(deepcopy(case))
        else:
            duplicate = deepcopy(root[(index + 1) % len(root)])
            duplicate.set("name", duplicate.get("name")[:-1] + "-duplicate]")
            root.remove(case)
            root.append(duplicate)
        with pytest.raises(
            ValueError, match="Expected exactly one|Did not pass|Duplicate test case"
        ):
            verify(save(), profile)
        root[:] = list(original)


@pytest.mark.parametrize("profile,count", PHOTON_VARIANCE_PROFILES[:7])
@pytest.mark.parametrize("corruption", ["missing", "wrong", "duplicate"])
def test_photon_variance_guard_requires_effective_configuration(
    photon_variance_report, profile, count, corruption
):
    root, save = photon_variance_report(profile)
    assert verify(save(), profile) == count
    for case in root:
        props = case.find("properties")
        for index in range(len(props)):
            original = deepcopy(props)
            prop = props[index]
            if corruption == "missing":
                props.remove(prop)
            elif corruption == "wrong":
                prop.set("value", "incorrect")
            else:
                props.append(deepcopy(prop))
            with pytest.raises(
                ValueError, match="Missing or incorrect photon-variance configuration"
            ):
                verify(save(), profile)
            props[:] = list(original)


@pytest.mark.parametrize("profile,count", PHOTON_VARIANCE_PROFILES[:7])
@pytest.mark.parametrize("outcome", [None, "skipped", "failure", "error"])
def test_photon_variance_guard_rejects_unknown_extra_with_complete_matrix(
    photon_variance_report, profile, count, outcome
):
    root, save = photon_variance_report(profile)
    assert verify(save(), profile) == count
    extra = deepcopy(root[0])
    extra.set("name", extra.get("name")[:-1] + "-duplicate]")
    if outcome is not None:
        ET.SubElement(extra, outcome)
    root.append(extra)
    with pytest.raises(ValueError, match="Unknown photon-variance case"):
        verify(save(), profile)


def test_photon_variance_guard_allows_known_unselected_optional_cases(photon_variance_report):
    root, save = photon_variance_report("all-backends")
    for case in root:
        if "-tf-" in case.get("name"):
            ET.SubElement(case, "skipped")
    assert verify(save(), "non-tensorflow") == 16
