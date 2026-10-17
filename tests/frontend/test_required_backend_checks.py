# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""The CI guard must fail when maintained backend checks disappear or skip."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

checker = Path(__file__).resolve().parents[2] / "maintenance" / "check_required_tests.py"
spec = importlib.util.spec_from_file_location("required_backend_checks", checker)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
verify = module.verify

pytestmark = pytest.mark.frontend


@pytest.fixture(autouse=True)
def isolate_unbatched_contracts(monkeypatch):
    """Each synthetic report declares the additional contracts it exercises."""
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", set())


@pytest.fixture
def report(tmp_path, monkeypatch):
    """A bosonic job includes both shared beam-splitter and heterodyne checks."""
    root = ET.Element("testsuite")
    contracts = (
        ("backend.test_beamsplitter_operation.TestModeSubsets", "test_beamsplitter_on_mode_subset"),
        ("backend.test_heterodyne.TestHeterodyne", "test_mode_reset_vacuum"),
    )
    monkeypatch.setattr(module, "REQUIRED", {key: {"bosonic"} for key in contracts})
    for classname, function in contracts:
        for pure in ("True", "False"):
            ET.SubElement(
                root, "testcase", classname=classname, name=f"{function}[BosonicBackend-{pure}]"
            )

    def save():
        path = tmp_path / "tests.xml"
        ET.ElementTree(root).write(path)
        return path

    return root, save


def test_required_checks_pass(report):
    _, save = report
    assert verify(save(), "bosonic") == 4


def test_missing_state_configuration_fails(report):
    root, save = report
    root.remove(root[0])
    with pytest.raises(ValueError, match="Missing:.*pure=True"):
        verify(save(), "bosonic")


def test_empty_report_fails(report):
    root, save = report
    root.clear()
    with pytest.raises(ValueError, match="Missing:"):
        verify(save(), "bosonic")


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_nonpassing_required_case_fails(report, outcome):
    root, save = report
    ET.SubElement(root[0], outcome)
    with pytest.raises(ValueError, match="Did not pass:"):
        verify(save(), "bosonic")


def test_one_skipped_parameter_cannot_hide_behind_passes(report):
    root, save = report
    case = ET.SubElement(
        root,
        "testcase",
        classname=root[0].get("classname"),
        name=root[0].get("name")[:-1] + "-extra]",
    )
    ET.SubElement(case, "skipped")
    with pytest.raises(ValueError, match="Did not pass:"):
        verify(save(), "bosonic")


def test_unrelated_unsupported_case_can_skip(report):
    root, save = report
    case = ET.SubElement(
        root, "testcase", classname="backend.unsupported", name="test_feature[FockBackend-True]"
    )
    ET.SubElement(case, "skipped")
    assert verify(save(), "bosonic") == 4


def test_cutoff_parameter_before_backend_is_recognized(report):
    root, save = report
    for case in root:
        case.set("name", case.get("name").replace("[", "[12-"))
    assert verify(save(), "bosonic") == 4


@pytest.mark.parametrize("configuration,state", [("pure", "True"), ("mixed", "False")])
def test_batched_profile_requires_only_its_state(tmp_path, monkeypatch, configuration, state):
    classname = "backend.test_mzgate_operation.TestFockRepresentation"
    function = "test_mzgate_two_photon_input"
    monkeypatch.setattr(module, "REQUIRED", {(classname, function): {"tf"}})
    root = ET.Element("testsuite")
    case = ET.SubElement(
        root, "testcase", classname=classname, name=f"{function}[TFBackend-{state}]"
    )
    path = tmp_path / "tests.xml"
    ET.ElementTree(root).write(path)
    assert verify(path, f"tensorflow-batched-{configuration}") == 1
    case.set("name", f"{function}[TFBackend-{'False' if state == 'True' else 'True'}]")
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Missing:"):
        verify(path, f"tensorflow-batched-{configuration}")


@pytest.fixture
def unbatched_report(tmp_path, monkeypatch):
    """Multiple parameters must run for both TensorFlow storage modes."""
    classname = "backend.test_twomode_squeezing_operation.TestTwomodeSqueezing"
    function = "test_two_mode_squeezing"
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {(classname, function): {"tf": 2}})
    root = ET.Element("testsuite")
    for pure in ("True", "False"):
        for parameter in (0, 1):
            ET.SubElement(
                root,
                "testcase",
                classname=classname,
                name=f"{function}[TFBackend-{pure}-{parameter}]",
            )

    def save():
        path = tmp_path / "tests.xml"
        ET.ElementTree(root).write(path)
        return path

    return root, save


def test_all_unbatched_parameters_required(unbatched_report):
    _, save = unbatched_report
    assert verify(save(), "all-backends") == 4


def test_missing_unbatched_parameter_fails(unbatched_report):
    root, save = unbatched_report
    root.remove(root[0])
    with pytest.raises(ValueError, match="Expected 2 cases.*found 1"):
        verify(save(), "all-backends")


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_nonpassing_unbatched_parameter_fails(unbatched_report, outcome):
    root, save = unbatched_report
    ET.SubElement(root[0], outcome)
    with pytest.raises(ValueError, match="Did not pass:"):
        verify(save(), "all-backends")


@pytest.mark.parametrize("configuration", ["tensorflow-batched-pure", "tensorflow-batched-mixed"])
def test_unbatched_contracts_allow_batch_exclusions(unbatched_report, configuration):
    root, save = unbatched_report
    for case in root:
        ET.SubElement(case, "skipped")
    assert verify(save(), configuration) == 0


def test_unbatched_contracts_respect_optional_backend(unbatched_report):
    root, save = unbatched_report
    root.clear()
    assert verify(save(), "non-tensorflow") == 0


@pytest.mark.parametrize("separate_suites", [False, True])
def test_duplicate_cannot_replace_missing_parameter(unbatched_report, separate_suites):
    root, save = unbatched_report
    root.remove(root[0])
    duplicate = deepcopy(root[0])
    if separate_suites:
        cases = list(root)
        root.clear()
        root.tag = "testsuites"
        ET.SubElement(root, "testsuite", name="first").extend(cases)
        ET.SubElement(root, "testsuite", name="second").append(duplicate)
    else:
        root.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate test case:"):
        verify(save(), "all-backends")


def test_duplicate_required_case_without_parameter_count_fails(report):
    root, save = report
    root.append(deepcopy(root[0]))
    with pytest.raises(ValueError, match="Duplicate test case:"):
        verify(save(), "bosonic")


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
@pytest.mark.parametrize("nonpassing_first", [False, True])
def test_duplicate_retains_nonpassing_result(unbatched_report, outcome, nonpassing_first):
    root, save = unbatched_report
    root.remove(root[1])
    duplicate = deepcopy(root[0])
    root.append(duplicate)
    ET.SubElement(root[0] if nonpassing_first else duplicate, outcome)
    with pytest.raises(ValueError, match="Duplicate test case:") as error:
        verify(save(), "all-backends")
    assert "Did not pass:" in str(error.value)


def test_same_name_in_different_classes_is_not_duplicate(report, monkeypatch):
    root, save = report
    contracts = {}
    for case in root:
        case.set("name", "test_shared[" + case.get("name").split("[", 1)[1])
        contracts[(case.get("classname"), "test_shared")] = {"bosonic"}
    monkeypatch.setattr(module, "REQUIRED", contracts)
    assert verify(save(), "bosonic") == 4


@pytest.mark.parametrize("configuration,state", [("pure", "True"), ("mixed", "False")])
def test_batched_profile_rejects_duplicate_cases(tmp_path, monkeypatch, configuration, state):
    classname = "backend.test_mzgate_operation.TestFockRepresentation"
    function = "test_mzgate_two_photon_input"
    monkeypatch.setattr(module, "REQUIRED", {(classname, function): {"tf"}})
    root = ET.Element("testsuite")
    case = ET.SubElement(
        root, "testcase", classname=classname, name=f"{function}[TFBackend-{state}]"
    )
    root.append(deepcopy(case))
    path = tmp_path / "tests.xml"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Duplicate test case:"):
        verify(path, f"tensorflow-batched-{configuration}")


@pytest.fixture(
    params=["test_squeeze_variance_frontend", "test_x_displacement", "test_z_displacement"]
)
def expanded_report(tmp_path, monkeypatch, request):
    classname = "integration.test_hbar_integration.TestIntegration"
    function = request.param
    contract = (classname, function)
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {contract: {"tf": 2}})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    root = ET.Element("testsuite")
    for unit in (0.5, 1.0):
        case = ET.SubElement(
            root, "testcase", classname=classname, name=f"{function}[tf-True-{unit}]"
        )
        properties = ET.SubElement(case, "properties")
        for name, value in (
            ("sf_hbar", str(unit)),
            ("requested_hbar", str(unit)),
            ("cutoff", "6"),
            ("batch_size", "2"),
        ):
            ET.SubElement(properties, "property", name=name, value=value)

    def save():
        path = tmp_path / "expanded.xml"
        ET.ElementTree(root).write(path)
        return path

    return root, save


def test_expanded_batch_checks_require_effective_configuration(expanded_report):
    _, save = expanded_report
    assert verify(save(), "tensorflow-batched-pure") == 2


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_expanded_batch_checks_cannot_skip(expanded_report, outcome):
    root, save = expanded_report
    ET.SubElement(root[0], outcome)
    with pytest.raises(ValueError, match="Did not pass:"):
        verify(save(), "tensorflow-batched-pure")


def test_expanded_batch_checks_require_every_parameter(expanded_report):
    root, save = expanded_report
    root.remove(root[0])
    with pytest.raises(ValueError, match="Expected 2 cases.*found 1"):
        verify(save(), "tensorflow-batched-pure")


@pytest.mark.parametrize(
    "property_name,bad_value",
    [
        ("sf_hbar", "1.7"),
        ("sf_hbar", "0.5"),
        ("cutoff", "4"),
        ("batch_size", "None"),
        ("batch_size", "3"),
    ],
)
def test_mislabelled_configuration_is_rejected(expanded_report, property_name, bad_value):
    root, save = expanded_report
    for case in root:
        case.find(f"./properties/property[@name='{property_name}']").set("value", bad_value)
    with pytest.raises(ValueError, match="effective (units|configuration)"):
        verify(save(), "tensorflow-batched-pure")


def test_configuration_evidence_cannot_be_missing(expanded_report):
    root, save = expanded_report
    root[0].remove(root[0].find("properties"))
    with pytest.raises(ValueError, match="effective configuration"):
        verify(save(), "tensorflow-batched-pure")


@pytest.fixture
def reduction_report(tmp_path, monkeypatch):
    """Keep six independent circuit parameters for each of three requested units."""
    classname = "backend.test_state_reduction.TestStateReduction"
    function = "test_entangled_marginal_is_thermal"
    contract = (classname, function)
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {contract: {"tf": 18}})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    root = ET.Element("testsuite")
    for pure in ("True", "False"):
        for unit in (1.0, 1.7, 2.0):
            for parameter in range(6):
                case = ET.SubElement(
                    root,
                    "testcase",
                    classname=classname,
                    name=f"{function}[tf-{pure}-{unit}-{parameter}]",
                )
                properties = ET.SubElement(case, "properties")
                for name, value in (
                    ("sf_hbar", str(unit)),
                    ("requested_hbar", str(unit)),
                    ("cutoff", "6"),
                    ("batch_size", "None"),
                ):
                    ET.SubElement(properties, "property", name=name, value=value)

    def save(configuration="all-backends"):
        for case in root:
            case.find("./properties/property[@name='batch_size']").set(
                "value", "2" if configuration.startswith("tensorflow-batched-") else "None"
            )
        path = tmp_path / "reduction.xml"
        ET.ElementTree(root).write(path)
        return path

    return root, save


@pytest.mark.parametrize(
    "configuration,expected",
    [("all-backends", 36), ("tensorflow-batched-pure", 18), ("tensorflow-batched-mixed", 18)],
)
def test_state_reduction_requires_all_units(reduction_report, configuration, expected):
    _, save = reduction_report
    assert verify(save(configuration), configuration) == expected


@pytest.mark.parametrize("change_requested", [False, True])
def test_state_reduction_rejects_collapsed_units(reduction_report, change_requested):
    root, save = reduction_report
    for case in root:
        case.find("./properties/property[@name='sf_hbar']").set("value", "1.7")
        if change_requested:
            case.find("./properties/property[@name='requested_hbar']").set("value", "1.7")
    with pytest.raises(ValueError, match="effective units"):
        verify(save(), "all-backends")


def test_state_reduction_rejects_swapped_units(reduction_report):
    """The correct total per unit cannot conceal circuits run at the wrong units."""
    root, save = reduction_report
    root[0].find("./properties/property[@name='sf_hbar']").set("value", "1.7")
    root[6].find("./properties/property[@name='sf_hbar']").set("value", "1.0")
    with pytest.raises(ValueError, match="effective configuration"):
        verify(save(), "all-backends")


def test_state_reduction_rejects_uneven_unit_counts(reduction_report):
    """All units and the total count remain present, but one circuit/unit is missing."""
    root, save = reduction_report
    for name in ("sf_hbar", "requested_hbar"):
        root[0].find(f"./properties/property[@name='{name}']").set("value", "1.7")
    with pytest.raises(ValueError, match="effective units counts"):
        verify(save(), "all-backends")


def test_state_reduction_requires_requested_unit_evidence(reduction_report):
    root, save = reduction_report
    properties = root[0].find("properties")
    properties.remove(properties.find("property[@name='requested_hbar']"))
    with pytest.raises(ValueError, match="effective configuration"):
        verify(save(), "all-backends")
