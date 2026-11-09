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
PARITY_REQUIREMENTS = {
    key: deepcopy(module.UNBATCHED_REQUIRED[key])
    for key in (
        ("backend.test_squeeze_operation.TestFockRepresentation", "test_no_odd_fock"),
        ("backend.test_squeezed_state_preparation.TestFockRepresentation", "test_no_odd_fock"),
    )
}
ADAPTER_CASES = [
    (
        ("integration.test_utils_integration.TestInitialStatesAgreeFock", function),
        backend,
        "all-backends",
    )
    for function in ("test_vacuum", "test_coherent", "test_squeezed", "test_displaced_squeezed")
    for backend in ("gaussian", "bosonic")
]
ADAPTER_CASES += [
    (
        ("backend.test_modes.TestFockRepresentation", "test_normalized_add_mode"),
        "gaussian",
        "all-backends",
    )
]
ADAPTER_CASES += [
    (("backend.test_states.TestBosonicState", "test_density_matrix"), backend, profile)
    for backend, profile in (
        ("fock", "all-backends"),
        ("tf", "all-backends"),
        ("tf", "tensorflow-batched-pure"),
        ("tf", "tensorflow-batched-mixed"),
    )
]
ADAPTER_CASES += [
    (("integration.test_utils_integration." + classname, function + suffix), "tf", profile)
    for classname, suffix in (
        ("TestExtractChannelOneMode", ""),
        ("TestExtractChannelTwoMode", ""),
        ("TestExtractChannelTwoMode", "_vectorize"),
    )
    for function in (
        "test_extract_choi_channel",
        "test_extract_liouville_channel",
        "test_extract_kraus_channel",
    )
    for profile in ("tensorflow-batched-pure", "tensorflow-batched-mixed")
]
ADAPTER_CASES += [
    (("integration.test_decompositions_integration." + classname, function), backend, profile)
    for classname, functions in (
        (
            "TestGaussianBackendDecomposeState",
            ("test_vacuum", "test_squeezed", "test_displaced_squeezed", "test_rotated_squeezed"),
        ),
        ("TestFockBackendDecomposeState", ("test_squeezed",)),
    )
    for function in functions
    for backend, profile in (
        ("gaussian", "all-backends"),
        ("fock", "all-backends"),
        ("tf", "all-backends"),
        ("tf", "tensorflow-batched-pure"),
        ("tf", "tensorflow-batched-mixed"),
    )
]
OPTICS_CUTOFFS = {
    ("integration.test_decompositions_integration." + classname, function): cutoff
    for classname, functions, cutoff in (
        (
            "TestGaussianBackendDecompositions",
            (
                "test_passive_gaussian_transform",
                "test_interferometer",
                "test_interferometer_rectangular_compact",
                "test_interferometer_triangular_compact",
                "test_interferometer_sun_compact",
            ),
            16,
        ),
        ("TestGaussianBackendDecomposeState", ("test_thermal",), 12),
        ("TestFockBackendDecomposeState", ("test_rotated_squeezed",), 6),
    )
    for function in functions
}
ADAPTER_CASES += [
    (contract, backend, profile)
    for contract in OPTICS_CUTOFFS
    for backend, profile in (
        ("gaussian", "all-backends"),
        ("fock", "all-backends"),
        ("tf", "all-backends"),
        ("tf", "tensorflow-batched-pure"),
        ("tf", "tensorflow-batched-mixed"),
    )
]
ADAPTER_CATALOG = {
    contract: (
        deepcopy(module.UNBATCHED_REQUIRED.get(contract, {})),
        deepcopy(module.BATCH_CAPABLE_REQUIRED.get(contract, {})),
    )
    for contract, _, _ in ADAPTER_CASES
}
GRADIENT_CLASS = "integration.test_tf_integration.TestGradient"
GRADIENT_FUNCTIONS = (
    "test_displacement_mean_photon_gradient",
    "test_displaced_thermal_mean_photon_gradient",
    "test_coherent_ket_gradient",
    "test_coherent_dm_gradient",
    "test_displaced_squeezed_mean_photon_gradient",
    "test_BS_state_gradients",
    "test_MZ_state_gradients",
    "test_2mode_squeezed_vacuum_gradients",
)
GRADIENT_CATALOG = {
    (GRADIENT_CLASS, function): deepcopy(module.BATCH_CAPABLE_REQUIRED[(GRADIENT_CLASS, function)])
    for function in GRADIENT_FUNCTIONS
}

PASSIVE_CONTROL = ("integration.test_decomposition_controls", "test_distinct_passive_inputs")
PASSIVE_REQUIREMENT = deepcopy(module.BATCH_CAPABLE_REQUIRED[PASSIVE_CONTROL])
PASSIVE_PARAMETERS = module.REQUIRED_PARAMETERS[PASSIVE_CONTROL]
INITIAL_CLASS = "integration.test_utils_integration.TestInitialStatesAgreeGaussian"
INITIAL_CASES = {
    "test_vacuum": ((), 1, 6),
    "test_coherent": (("a",), 4, 6),
    "test_squeezed": (("r", "phi"), 4, 8),
    "test_displaced_squeezed": (("a", "r", "phi"), 4, 8),
}
INITIAL_REQUIREMENTS = {
    (INITIAL_CLASS, function): deepcopy(module.BATCH_CAPABLE_REQUIRED[(INITIAL_CLASS, function)])
    for function in INITIAL_CASES
}
INITIAL_PARAMETERS = {key: module.REQUIRED_PARAMETERS[key] for key in INITIAL_REQUIREMENTS}
GRAPH_EMBEDDING = (
    "integration.test_measurement_integration.TestPostselection",
    "test_embed_graph",
)
GRAPH_REQUIREMENT = deepcopy(module.UNBATCHED_REQUIRED[GRAPH_EMBEDDING])
GRAPH_CONFIGURED = GRAPH_EMBEDDING in module.CONFIGURATION_REQUIRED
GRAPH_BATCH_REQUIREMENT = module.BATCH_CAPABLE_REQUIRED.get(GRAPH_EMBEDDING)
THREE_MODE_GRAPH = (
    "integration.test_decompositions_integration.TestGaussianBackendDecompositions",
    "test_graph_embed",
)
THREE_MODE_GRAPH_REQUIREMENT = deepcopy(module.UNBATCHED_REQUIRED[THREE_MODE_GRAPH])
THREE_MODE_GRAPH_CONFIGURED = THREE_MODE_GRAPH in module.CONFIGURATION_REQUIRED
THREE_MODE_GRAPH_BATCH = module.BATCH_CAPABLE_REQUIRED.get(THREE_MODE_GRAPH)
INITIAL_CUTOFFS = {
    (classname, function): cutoff
    for classname, functions, cutoff in (
        (INITIAL_CLASS, ("test_vacuum", "test_coherent"), 6),
        (INITIAL_CLASS, ("test_squeezed", "test_displaced_squeezed"), 8),
        (
            "integration.test_initial_state_controls.TestDistinctInitialMoments",
            ("test_coherent",),
            6,
        ),
        (
            "integration.test_initial_state_controls.TestDistinctInitialMoments",
            ("test_squeezed", "test_displaced_squeezed"),
            8,
        ),
    )
    for function in functions
}
ADAPTER_CASES += [
    (contract, "tf", profile)
    for contract in INITIAL_CUTOFFS
    if contract[0] != INITIAL_CLASS
    for profile in (
        "all-backends",
        "tensorflow-batched-pure",
        "tensorflow-batched-mixed",
    )
]
ADAPTER_CATALOG.update(
    {
        contract: ({}, deepcopy(module.BATCH_CAPABLE_REQUIRED[contract]))
        for contract in INITIAL_CUTOFFS
        if contract[0] != INITIAL_CLASS
    }
)

pytestmark = pytest.mark.frontend


@pytest.fixture(autouse=True)
def isolate_unbatched_contracts(monkeypatch):
    """Each synthetic report declares the additional contracts it exercises."""
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", set())
    monkeypatch.setattr(module, "REQUIRED_PARAMETERS", {})


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
            ("setup_hbar", str(unit)),
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


def test_expanded_batch_checks_require_setup_configuration(expanded_report):
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
        ("setup_hbar", "1.7"),
        ("setup_hbar", "0.5"),
        ("cutoff", "4"),
        ("batch_size", "None"),
        ("batch_size", "3"),
    ],
)
def test_mislabelled_configuration_is_rejected(expanded_report, property_name, bad_value):
    root, save = expanded_report
    for case in root:
        case.find(f"./properties/property[@name='{property_name}']").set("value", bad_value)
    with pytest.raises(ValueError, match="setup (units|configuration)"):
        verify(save(), "tensorflow-batched-pure")


def test_configuration_evidence_cannot_be_missing(expanded_report):
    root, save = expanded_report
    root[0].remove(root[0].find("properties"))
    with pytest.raises(ValueError, match="setup configuration"):
        verify(save(), "tensorflow-batched-pure")


def test_ambiguous_unit_metadata_is_rejected(expanded_report):
    """Legacy metadata cannot substitute for explicitly labelled setup units."""
    root, save = expanded_report
    for case in root:
        case.find("./properties/property[@name='setup_hbar']").set("name", "sf_hbar")
    with pytest.raises(ValueError, match="setup configuration"):
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
                    ("setup_hbar", str(unit)),
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
        case.find("./properties/property[@name='setup_hbar']").set("value", "1.7")
        if change_requested:
            case.find("./properties/property[@name='requested_hbar']").set("value", "1.7")
    with pytest.raises(ValueError, match="setup units"):
        verify(save(), "all-backends")


def test_state_reduction_rejects_swapped_units(reduction_report):
    """The correct total per unit cannot conceal circuits run at the wrong units."""
    root, save = reduction_report
    root[0].find("./properties/property[@name='setup_hbar']").set("value", "1.7")
    root[6].find("./properties/property[@name='setup_hbar']").set("value", "1.0")
    with pytest.raises(ValueError, match="setup configuration"):
        verify(save(), "all-backends")


def test_state_reduction_rejects_uneven_unit_counts(reduction_report):
    """All units and the total count remain present, but one circuit/unit is missing."""
    root, save = reduction_report
    for name in ("setup_hbar", "requested_hbar"):
        root[0].find(f"./properties/property[@name='{name}']").set("value", "1.7")
    with pytest.raises(ValueError, match="setup units counts"):
        verify(save(), "all-backends")


def test_state_reduction_requires_requested_unit_evidence(reduction_report):
    root, save = reduction_report
    properties = root[0].find("properties")
    properties.remove(properties.find("property[@name='requested_hbar']"))
    with pytest.raises(ValueError, match="setup configuration"):
        verify(save(), "all-backends")


@pytest.mark.parametrize(
    "configuration", ["all-backends", "tensorflow-batched-pure", "tensorflow-batched-mixed"]
)
@pytest.mark.parametrize("corruption", [None, "skipped", "missing_state", "ambiguous_state"])
def test_storage_first_labels(tmp_path, monkeypatch, configuration, corruption):
    """Legacy storage-first IDs must be counted without accepting conflicting labels."""
    contract = ("integration.test_tf_integration.TestOneMode", "test_eng_run_state_dm")
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {contract: {"tf": 1}})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    batched = configuration.startswith("tensorflow-batched-")
    states = [configuration.endswith("pure")] if batched else [True, False]
    root = ET.Element("testsuite")
    for pure in states:
        case = ET.SubElement(
            root, "testcase", classname=contract[0], name=f"{contract[1]}[{pure}-tf]"
        )
        props = ET.SubElement(case, "properties")
        for name, value in (
            ("setup_hbar", "1.7"),
            ("cutoff", "6"),
            ("batch_size", "2" if batched else "None"),
        ):
            ET.SubElement(props, "property", name=name, value=value)
    if corruption == "skipped":
        ET.SubElement(root[0], "skipped")
    elif corruption == "missing_state":
        root[0].set("name", f"{contract[1]}[tf]")
    elif corruption == "ambiguous_state":
        root[0].set("name", f"{contract[1]}[True-tf-False]")
    path = tmp_path / "storage-first.xml"
    ET.ElementTree(root).write(path)
    if corruption is None:
        assert verify(path, configuration) == len(states)
    else:
        match = {
            "skipped": "Did not pass",
            "missing_state": "Missing:",
            "ambiguous_state": "Ambiguous backend/state",
        }[corruption]
        with pytest.raises(ValueError, match=match):
            verify(path, configuration)


@pytest.mark.parametrize("contract", PARITY_REQUIREMENTS)
def test_extending_parity_keeps_gaussian_requirements(tmp_path, monkeypatch, contract):
    """A complete Bosonic report must not hide dropped Gaussian parity checks."""
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {contract: PARITY_REQUIREMENTS[contract]})
    root = ET.Element("testsuite")
    for pure in (True, False):
        for parameter in range(15):
            ET.SubElement(
                root,
                "testcase",
                classname=contract[0],
                name=f"{contract[1]}[BosonicBackend-{pure}-{parameter}]",
            )
    path = tmp_path / "bosonic-without-gaussian.xml"
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Missing: test_no_odd_fock/gaussian"):
        verify(path, "all-backends")


@pytest.mark.parametrize("contract,backend,profile", ADAPTER_CASES)
def test_density_adapters_cannot_disappear(tmp_path, monkeypatch, contract, backend, profile):
    """Exercise the real catalog: every enabled adapter must be required in CI."""
    unbatched, batch_capable = ADAPTER_CATALOG[contract]
    batched = profile.startswith("tensorflow-batched-")
    expected = batch_capable if batched else {**unbatched, **batch_capable}
    assert expected[backend] == 1
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {contract: unbatched} if unbatched else {})
    monkeypatch.setattr(
        module, "BATCH_CAPABLE_REQUIRED", {contract: batch_capable} if batch_capable else {}
    )
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    states = [profile.endswith("pure")] if batched else [True, False]
    root = ET.Element("testsuite")
    for required_backend in sorted(expected.keys() & module.MATRIX[profile]):
        for pure in states:
            case = ET.SubElement(
                root,
                "testcase",
                classname=contract[0],
                name=f"{contract[1]}[{required_backend}-{pure}]",
            )
            properties = ET.SubElement(case, "properties")
            for name, value in (
                ("setup_hbar", "1.7"),
                ("cutoff", str({**OPTICS_CUTOFFS, **INITIAL_CUTOFFS}.get(contract, 6))),
                ("batch_size", "2" if batched else "None"),
            ):
                ET.SubElement(properties, "property", name=name, value=value)
    path = tmp_path / "density-adapters.xml"
    ET.ElementTree(root).write(path)
    assert verify(path, profile) == len(root)
    removed = next(case for case in root if f"[{backend}-" in case.get("name"))
    root.remove(removed)
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Missing:"):
        verify(path, profile)
    root.append(removed)
    ET.SubElement(removed, "skipped")
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="Did not pass:"):
        verify(path, profile)


@pytest.mark.parametrize(
    "parameters,ambiguous",
    [
        ("tf-{state}-tf-{other}", True),
        ("{state}-tf-{other}", True),
        ("tf-{state}-{other}-tf", True),
        ("{state}-tf-{state}-fock", True),
        ("tf-{state}-extra-TFBackend-{other}", True),
        ("tf-{state}-tf-{state}-tf-{other}", True),
        ("{state}-tf-{state}-tf-{other}", True),
        ("tf-{state}-tf", False),
        ("{state}-tf-{state}", False),
        ("tf-{state}-TFBackend-{state}", False),
        ("TFBackend-{state}-tf-{state}", False),
        ("0.2-{state}-TFBackend-0.7", False),
    ],
)
def test_all_overlapping_labels_are_checked(tmp_path, monkeypatch, parameters, ambiguous):
    """Later conflicts cannot masquerade as a passing scalar or batched case."""
    contract = ("integration.test_guard", "test_contract")
    monkeypatch.setattr(module, "REQUIRED", {contract: {"tf"}})
    for profile in ("all-backends", "tensorflow-batched-pure", "tensorflow-batched-mixed"):
        states = [True, False] if profile == "all-backends" else [profile.endswith("pure")]
        root = ET.Element("testsuite")
        for state in states:
            ET.SubElement(
                root, "testcase", classname=contract[0], name=f"{contract[1]}[tf-{state}]"
            )
        labels = parameters.format(state=states[0], other=not states[0])
        root[0].set("name", f"{contract[1]}[{labels}]")
        path = tmp_path / "overlapping-labels.xml"
        ET.ElementTree(root).write(path)
        if ambiguous:
            with pytest.raises(ValueError, match="Ambiguous backend/state"):
                verify(path, profile)
        else:
            assert verify(path, profile) == len(states)


@pytest.fixture
def gradient_report(tmp_path, monkeypatch):
    """Use the real catalog, but list the supported gradient cases independently."""
    assert all(counts == {"tf": 1} for counts in GRADIENT_CATALOG.values())
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", GRADIENT_CATALOG)
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", set(GRADIENT_CATALOG))

    def build(profile):
        root = ET.Element("testsuite")
        batched = profile.startswith("tensorflow-batched-")
        states = [profile.endswith("pure")] if batched else [True, False]
        for function in GRADIENT_FUNCTIONS:
            for pure in states:
                ket = function == "test_coherent_ket_gradient"
                parameters = f"{pure}-tf" if ket else f"tf-{pure}"
                case = ET.SubElement(
                    root, "testcase", classname=GRADIENT_CLASS, name=f"{function}[{parameters}]"
                )
                if ket and not pure:
                    ET.SubElement(case, "skipped", message="Test skipped in mixed state mode")
                properties = ET.SubElement(case, "properties")
                for name, value in (
                    ("setup_hbar", "1.7"),
                    ("cutoff", "6"),
                    ("batch_size", "2" if batched else "None"),
                ):
                    ET.SubElement(properties, "property", name=name, value=value)

        def save():
            path = tmp_path / "gradients.xml"
            ET.ElementTree(root).write(path)
            return path

        return root, save

    return build


@pytest.mark.parametrize(
    "function,pure",
    [
        (function, pure)
        for function in GRADIENT_FUNCTIONS
        for pure in (True, False)
        if pure or function != "test_coherent_ket_gradient"
    ],
)
def test_gradient_guards_require_every_supported_case(gradient_report, function, pure):
    """All 15 cases must pass in scalar and batch jobs; the mixed ket stays excluded."""
    for profile in ("all-backends", "tensorflow-batched-" + ("pure" if pure else "mixed")):
        root, save = gradient_report(profile)
        assert verify(save(), profile) == (15 if profile == "all-backends" else 8 if pure else 7)
        parameters = f"{pure}-tf" if function == "test_coherent_ket_gradient" else f"tf-{pure}"
        case = next(case for case in root if case.get("name") == f"{function}[{parameters}]")
        root.remove(case)
        with pytest.raises(ValueError, match="Missing:"):
            verify(save(), profile)
        root.append(case)
        ET.SubElement(case, "skipped")
        with pytest.raises(ValueError, match="Did not pass:"):
            verify(save(), profile)


def test_only_ket_contracts_have_storage_exemptions():
    """The ket exception must not suppress mixed-state coverage elsewhere."""
    assert module.REQUIRED_STATES == {
        (GRADIENT_CLASS, "test_coherent_ket_gradient", "tf"): {"True"},
        (*THREE_MODE_GRAPH, "fock"): {"True"},
    }


@pytest.mark.parametrize("profile", ["non-tensorflow", "local-without-cloud"])
def test_gradient_guards_allow_missing_optional_tensorflow(gradient_report, profile):
    root, save = gradient_report(profile)
    root.clear()
    assert verify(save(), profile) == 0


@pytest.mark.parametrize(
    "contract,cutoff",
    list(OPTICS_CUTOFFS.items())
    + list(INITIAL_CUTOFFS.items())
    + [
        (
            (
                "integration.test_decompositions_integration.TestGaussianBackendDecomposeState",
                "test_vacuum",
            ),
            6,
        ),
    ],
)
@pytest.mark.parametrize(
    "profile", ["all-backends", "tensorflow-batched-pure", "tensorflow-batched-mixed"]
)
def test_per_test_cutoff_is_required(tmp_path, monkeypatch, contract, cutoff, profile):
    """A larger cutoff is a per-test contract, never a global allowlist."""
    monkeypatch.setattr(module, "REQUIRED", {contract: {"tf"}})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    states = [True, False] if profile == "all-backends" else [profile.endswith("pure")]
    root = ET.Element("testsuite")
    cutoffs = []
    for state in states:
        case = ET.SubElement(
            root, "testcase", classname=contract[0], name=f"{contract[1]}[tf-{state}]"
        )
        props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="setup_hbar", value="1.7")
        ET.SubElement(
            props, "property", name="batch_size", value="None" if profile == "all-backends" else "2"
        )
        cutoffs.append(ET.SubElement(props, "property", name="cutoff", value=str(cutoff)))
    path = tmp_path / "per-test-cutoff.xml"
    ET.ElementTree(root).write(path)
    assert verify(path, profile) == len(states)
    for wrong in {"6", "8", "12", "16", "18", "48", "72", "invalid", ""} - {str(cutoff)}:
        cutoffs[0].set("value", wrong)
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="incorrect setup configuration"):
            verify(path, profile)
    props = root[0].find("properties")
    props.remove(cutoffs[0])
    ET.ElementTree(root).write(path)
    with pytest.raises(ValueError, match="incorrect setup configuration"):
        verify(path, profile)


def test_only_audited_tests_override_cutoff():
    assert module.REQUIRED_CUTOFFS == {
        key: value
        for key, value in {
            **OPTICS_CUTOFFS,
            **INITIAL_CUTOFFS,
            GRAPH_EMBEDDING: 48,
            THREE_MODE_GRAPH: 72,
        }.items()
        if value != 6
    }


@pytest.fixture
def graph_report(tmp_path, monkeypatch):
    """Both Fock storage modes and the Gaussian controls require cutoff 48."""
    assert GRAPH_REQUIREMENT == {"gaussian": 1, "fock": 1}
    assert GRAPH_CONFIGURED
    assert GRAPH_BATCH_REQUIREMENT is None
    assert module.REQUIRED_CUTOFFS[GRAPH_EMBEDDING] == 48
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "UNBATCHED_REQUIRED", {GRAPH_EMBEDDING: GRAPH_REQUIREMENT})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {GRAPH_EMBEDDING})
    root = ET.Element("testsuite")
    for backend in ("gaussian", "fock", "tf", "bosonic"):
        for pure in (True, False):
            case = ET.SubElement(
                root,
                "testcase",
                classname=GRAPH_EMBEDDING[0],
                name=f"test_embed_graph[{backend}-{pure}-48]",
            )
            if backend not in GRAPH_REQUIREMENT:
                ET.SubElement(case, "skipped")
            properties = ET.SubElement(case, "properties")
            for name, value in (("setup_hbar", "1.7"), ("cutoff", "48"), ("batch_size", "None")):
                ET.SubElement(properties, "property", name=name, value=value)

    def save():
        path = tmp_path / "graph-embedding.xml"
        ET.ElementTree(root).write(path)
        return path

    return root, save


@pytest.mark.parametrize(
    "profile",
    ["all-backends", "non-tensorflow", "local-without-cloud", "fock", "frontend-apps-api-gaussian"],
)
def test_graph_guard_requires_each_storage_and_configuration(graph_report, profile):
    root, save = graph_report
    selected = module.MATRIX[profile] & GRAPH_REQUIREMENT.keys()
    assert verify(save(), profile) == 2 * len(selected)
    for case in list(root):
        if not any(f"[{backend}-" in case.get("name") for backend in selected):
            continue
        root.remove(case)
        with pytest.raises(ValueError, match="Missing:"):
            verify(save(), profile)
        root.append(case)
        skipped = ET.SubElement(case, "skipped")
        with pytest.raises(ValueError, match="Did not pass:"):
            verify(save(), profile)
        case.remove(skipped)
        duplicate = deepcopy(case)
        duplicate.set("name", duplicate.get("name")[:-1] + "-copy]")
        root.append(duplicate)
        with pytest.raises(ValueError, match="Expected 1 cases"):
            verify(save(), profile)
        root.remove(duplicate)
        properties = case.find("properties")
        for name, wrong_values in (
            ("cutoff", ("6", "8", "12", "16", "40", "56", "invalid", "")),
            ("setup_hbar", ("2.0",)),
            ("batch_size", ("2",)),
        ):
            prop = properties.find(f"property[@name='{name}']")
            correct = prop.get("value")
            for wrong in wrong_values:
                prop.set("value", wrong)
                with pytest.raises(ValueError, match="[Ii]ncorrect setup"):
                    verify(save(), profile)
            prop.set("value", correct)
            properties.remove(prop)
            with pytest.raises(ValueError, match="incorrect setup configuration"):
                verify(save(), profile)
            properties.append(prop)


@pytest.mark.parametrize(
    "profile", ["tensorflow", "tensorflow-batched-pure", "tensorflow-batched-mixed", "bosonic"]
)
def test_graph_guard_preserves_unsupported_exclusions(graph_report, profile):
    _, save = graph_report
    assert verify(save(), profile) == 0


@pytest.mark.parametrize("profile", sorted(module.MATRIX))
def test_three_mode_graph_guard(tmp_path, monkeypatch, profile):
    """Require pure Fock and both Gaussian controls; preserve the other exclusions."""
    assert THREE_MODE_GRAPH_REQUIREMENT == {"gaussian": 1, "fock": 1}
    assert THREE_MODE_GRAPH_CONFIGURED and THREE_MODE_GRAPH_BATCH is None
    assert module.REQUIRED_CUTOFFS[THREE_MODE_GRAPH] == 72
    assert module.REQUIRED_STATES[(*THREE_MODE_GRAPH, "fock")] == {"True"}
    assert (*THREE_MODE_GRAPH, "gaussian") not in module.REQUIRED_STATES
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(
        module, "UNBATCHED_REQUIRED", {THREE_MODE_GRAPH: THREE_MODE_GRAPH_REQUIREMENT}
    )
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {THREE_MODE_GRAPH})
    root = ET.Element("testsuite")
    for backend in ("gaussian", "fock", "tf", "bosonic"):
        for pure in (True, False):
            case = ET.SubElement(
                root,
                "testcase",
                classname=THREE_MODE_GRAPH[0],
                name=f"test_graph_embed[{backend}-{pure}-72]",
            )
            if backend != "gaussian" and (backend != "fock" or not pure):
                ET.SubElement(case, "skipped")
            properties = ET.SubElement(case, "properties")
            for name, value in (("setup_hbar", "1.7"), ("cutoff", "72"), ("batch_size", "None")):
                ET.SubElement(properties, "property", name=name, value=value)
    path = tmp_path / "three-mode-graph.xml"

    def save():
        ET.ElementTree(root).write(path)
        return path

    selected = module.MATRIX[profile] & THREE_MODE_GRAPH_REQUIREMENT.keys()
    expected = (2 if "gaussian" in selected else 0) + (1 if "fock" in selected else 0)
    assert verify(save(), profile) == expected
    for case in list(root):
        if case.find("skipped") is not None or not any(
            f"[{backend}-" in case.get("name") for backend in selected
        ):
            continue
        root.remove(case)
        with pytest.raises(ValueError, match="Missing:"):
            verify(save(), profile)
        root.append(case)
        skipped = ET.SubElement(case, "skipped")
        with pytest.raises(ValueError, match="Did not pass:"):
            verify(save(), profile)
        case.remove(skipped)
        duplicate = deepcopy(case)
        duplicate.set("name", duplicate.get("name")[:-1] + "-copy]")
        root.append(duplicate)
        with pytest.raises(ValueError, match="Expected 1 cases"):
            verify(save(), profile)
        root.remove(duplicate)
        properties = case.find("properties")
        for name, wrong_values in (
            ("cutoff", ("6", "48", "64", "80", "invalid", "")),
            ("setup_hbar", ("2.0",)),
            ("batch_size", ("2",)),
        ):
            prop = properties.find(f"property[@name='{name}']")
            correct = prop.get("value")
            for wrong in wrong_values:
                prop.set("value", wrong)
                with pytest.raises(ValueError, match="[Ii]ncorrect setup"):
                    verify(save(), profile)
            prop.set("value", correct)
            properties.remove(prop)
            with pytest.raises(ValueError, match="incorrect setup configuration"):
                verify(save(), profile)
            properties.append(prop)


@pytest.mark.parametrize(
    "profile", ["all-backends", "tensorflow-batched-pure", "tensorflow-batched-mixed"]
)
def test_each_passive_mesh_is_required(tmp_path, monkeypatch, profile):
    assert PASSIVE_REQUIREMENT == {"tf": 5}
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {PASSIVE_CONTROL: PASSIVE_REQUIREMENT})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {PASSIVE_CONTROL})
    monkeypatch.setattr(module, "REQUIRED_PARAMETERS", {PASSIVE_CONTROL: PASSIVE_PARAMETERS})
    assert PASSIVE_PARAMETERS == ("mesh",)
    states = [True, False] if profile == "all-backends" else [profile.endswith("pure")]
    root = ET.Element("testsuite")
    for state in states:
        for mesh in (
            "passive",
            "rectangular",
            "rectangular_compact",
            "triangular_compact",
            "sun_compact",
        ):
            case = ET.SubElement(
                root,
                "testcase",
                classname=PASSIVE_CONTROL[0],
                name=f"{PASSIVE_CONTROL[1]}[{mesh}-tf-{state}]",
            )
            props = ET.SubElement(case, "properties")
            for name, value in (
                ("setup_hbar", "1.7"),
                ("cutoff", "6"),
                ("batch_size", "None" if profile == "all-backends" else "2"),
                ("parameter_mesh", mesh),
            ):
                ET.SubElement(props, "property", name=name, value=value)
    path = tmp_path / "distinct-meshes.xml"
    ET.ElementTree(root).write(path)
    assert verify(path, profile) == 5 * len(states)
    for case in list(root):
        root.remove(case)
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Expected 5 cases"):
            verify(path, profile)
        root.append(case)
        skipped = ET.SubElement(case, "skipped")
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Did not pass"):
            verify(path, profile)
        case.remove(skipped)
        original_name = case.get("name")
        mesh_property = case.find("./properties/property[@name='parameter_mesh']")
        original_mesh = mesh_property.get("value")
        other = next(
            c
            for c in root
            if c.get("name").endswith(original_name.rsplit("-", 1)[1]) and c is not case
        )
        other_name = other.get("name")
        duplicate = other.find("./properties/property[@name='parameter_mesh']").get("value")
        # Pytest disambiguates repeated values with 0/1 suffixes, so their IDs differ.
        other.set("name", other_name.replace(f"[{duplicate}-", f"[{duplicate}0-"))
        case.set("name", other_name.replace(f"[{duplicate}-", f"[{duplicate}1-"))
        mesh_property.set("value", duplicate)
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Incorrect mesh coverage"):
            verify(path, profile)
        other.set("name", other_name)
        case.set("name", original_name)
        mesh_property.set("value", "unknown_mesh")
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Incorrect mesh coverage"):
            verify(path, profile)
        mesh_property.set("value", original_mesh)
        props = case.find("properties")
        props.remove(mesh_property)
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Missing or ambiguous parameter mesh"):
            verify(path, profile)
        props.append(mesh_property)
        duplicate_property = deepcopy(mesh_property)
        props.append(duplicate_property)
        ET.ElementTree(root).write(path)
        with pytest.raises(ValueError, match="Missing or ambiguous parameter mesh"):
            verify(path, profile)
        props.remove(duplicate_property)


@pytest.mark.parametrize("function", INITIAL_CASES)
@pytest.mark.parametrize(
    "backend,profile",
    [(backend, "all-backends") for backend in ("gaussian", "bosonic", "fock", "tf")]
    + [("tf", "tensorflow-batched-pure"), ("tf", "tensorflow-batched-mixed")],
)
def test_every_initial_moment_case_is_required(tmp_path, monkeypatch, function, backend, profile):
    """Keep every new and native case; duplicate parameter values cannot replace one."""
    contract = INITIAL_CLASS, function
    parameters, count, cutoff = INITIAL_CASES[function]
    assert INITIAL_REQUIREMENTS[contract] == {b: count for b in module.BACKENDS}
    assert INITIAL_PARAMETERS[contract] == parameters
    monkeypatch.setattr(module, "REQUIRED", {})
    monkeypatch.setattr(module, "BATCH_CAPABLE_REQUIRED", {contract: {backend: count}})
    monkeypatch.setattr(module, "CONFIGURATION_REQUIRED", {contract})
    monkeypatch.setattr(module, "REQUIRED_PARAMETERS", {contract: parameters})
    states = [True, False] if profile == "all-backends" else [profile.endswith("pure")]
    root = ET.Element("testsuite")
    for state in states:
        for index in range(count):
            case = ET.SubElement(
                root,
                "testcase",
                classname=INITIAL_CLASS,
                name=f"{function}[{backend}-{state}-{index}]",
            )
            props = ET.SubElement(case, "properties")
            for name, value in (
                ("setup_hbar", "1.7"),
                ("cutoff", str(cutoff)),
                ("batch_size", "None" if profile == "all-backends" else "2"),
            ):
                ET.SubElement(props, "property", name=name, value=value)
            for parameter in parameters:
                ET.SubElement(props, "property", name="parameter_" + parameter, value=str(index))
    path = tmp_path / "initial-moments.xml"

    def save():
        ET.ElementTree(root).write(path)
        return path

    assert verify(save(), profile) == count * len(states)
    for case in list(root):
        root.remove(case)
        with pytest.raises(ValueError, match="Missing:|Expected .* cases"):
            verify(save(), profile)
        root.append(case)
        skipped = ET.SubElement(case, "skipped")
        with pytest.raises(ValueError, match="Did not pass"):
            verify(save(), profile)
        case.remove(skipped)
        for parameter in parameters:
            props = case.find("properties")
            prop = props.find(f"property[@name='parameter_{parameter}']")
            props.remove(prop)
            with pytest.raises(ValueError, match="Missing or ambiguous parameter"):
                verify(save(), profile)
            props.append(prop)
        if parameters:
            values = case.findall("./properties/property")
            original = [
                (p, p.get("value")) for p in values if p.get("name").startswith("parameter_")
            ]
            for prop, value in original:
                prop.set("value", str((int(value) + 1) % count))
            with pytest.raises(ValueError, match="Missing or repeated parameter values"):
                verify(save(), profile)
            for prop, value in original:
                prop.set("value", value)
