# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Reject missing or skipped checks for the maintained backend contracts."""

import argparse
from collections import Counter
import re
import xml.etree.ElementTree as ET

BACKENDS = {"fock", "gaussian", "bosonic", "tf"}
MATRIX = {
    "all-backends": BACKENDS,
    "frontend-apps-api-gaussian": {"gaussian"},
    "fock": {"fock"},
    "tensorflow": {"tf"},
    "tensorflow-batched-pure": {"tf"},
    "tensorflow-batched-mixed": {"tf"},
    "bosonic": {"bosonic"},
    "non-tensorflow": BACKENDS - {"tf"},
    "local-without-cloud": BACKENDS - {"tf"},
}
REQUIRED = {
    (
        "backend.test_beamsplitter_operation.TestModeSubsets",
        "test_beamsplitter_on_mode_subset",
    ): BACKENDS,
    (
        "backend.test_twomode_squeezing_operation.TestTwomodeSqueezing",
        "test_squeezing_on_mode_subset",
    ): {"fock", "tf"},
    ("backend.test_heterodyne.TestHeterodyne", "test_mode_reset_vacuum"): {
        "gaussian",
        "bosonic",
    },
    (
        "integration.test_engine_integration.TestProperExecution",
        "test_measurefock_shots_exception",
    ): {"fock", "tf"},
    (
        "backend.test_fock_measurement.TestRepresentationIndependent",
        "test_two_mode_squeezed_measurements",
    ): {"fock", "tf", "gaussian"},
}
REQUIRED.update(
    {
        ("backend.test_mzgate_operation.TestFockRepresentation", function): {
            "fock",
            "tf",
        }
        for function in (
            "test_mzgate_two_photon_input",
            "test_mzgate_operation_equals_decomposition",
        )
    }
)
# Expected parameter counts per pure/mixed state for the newly enabled checks.
# Their TensorFlow assertions and APIs are verified only in unbatched mode.
UNBATCHED_REQUIRED = {
    ("backend.test_squeeze_operation.TestFockRepresentation", "test_no_odd_fock"): {"gaussian": 15},
    (
        "backend.test_squeezed_state_preparation.TestFockRepresentation",
        "test_no_odd_fock",
    ): {"gaussian": 15},
    ("backend.test_states.TestBaseFockKetDensityMatrix", "test_rdm"): {"bosonic": 1},
    (
        "backend.test_states.TestBaseFockKetDensityMatrix",
        "test_density_matrix_thermal_state",
    ): {"bosonic": 1},
    ("backend.test_states.TestFidelities", "test_vacuum"): {"bosonic": 1},
    (
        "backend.test_states.TestNumberExpectation",
        "test_number_expectation_four_modes",
    ): {"gaussian": 1},
    ("backend.test_states.TestParityExpectation", "test_thermal"): {
        "gaussian": 1,
        "bosonic": 1,
    },
    (
        "backend.test_states_polyquad.TestMultiModePolyQuadratureExpectations",
        "test_three_mode_arbitrary",
    ): {"tf": 1},
    (
        "backend.test_twomode_squeezing_operation.TestTwomodeSqueezing",
        "test_two_mode_squeezing",
    ): {"tf": 15},
    (
        "integration.test_measurement_integration.TestDarkCounts",
        "test_fock_darkcounts_single_mode",
    ): {"tf": 4},
    (
        "integration.test_measurement_integration.TestDarkCounts",
        "test_fock_darkcounts_two_modes",
    ): {"tf": 3},
}
UNBATCHED_REQUIRED.update(
    {
        (
            "backend.test_states_polyquad.TestSingleModePolyQuadratureExpectations",
            function,
        ): {"tf": 1}
        for function in (
            "test_no_expectation",
            "test_constant",
            "test_linear_vacuum",
            "test_x_squeezed",
            "test_x_displaced",
            "test_x_displaced_squeezed",
            "test_p_displaced_squeezed",
            "test_linear_combination",
            "test_n_thermal",
            "test_n_squeeze",
            "test_x_squared",
            "test_p_squared",
            "test_xp_vacuum",
            "test_xp_displaced_squeezed",
            "test_arbitrary_quadratic",
        )
    }
)
# These assertions check full batch shapes and every member's values.
BATCH_CAPABLE_REQUIRED = {
    key: UNBATCHED_REQUIRED.pop(key)
    for key in (
        (
            "backend.test_twomode_squeezing_operation.TestTwomodeSqueezing",
            "test_two_mode_squeezing",
        ),
        (
            "integration.test_measurement_integration.TestDarkCounts",
            "test_fock_darkcounts_single_mode",
        ),
        (
            "integration.test_measurement_integration.TestDarkCounts",
            "test_fock_darkcounts_two_modes",
        ),
    )
}
BATCH_CAPABLE_REQUIRED.update(
    {
        ("integration.test_engine_integration.TestEngineReset", "test_eng_reset"): {
            "fock": 1,
            "tf": 1,
        },
        ("backend.test_states.TestBosonicState", "test_is_vacuum"): {
            "gaussian": 1,
            "bosonic": 1,
            "tf": 1,
        },
        (
            "integration.test_engine_integration.TestPhotonStatistics",
            "test_three_mode_fock_state_mean_photon",
        ): {"fock": 1, "tf": 1},
        (
            "backend.test_twomode_squeezing_operation.TestTwomodeSqueezing",
            "test_distinct_squeezing_inputs",
        ): {"fock": 1, "tf": 1},
        (
            "integration.test_measurement_integration.TestDarkCounts",
            "test_distinct_fock_inputs_dark_counts",
        ): {"fock": 1, "tf": 1},
        (
            "integration.test_hbar_integration.TestIntegration",
            "test_squeeze_variance_frontend",
        ): {backend: 2 for backend in BACKENDS},
    }
)
BATCH_CAPABLE_REQUIRED.update(
    {
        ("integration.test_hbar_integration.TestIntegration", function): {
            backend: 2 for backend in BACKENDS
        }
        for function in ("test_x_displacement", "test_z_displacement")
    }
)
BATCH_CAPABLE_REQUIRED.update(
    {
        ("backend.test_state_reduction.TestStateReduction", function): {
            backend: count for backend in BACKENDS
        }
        for function, count in (
            ("test_entangled_marginal_is_thermal", 18),
            ("test_retained_entangled_pair_tensor", 6),
            ("test_mixed_marginal_with_coherent_spectator", 6),
            ("test_mixed_parent_can_have_pure_subsystem", 3),
            ("test_photon_probabilities_of_squeezed_pair", 3),
            ("test_coherent_marginal_retains_cutoff_tail", 3),
        )
    }
)
EXPANDED_UNBATCHED = {
    (
        "backend.test_displacement_operation.TestFockRepresentation",
        "test_coherent_state_fock_elements",
    ): {"gaussian": 28, "bosonic": 28},
    (
        "backend.test_loss_channel.TestFockRepresentation",
        "test_loss_channel_on_coherent_states",
    ): {"gaussian": 27, "bosonic": 27},
    (
        "backend.test_beamsplitter_operation.TestFockRepresentation",
        "test_coherent_vacuum_interfered_fock_elements",
    ): {"gaussian": 18, "bosonic": 18},
    (
        "backend.test_squeeze_operation.TestFockRepresentation",
        "test_reference_squeezed_vacuum",
    ): {"gaussian": 15, "bosonic": 15},
    (
        "backend.test_squeezed_state_preparation.TestFockRepresentation",
        "test_reference_squeezed_states",
    ): {"gaussian": 15, "bosonic": 15},
    (
        "backend.test_displaced_squeezed_state_preparation.TestFockRepresentation",
        "test_displaced_squeezed_with_no_displacement",
    ): {"gaussian": 4, "bosonic": 4},
    ("integration.test_algorithms", "test_gaussian_boson_sampling_fock_probs"): {"bosonic": 1},
    (
        "integration.test_decompositions_integration.TestFockBackendDecomposeState",
        "test_vacuum",
    ): {"gaussian": 1},
    ("backend.test_states.TestBosonicState", "test_displacement"): {"gaussian": 1},
    ("backend.test_states.TestBosonicState", "test_density_matrix"): {"gaussian": 1},
}
UNBATCHED_REQUIRED.update(EXPANDED_UNBATCHED)
CONFIGURATION_REQUIRED = set(EXPANDED_UNBATCHED) | set(BATCH_CAPABLE_REQUIRED)
PARAMETRIZED_UNITS = {
    "integration.test_hbar_integration.TestIntegration": {0.5, 1.0},
    "backend.test_state_reduction.TestStateReduction": {1.0, 1.7, 2.0},
}
REQUIRED.update(
    {
        ("backend.test_states.TestBaseStateMeanPhotonNumber", function): BACKENDS
        for function in (
            "test_mean_photon_coherent",
            "test_mean_photon_squeezed",
            "test_mean_photon_displaced_squeezed",
            "test_mean_photon_displaced_thermal",
        )
    }
)
REQUIRED.update(
    {
        ("backend.test_states_wigner", "test_fock_1_state_quad"): {"fock", "tf"},
        ("backend.test_states_wigner", "test_vacuum_state_quad"): BACKENDS,
        ("backend.test_states_wigner", "test_coherent_state_quad"): BACKENDS,
    }
)
REQUIRED.update(
    {
        (
            "integration.test_utils_integration.TestInitialStatesAgreeGaussian",
            function,
        ): {
            "gaussian",
            "bosonic",
        }
        for function in (
            "test_vacuum",
            "test_coherent",
            "test_squeezed",
            "test_displaced_squeezed",
        )
    }
)
UNBATCHED_ONLY = {
    "test_measurefock_shots_exception",
    "test_fock_1_state_quad",
    "test_vacuum_state_quad",
    "test_coherent_state_quad",
}
LABELS = {
    "FockBackend": "fock",
    "GaussianBackend": "gaussian",
    "BosonicBackend": "bosonic",
    "TFBackend": "tf",
}
PARAMETERS = re.compile(
    r"(?:\[|-)(FockBackend|GaussianBackend|BosonicBackend|TFBackend|fock|gaussian|bosonic|tf)"
    r"-(True|False)(?:-|\])"
)


def verify(report, configuration):
    """Require every selected supported backend/state combination to execute."""
    selected = MATRIX[configuration]
    batched = configuration.startswith("tensorflow-batched-")
    required = dict(REQUIRED)
    counts_required = dict(BATCH_CAPABLE_REQUIRED)
    if not batched:
        counts_required.update(UNBATCHED_REQUIRED)
    for key, counts in counts_required.items():
        required[key] = required.get(key, set()) | counts.keys()
    states = {"True", "False"}
    if batched:
        states = {"True" if configuration.endswith("pure") else "False"}
    cases = list(ET.parse(report).getroot().iter("testcase"))
    errors = []
    seen = set()
    for case in cases:
        identity = (case.get("classname", ""), case.get("name", ""))
        if identity in seen:
            errors.append(f"Duplicate test case: {identity[0]}::{identity[1]}")
        seen.add(identity)
    checked = 0
    for (classname, function), supported in required.items():
        if batched and function in UNBATCHED_ONLY:
            continue
        found = {}
        for case in cases:
            name = case.get("name", "")
            if case.get("classname") != classname or name.split("[")[0] != function:
                continue
            match = PARAMETERS.search(name)
            if match:
                backend, pure = match.groups()
                found.setdefault((LABELS.get(backend, backend), pure), []).append(case)
        for backend in sorted(supported & selected):
            for pure in sorted(states):
                matching = found.get((backend, pure), [])
                label = f"{function}/{backend}/pure={pure}"
                if not matching:
                    errors.append(f"Missing: {label}")
                expected = counts_required.get((classname, function), {}).get(backend)
                if expected is not None and len(matching) != expected:
                    errors.append(f"Expected {expected} cases for {label}, found {len(matching)}")
                for case in matching:
                    if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
                        errors.append(f"Did not pass: {label}: {case.get('name')}")
                if (classname, function) in CONFIGURATION_REQUIRED:
                    units = Counter()
                    for case in matching:
                        properties = {
                            prop.get("name"): prop.get("value")
                            for prop in case.findall("./properties/property")
                        }
                        try:
                            effective = float(properties["sf_hbar"])
                            units[effective] += 1
                            if classname in PARAMETRIZED_UNITS:
                                assert effective == float(properties["requested_hbar"])
                            assert int(properties["cutoff"]) == 6
                            assert properties["batch_size"] == ("2" if batched else "None")
                        except (KeyError, ValueError, AssertionError):
                            errors.append(
                                f"Missing or incorrect effective configuration: {label}: {case.get('name')}"
                            )
                    expected_units = PARAMETRIZED_UNITS.get(classname, {1.7})
                    if matching and set(units) != expected_units:
                        errors.append(
                            f"Incorrect effective units for {label}: expected {sorted(expected_units)}, found {sorted(units)}"
                        )
                    if expected is not None and matching:
                        per_unit, remainder = divmod(expected, len(expected_units))
                        if remainder or units != {unit: per_unit for unit in expected_units}:
                            errors.append(
                                f"Incorrect effective units counts for {label}: {dict(units)}"
                            )
                checked += len(matching)
    if errors:
        raise ValueError(
            "Required backend checks were not executed successfully:\n" + "\n".join(errors)
        )
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("configuration", choices=sorted(MATRIX))
    args = parser.parse_args()
    print(f"Verified {verify(args.report, args.configuration)} required backend checks.")
