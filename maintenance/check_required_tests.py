# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Reject missing or skipped checks for the maintained backend contracts."""

import argparse
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
    ("backend.test_heterodyne.TestHeterodyne", "test_mode_reset_vacuum"): {"gaussian", "bosonic"},
    (
        "integration.test_engine_integration.TestProperExecution",
        "test_measurefock_shots_exception",
    ): {"fock", "tf"},
}
REQUIRED.update(
    {
        ("backend.test_mzgate_operation.TestFockRepresentation", function): {"fock", "tf"}
        for function in (
            "test_mzgate_two_photon_input",
            "test_mzgate_operation_equals_decomposition",
        )
    }
)
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
        ("integration.test_utils_integration.TestInitialStatesAgreeGaussian", function): {
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
    states = {"True", "False"}
    if batched:
        states = {"True" if configuration.endswith("pure") else "False"}
    cases = list(ET.parse(report).getroot().iter("testcase"))
    errors = []
    checked = 0
    for (classname, function), supported in REQUIRED.items():
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
                for case in matching:
                    if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
                        errors.append(f"Did not pass: {label}: {case.get('name')}")
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
