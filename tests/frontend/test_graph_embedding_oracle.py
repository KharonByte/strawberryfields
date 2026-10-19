# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Independent finite-state controls for the graph photon-number assertion."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.backends.states import BaseFockState

pytestmark = pytest.mark.frontend


@pytest.fixture
def graph_assertion():
    """Exercise the maintained integration assertion without its simulator output."""
    path = Path(__file__).resolve().parents[1] / "integration/test_measurement_integration.py"
    spec = importlib.util.spec_from_file_location("graph_integration_assertion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TestPostselection().test_embed_graph


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize("pure", [True, False])
@pytest.mark.parametrize(
    "corruption",
    [None, "energy", "normalization", "displacement", "units", "shape", "nonfinite", "storage"],
)
def test_graph_photon_number_reference(
    graph_assertion, state_moments, cutoff, hbar, pure, corruption
):
    """Known Fock populations expose errors that a relaxed 0.001 tolerance hides."""
    ket = np.zeros((cutoff, cutoff), dtype=complex)
    single_photon = 0.01 if corruption == "displacement" else 0
    mean_photon = 2.0001 if corruption == "energy" else 2.0
    pair_population = (mean_photon - single_photon) / 4
    ket[0, 0] = np.sqrt(1 - pair_population - single_photon)
    ket[1, 0] = np.sqrt(single_photon)
    ket[2, 2] = np.sqrt(pair_population) * np.exp(0.37j)
    data = ket if pure else np.einsum("ij,kl->ikjl", ket, ket.conj())
    state = BaseFockState(data, num_modes=2, pure=pure, cutoff_dim=cutoff)
    original = state.reduced_dm

    def reduced_dm(modes):
        density = original(modes).copy()
        if corruption == "normalization":
            density *= 0.99
        elif corruption == "shape":
            density = density[..., :-1]
        elif corruption == "nonfinite":
            density.flat[0] = np.nan
        return density

    state.reduced_dm = reduced_dm

    def setup_eng(modes):
        assert modes == 2
        engine = SimpleNamespace(
            backend_name="fock",
            backend_options={"pure": not pure if corruption == "storage" else pure},
            run=lambda program: SimpleNamespace(state=state),
        )
        return engine, sf.Program(modes)

    def check():
        graph_assertion(setup_eng, hbar * (2 if corruption == "units" else 1), state_moments)

    if corruption is None:
        check()
    else:
        if corruption == "energy":
            assert np.allclose(mean_photon, 2, atol=1e-3, rtol=0)
        with pytest.raises(AssertionError):
            check()
