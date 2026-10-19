# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Analytic and negative controls for the pure three-mode graph test."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields.backends.states import BaseFockState

pytestmark = pytest.mark.frontend


@pytest.fixture(scope="module")
def graph_module():
    path = Path(__file__).resolve().parents[1] / "integration/test_decompositions_integration.py"
    spec = importlib.util.spec_from_file_location("three_mode_graph_assertion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def graph_ket_moments(graph_module, cutoff, hbar, tol):
    return graph_module.graph_ket_moments.__wrapped__(cutoff, hbar, tol)


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize("reference", ["boundary", "complex_pair", "displaced_product"])
def test_three_mode_analytic_moments(graph_ket_moments, cutoff, hbar, reference):
    """Keep boundary terms, complex correlations, xxpp order and covariance centering."""
    ket = np.zeros((cutoff,) * 3, dtype=complex)
    means = np.zeros(6)
    if reference == "boundary":
        occupations = np.array([cutoff - 1, 1, 0])
        ket[tuple(occupations)] = 1
        covariance = np.diag(hbar * np.tile(occupations + 0.5, 2))
    elif reference == "complex_pair":
        phase = 0.4
        ket[0, 0, 1] = 0.6
        ket[1, 1, 1] = 0.8 * np.exp(1j * phase)
        covariance = np.diag(hbar * np.tile(np.array([0.64, 0.64, 1]) + 0.5, 2))
        xx, xp = hbar * 0.48 * np.array([np.cos(phase), np.sin(phase)])
        covariance[0, 1] = covariance[1, 0] = xx
        covariance[3, 4] = covariance[4, 3] = -xx
        covariance[0, 4] = covariance[4, 0] = xp
        covariance[1, 3] = covariance[3, 1] = xp
    else:
        amplitudes = np.array([0.2 + 0.3j, -0.4 + 0.1j, 0.1 - 0.2j])
        photons = abs(amplitudes) ** 2 / (1 + abs(amplitudes) ** 2)
        ladder_mean = amplitudes / (1 + abs(amplitudes) ** 2)
        means = np.sqrt(2 * hbar) * np.r_[ladder_mean.real, ladder_mean.imag]
        covariance = np.diag(hbar * np.tile(photons + 0.5, 2) - means**2)
        for mode in range(3):
            covariance[mode, mode + 3] = covariance[mode + 3, mode] = -means[mode] * means[mode + 3]
        factors = np.zeros((3, cutoff), dtype=complex)
        factors[:, 0], factors[:, 1] = 1, amplitudes
        factors /= np.sqrt(1 + abs(amplitudes) ** 2)[:, None]
        ket = np.einsum("i,j,k->ijk", *factors)
    state = BaseFockState(ket, 3, True, cutoff)
    actual_means, actual_covariance = graph_ket_moments(state)
    np.testing.assert_allclose(actual_means, means, atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_covariance, covariance, atol=1e-12, rtol=0)


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize("cutoff", [3, 4, 6], indirect=True)
def test_three_mode_ket_matches_density(graph_ket_moments, state_moments, cutoff):
    """An unrelated density contraction checks arbitrary complex states."""
    rng = np.random.default_rng(1977)
    ket = rng.normal(size=(cutoff,) * 3) + 1j * rng.normal(size=(cutoff,) * 3)
    ket /= np.linalg.norm(ket)
    state = BaseFockState(ket, 3, True, cutoff)
    for actual, expected in zip(graph_ket_moments(state), state_moments(state)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize(
    "corruption", ["shape", "nonfinite", "normalization", "mixed", "units", "modes", "batch"]
)
def test_three_mode_ket_rejects_invalid_state(graph_ket_moments, cutoff, hbar, corruption):
    ket = np.zeros((cutoff,) * 3, dtype=complex)
    ket[0, 0, 0] = 1
    if corruption == "shape":
        ket = ket[..., :-1]
    elif corruption == "nonfinite":
        ket[0, 0, 0] = np.nan
    elif corruption == "normalization":
        ket *= np.sqrt(0.99)
    elif corruption == "batch":
        ket = np.stack([ket, ket])
    state = SimpleNamespace(
        is_pure=corruption != "mixed",
        num_modes=2 if corruption == "modes" else 3,
        hbar=2 * hbar if corruption == "units" else hbar,
        ket=lambda: ket,
    )
    with pytest.raises(AssertionError):
        graph_ket_moments(state)


@pytest.mark.parametrize("cutoff", [72], indirect=True)
def test_three_mode_ket_never_materializes_density(graph_ket_moments, cutoff, hbar, monkeypatch):
    ket = np.zeros((cutoff,) * 3, dtype=complex)
    ket[0, 0, 0] = 1
    state = BaseFockState(ket, 3, True, cutoff)

    def forbidden(*args, **kwargs):
        pytest.fail("The pure-ket oracle must not expand the full density matrix")

    monkeypatch.setattr(state, "dm", forbidden)
    monkeypatch.setattr(state, "reduced_dm", forbidden)
    means, covariance = graph_ket_moments(state)
    np.testing.assert_allclose(means, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(covariance, np.eye(6) * hbar / 2, atol=1e-12, rtol=0)


@pytest.fixture(scope="module")
def graph_kets(graph_module):
    result = {}
    for name, photons in (("reference", 1), ("energy", 0.8)):
        program = sf.Program(3)
        with program.context as q:
            sf.ops.GraphEmbed(graph_module.A, photons) | q
        engine = sf.Engine("fock", backend_options={"cutoff_dim": 72, "pure": True})
        result[name] = engine.run(program).state.ket()
    return result


@pytest.mark.parametrize("cutoff", [72], indirect=True)
@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "normalization",
        "conjugate",
        "permutation",
        "displacement",
        "shape",
        "nonfinite",
        "units",
        "storage",
        "energy",
    ],
)
def test_three_mode_graph_assertion(
    graph_module, graph_kets, graph_ket_moments, cutoff, hbar, tol, corruption
):
    """Exercise the actual integration assertion, including full complex graph moments."""
    ket = graph_kets["energy" if corruption == "energy" else "reference"].copy()
    if corruption == "normalization":
        ket *= np.sqrt(0.99)
    elif corruption == "conjugate":
        ket = ket.conj()
    elif corruption == "permutation":
        ket = ket.swapaxes(0, 1)
    elif corruption == "displacement":
        ket[1, 0, 0] += 0.01
    elif corruption == "shape":
        ket = ket[..., :-1]
    elif corruption == "nonfinite":
        ket.flat[0] = np.nan
    state = SimpleNamespace(
        is_pure=corruption != "storage",
        num_modes=3,
        hbar=2 * hbar if corruption == "units" else hbar,
        ket=lambda: ket,
    )

    def setup_eng(modes):
        assert modes == 3
        return SimpleNamespace(
            backend_name="fock", run=lambda program: SimpleNamespace(state=state)
        ), sf.Program(modes)

    def check():
        graph_module.TestGaussianBackendDecompositions().test_graph_embed(
            setup_eng, tol, True, hbar, graph_ket_moments
        )

    if corruption is None:
        check()
    else:
        with pytest.raises(AssertionError):
            check()


def test_three_mode_mixed_exclusion_precedes_execution(graph_module, graph_ket_moments, hbar, tol):
    """The mixed-state exclusion must happen before an oversized allocation."""

    def setup_eng(modes):
        def forbidden(program):
            pytest.fail("Mixed Fock execution must remain excluded")

        return SimpleNamespace(backend_name="fock", run=forbidden), sf.Program(modes)

    with pytest.raises(pytest.skip.Exception, match="pure Fock ket"):
        graph_module.TestGaussianBackendDecompositions().test_graph_embed(
            setup_eng, tol, False, hbar, graph_ket_moments
        )
