# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Bosonic Fock conversion checked against independent state expansions."""

import itertools

import numpy as np
import pytest
from scipy.linalg import block_diag
from scipy.special import factorial

import strawberryfields as sf
from strawberryfields import ops
from strawberryfields.backends.states import BaseBosonicState


@pytest.fixture(params=[1.0, 1.7, 2.0, 4.0])
def conversion_units(request, monkeypatch):
    """A change of quadrature units cannot change Fock matrix elements."""
    monkeypatch.setattr(sf, "hbar", request.param)


def coherent_coefficients(alpha, cutoff):
    """Exact coherent-state coefficients without renormalizing the cutoff tail."""
    n = np.arange(cutoff)
    return np.exp(-abs(alpha) ** 2 / 2) * alpha**n / np.sqrt(factorial(n))


def coherent_overlap(left, right):
    """Inner product of multimode coherent states."""
    return np.exp(-np.vdot(left, left) / 2 - np.vdot(right, right) / 2 + np.vdot(left, right))


def cat_components(amplitude, phase, parity):
    """Coherent amplitudes and normalized superposition coefficients."""
    alpha = amplitude * np.exp(1j * phase)
    norm = np.sqrt(2 * (1 + np.cos(np.pi * parity) * np.exp(-2 * amplitude**2)))
    return np.array([alpha, -alpha]), np.array([1, np.exp(1j * np.pi * parity)]) / norm


def attenuated_components(amplitudes, coefficients, transmission):
    """Trace the loss ports using exact overlaps of coherent states."""
    environment = amplitudes * np.sqrt(1 - np.asarray(transmission))
    weights = np.outer(coefficients, coefficients.conj())
    for i, j in itertools.product(range(len(coefficients)), repeat=2):
        weights[i, j] *= coherent_overlap(environment[j], environment[i])
    return amplitudes * np.sqrt(transmission), weights


def coherent_sum_density(amplitudes, weights, modes, cutoff):
    """Density tensor of coherent dyads, tracing discarded modes analytically."""
    discarded = [mode for mode in range(amplitudes.shape[1]) if mode not in modes]
    kets = []
    for branch in amplitudes:
        ket = np.array(1)
        for mode in modes:
            ket = np.multiply.outer(ket, coherent_coefficients(branch[mode], cutoff))
        kets.append(ket)
    density = np.zeros([cutoff] * (2 * len(modes)), dtype=complex)
    order = np.arange(2 * len(modes)).reshape(2, -1).T.flatten()
    for i, j in itertools.product(range(len(weights)), repeat=2):
        overlap = coherent_overlap(amplitudes[j, discarded], amplitudes[i, discarded])
        density += (
            weights[i, j] * overlap * np.multiply.outer(kets[i], kets[j].conj()).transpose(order)
        )
    return density


def assert_density(actual, expected):
    """Check every coherence, Hermiticity and positivity at the chosen cutoff."""
    np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
    order = list(range(0, actual.ndim, 2)) + list(range(1, actual.ndim, 2))
    dimension = actual.shape[0] ** (actual.ndim // 2)
    matrix = actual.transpose(order).reshape(dimension, dimension)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-11, rtol=0)
    assert np.linalg.eigvalsh(matrix).min() >= -2e-11


@pytest.mark.bosonic
@pytest.mark.usefixtures("conversion_units")
class TestBosonicFockConversion:
    """Physical states containing non-Hermitian Gaussian components."""

    @pytest.mark.parametrize("amplitude", [0.4, 1.1])
    @pytest.mark.parametrize("phase", [0.0, 0.6])
    @pytest.mark.parametrize("parity", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("transmission", [0.0, 0.4, 1.0])
    @pytest.mark.parametrize("output", ["dm", "fock_prob"])
    def test_cat_and_loss(self, amplitude, phase, parity, transmission, output):
        """Keep interference, phase and loss-induced mixedness in both public paths."""
        program = sf.Program(1)
        with program.context as q:
            ops.Catstate(amplitude, phi=phase, p=parity) | q[0]
            ops.LossChannel(transmission) | q[0]
        state = sf.Engine("bosonic").run(program).state
        branches, coefficients = cat_components(amplitude, phase, parity)
        branches, weights = attenuated_components(branches[:, None], coefficients, transmission)
        expected = coherent_sum_density(branches, weights, [0], 8)
        before = [array.copy() for array in (state.means(), state.covs(), state.weights())]
        if output == "dm":
            assert_density(state.dm(cutoff=8), expected)
            # Preserve omitted probability rather than renormalizing each component.
            assert_density(state.dm(cutoff=3), expected[:3, :3])
        else:
            actual = [state.fock_prob([n]) for n in range(8)]
            np.testing.assert_allclose(actual, expected.diagonal().real, atol=2e-11, rtol=0)
        for actual, original in zip((state.means(), state.covs(), state.weights()), before):
            np.testing.assert_array_equal(actual, original)

    @pytest.mark.parametrize("parity", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("modes", [[0], [2], [0, 2], [1, 2]])
    @pytest.mark.parametrize("transmission", [[1.0, 1.0, 1.0], [0.4, 0.8, 0.7]])
    def test_correlated_modes_and_reduction(self, parity, modes, transmission):
        """Retain ket/bra axis order for correlated modes and a coherent spectator."""
        amplitude, phase = 0.8, 0.6
        theta, bs_phase = 0.43, 0.37
        beta = 0.2 - 0.3j
        program = sf.Program(3)
        with program.context as q:
            ops.Catstate(amplitude, phi=phase, p=parity) | q[0]
            ops.Dgate(abs(beta), np.angle(beta)) | q[1]
            ops.BSgate(theta, bs_phase) | (q[0], q[2])
            for mode, eta in enumerate(transmission):
                ops.LossChannel(eta) | q[mode]
        state = sf.Engine("bosonic").run(program).state
        alpha, coefficients = cat_components(amplitude, phase, parity)
        branches = np.column_stack(
            [
                alpha * np.cos(theta),
                np.full(2, beta),
                alpha * np.exp(1j * bs_phase) * np.sin(theta),
            ]
        )
        branches, weights = attenuated_components(branches, coefficients, transmission)
        expected = coherent_sum_density(branches, weights, modes, 5)
        assert_density(state.reduced_dm(modes, cutoff=5), expected)
        full = coherent_sum_density(branches, weights, [0, 1, 2], 3)
        for photons in ([0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1]):
            index = tuple(value for n in photons for value in (n, n))
            np.testing.assert_allclose(
                state.fock_prob(photons), full[index].real, atol=2e-11, rtol=0
            )

    @pytest.mark.parametrize("num_modes", [1, 2])
    @pytest.mark.parametrize("preparation", ["vacuum", "coherent", "squeezed", "thermal"])
    def test_real_gaussian_controls(self, num_modes, preparation):
        """Physical Gaussian components retain the established Gaussian result."""

        def run(backend):
            program = sf.Program(num_modes)
            with program.context as q:
                if preparation == "thermal":
                    ops.Thermal(0.25) | q[0]
                if preparation in ("squeezed", "thermal"):
                    ops.Sgate(0.3, 0.7) | q[0]
                if preparation != "vacuum":
                    ops.Dgate(0.2, -0.4) | q[0]
                if num_modes == 2:
                    ops.BSgate(0.4, 0.3) | (q[0], q[1])
            return sf.Engine(backend).run(program).state

        state, control = run("bosonic"), run("gaussian")
        assert_density(state.dm(cutoff=5), control.dm(cutoff=5))
        for photons in itertools.product(range(2), repeat=num_modes):
            np.testing.assert_allclose(
                state.fock_prob(list(photons)),
                control.fock_prob(list(photons)),
                atol=2e-11,
                rtol=0,
            )

    @pytest.mark.parametrize("num_modes", [1, 3])
    @pytest.mark.parametrize("shears", [(0.2, 1.2), (15.0, -17.0)])
    def test_complex_covariances(self, num_modes, shears):
        """Superposed chirped vacua also have complex covariance components."""
        # Position wavefunctions exp[-a*x**2/2], a=(1-i*s)/hbar, yield these
        # Gaussian dyad covariances and overlaps by a one-dimensional integral.
        widths = (1 - 1j * np.asarray(shears)) / sf.hbar
        overlap = np.sqrt(2 / (sf.hbar * (widths[0] + widths[1].conj()))) ** num_modes
        coefficients = np.array([1, 1j]) / np.sqrt(2 + 2 * np.real(-1j * overlap))
        covs, weights = [], []
        for i, j in itertools.product(range(2), repeat=2):
            a, b = widths[i], widths[j].conjugate()
            cov = np.array(
                [
                    [1, 0.5j * sf.hbar * (a - b)],
                    [0.5j * sf.hbar * (a - b), sf.hbar**2 * a * b],
                ]
            ) / (a + b)
            covs.append(block_diag(*([cov] * num_modes)) * 2 / sf.hbar)
            weights.append(
                coefficients[i]
                * coefficients[j].conj()
                * np.sqrt(2 / (sf.hbar * (a + b))) ** num_modes
            )
        state = BaseBosonicState(
            (np.zeros((4, 2 * num_modes)), np.array(covs), np.array(weights)),
            num_modes,
            4,
        )
        # Independent Fock expansion of exp(i*s*x**2/(2*hbar))|0>.
        kets = []
        for shear in shears:
            ket = np.array(
                [
                    (1 - 0.5j * shear) ** -0.5,
                    0,
                    (1 - 0.5j * shear) ** -0.5 * (1j * shear / (2 - 1j * shear)) / np.sqrt(2),
                ]
            )
            multimode = np.array(1)
            for _ in range(num_modes):
                multimode = np.multiply.outer(multimode, ket)
            kets.append(multimode)
        ket = sum(coefficient * component for coefficient, component in zip(coefficients, kets))
        order = np.arange(2 * num_modes).reshape(2, -1).T.flatten()
        expected = np.multiply.outer(ket, ket.conj()).transpose(order)
        assert_density(state.dm(cutoff=3), expected)
        for photons in ([0] * num_modes, [2] + [0] * (num_modes - 1)):
            index = tuple(value for n in photons for value in (n, n))
            np.testing.assert_allclose(
                state.fock_prob(photons), expected[index].real, atol=2e-11, rtol=0
            )


@pytest.mark.bosonic
@pytest.mark.parametrize("hbar", [0.5, 1.0, 1.7, 2.0, 4.0])
@pytest.mark.parametrize("kind", ["cat27", "zero_weight_vacuum"])
def test_zero_weight_components_cannot_poison_fock_output(monkeypatch, hbar, kind):
    """An inactive component must not contaminate probabilities or density tensors."""
    monkeypatch.setattr(sf, "hbar", hbar)
    program = sf.Program(1)
    with program.context as q:
        if kind == "cat27":
            ops.Catstate(27.0) | q[0]
        else:
            ops.Bosonic(
                weights=np.array([1.0, 0.0]),
                means=np.array([[0.0, 0.0], [0.0, 60.0j]]),
                covs=np.array([np.eye(2), np.eye(2)]),
            ) | q[0]
    state = sf.Engine("bosonic").run(program).state
    probability = state.fock_prob([0])
    density = state.dm(cutoff=3)
    assert np.isfinite(probability)
    assert np.isfinite(density).all()
    if kind == "zero_weight_vacuum":
        np.testing.assert_allclose(probability, 1, atol=2e-14, rtol=0)
        expected = np.zeros((3, 3))
        expected[0, 0] = 1
        np.testing.assert_allclose(density, expected, atol=2e-14, rtol=0)


@pytest.mark.bosonic
def test_all_zero_weights_preserve_density_tensor_shape():
    """An empty weighted sum still returns a density tensor of the requested shape."""
    state = BaseBosonicState((np.zeros((1, 4)), np.array([np.eye(4)]), np.array([0.0])), 2, 1)
    np.testing.assert_array_equal(state.dm(cutoff=3), np.zeros((3, 3, 3, 3)))
    np.testing.assert_array_equal(state.reduced_dm(0, cutoff=3), np.zeros((3, 3)))
    assert state.fock_prob([0, 0]) == 0
