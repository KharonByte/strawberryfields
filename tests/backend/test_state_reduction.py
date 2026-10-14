# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Reduced-state contracts checked against analytic optical states."""

import numpy as np
import pytest
from scipy.special import factorial

import strawberryfields as sf
from strawberryfields import ops


@pytest.fixture(params=[1.0, 1.7, 2.0])
def quadrature_units(request, monkeypatch):
    """Photon probabilities and density tensors do not depend on quadrature units."""
    monkeypatch.setattr(sf, "hbar", request.param)


def coherent_density(alpha, cutoff):
    """Analytic Fock coefficients, retaining the omitted probability tail."""
    n = np.arange(cutoff)
    ket = np.exp(-abs(alpha) ** 2 / 2) * alpha**n / np.sqrt(factorial(n))
    return np.outer(ket, ket.conj())


def assert_density(actual, expected, state):
    """Check values and interleaved ket/bra axes, including TensorFlow batches."""
    actual = np.asarray(actual)
    if getattr(state, "batched", False):
        expected = np.broadcast_to(expected, (state.data.shape[0],) + expected.shape)
    assert actual.shape == expected.shape
    # TensorFlow defaults to complex64; the NumPy backends use complex128.
    tolerance = 5e-7 if actual.dtype == np.complex64 else 1e-10
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)


@pytest.mark.usefixtures("quadrature_units")
class TestStateReduction:
    """Exercise the same public circuits on Gaussian, bosonic, Fock and TF."""

    @pytest.mark.parametrize("squeezing", [0.0, 0.4, 0.8])
    @pytest.mark.parametrize("mode", [0, 1])
    def test_entangled_marginal_is_thermal(self, setup_eng, cutoff, squeezing, mode):
        """Tracing either half of a squeezed pair gives a thermal distribution."""
        engine, program = setup_eng(2)
        with program.context as q:
            ops.S2gate(squeezing, 0.37) | (q[0], q[1])
        state = engine.run(program).state
        nbar = np.sinh(squeezing) ** 2
        expected = np.diag(nbar ** np.arange(cutoff) / (1 + nbar) ** (np.arange(cutoff) + 1))
        assert_density(state.reduced_dm(mode, cutoff=cutoff), expected, state)

    @pytest.mark.parametrize("modes", [[0, 1], [0, 2]])
    def test_retained_entangled_pair_tensor(self, setup_eng, cutoff, modes):
        """A complete pure pair keeps its coherences and one ket/bra pair per mode."""
        engine, program = setup_eng(max(modes) + 1)
        squeezing, phase = 0.25, 0.37
        with program.context as q:
            ops.S2gate(squeezing, phase) | tuple(q[m] for m in modes)
        state = engine.run(program).state
        coefficients = (np.exp(1j * phase) * np.tanh(squeezing)) ** np.arange(cutoff)
        ket = np.diag(coefficients / np.cosh(squeezing))
        expected = np.einsum("ab,cd->acbd", ket, ket.conj())
        assert_density(state.reduced_dm(modes, cutoff=cutoff), expected, state)
        if modes == [0, 1]:
            assert_density(state.dm(cutoff=cutoff), expected, state)

    @pytest.mark.parametrize("modes", [[0, 1], [1, 2]])
    def test_mixed_marginal_with_coherent_spectator(self, setup_eng, cutoff, modes):
        """Keep one half of a pair and an independent coherent mode, in index order."""
        engine, program = setup_eng(3)
        squeezing, alpha = 0.25, 0.2 + 0.15j
        with program.context as q:
            ops.S2gate(squeezing, 0.37) | (q[0], q[2])
            ops.Dgate(abs(alpha), np.angle(alpha)) | q[1]
        state = engine.run(program).state
        nbar = np.sinh(squeezing) ** 2
        thermal = np.diag(nbar ** np.arange(cutoff) / (1 + nbar) ** (np.arange(cutoff) + 1))
        coherent = coherent_density(alpha, cutoff)
        first, second = (thermal, coherent) if modes == [0, 1] else (coherent, thermal)
        expected = np.einsum("ab,cd->abcd", first, second)
        assert_density(state.reduced_dm(modes, cutoff=cutoff), expected, state)

    def test_mixed_parent_can_have_pure_subsystem(self, setup_eng, cutoff):
        """A discarded thermal mode must not mix the independent retained modes."""
        engine, program = setup_eng(3)
        nbar, alpha = 0.3, 0.2 + 0.15j
        with program.context as q:
            ops.Thermal(nbar) | q[0]
            ops.Dgate(abs(alpha), np.angle(alpha)) | q[2]
        state = engine.run(program).state
        vacuum = np.zeros((cutoff, cutoff))
        vacuum[0, 0] = 1
        expected = np.einsum("ab,cd->abcd", vacuum, coherent_density(alpha, cutoff))
        if engine.backend.short_name in ("fock", "tf"):
            # Fock simulators also truncate the mode being traced out.
            expected *= 1 - (nbar / (1 + nbar)) ** cutoff
        assert_density(state.reduced_dm([1, 2], cutoff=cutoff), expected, state)

    def test_photon_probabilities_of_squeezed_pair(self, setup_eng):
        """The same quadrature ordering must be used for matrix elements and probabilities."""
        engine, program = setup_eng(2)
        squeezing = 0.25
        with program.context as q:
            ops.S2gate(squeezing, 0.37) | (q[0], q[1])
        state = engine.run(program).state
        tolerance = 5e-7 if engine.backend.short_name == "tf" else 1e-10
        np.testing.assert_allclose(
            state.fock_prob([1, 1]),
            np.tanh(squeezing) ** 2 / np.cosh(squeezing) ** 2,
            atol=tolerance,
            rtol=0,
        )
        np.testing.assert_allclose(state.fock_prob([1, 0]), 0, atol=tolerance, rtol=0)

    def test_coherent_marginal_retains_cutoff_tail(self, setup_eng, cutoff):
        """A truncated density matrix must not be silently renormalised to trace one."""
        engine, program = setup_eng(2)
        alpha = 1.1 + 0.4j
        with program.context as q:
            ops.Dgate(abs(alpha), np.angle(alpha)) | q[0]
        state = engine.run(program).state
        expected = coherent_density(alpha, cutoff)
        assert np.trace(expected).real < 0.999
        assert_density(state.reduced_dm([0], cutoff=cutoff), expected, state)


@pytest.mark.backends("tf")
def test_tensorflow_reduction_batch_gradient(setup_eng):
    """Distinct squeezed pairs retain separate batch values and analytic gradients."""
    tf = pytest.importorskip("tensorflow")
    squeezing = tf.Variable([0.2, 0.4], dtype=tf.float64)
    engine, program = setup_eng(2, cutoff_dim=6, batch_size=2, dtype=tf.complex128)
    with program.context as q:
        ops.S2gate(squeezing) | (q[0], q[1])
    with tf.GradientTape() as tape:
        state = engine.run(program).state
        reduced = state.reduced_dm([0])
        vacuum = tf.math.real(reduced[:, 0, 0])
        objective = tf.reduce_sum(vacuum)
    gradient = tape.gradient(objective, squeezing)
    assert reduced.shape == (2, 6, 6)
    assert gradient is not None
    expected = 1 / np.cosh(squeezing.numpy()) ** 2
    np.testing.assert_allclose(vacuum.numpy(), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        gradient.numpy(), -2 * np.tanh(squeezing.numpy()) * expected, atol=1e-12, rtol=0
    )
