# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Tests for the various Tensorflow-specific symbolic options of the frontend/backend.
"""

# pylint: disable=expression-not-assigned,too-many-public-methods,pointless-statement,no-self-use

import pytest
import numpy as np
from scipy.special import factorial

tf = pytest.importorskip("tensorflow", minversion="2.0")

from strawberryfields.ops import (
    Dgate,
    Sgate,
    MeasureX,
    Thermal,
    BSgate,
    MZgate,
    Ket,
    DensityMatrix,
    S2gate,
    Fock,
)

# this test file is only supported by the TF backend
pytestmark = pytest.mark.backends("tf")


ALPHA = 0.5


def coherent_state(alpha, cutoff):
    """Returns the Fock representation of the coherent state |alpha> up to dimension given by cutoff"""
    n = np.arange(cutoff)
    return np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**n / np.sqrt(factorial(n))


def _vac_ket(cutoff):
    """Returns the ket of the vacuum state up to dimension given by cutoff"""
    vac = np.zeros(cutoff)
    vac[0] = 1
    return vac


def _vac_dm(cutoff):
    """Returns the density matrix of the vacuum state up to dimension given by cutoff"""
    vac = _vac_ket(cutoff)
    return np.outer(vac, np.conj(vac))


class TestOneMode:
    """Tests for workflows on one mode."""

    def test_eng_float_parameter_returns_tensor(self, setup_eng):
        """Tests whether eng.run for programs with non-tensor parameters
        successfully returns a Tensor."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        state_data = state.data

        assert isinstance(state_data, tf.Tensor)

    def test_eng_tensor_parameter_returns_tensor(self, setup_eng):
        """Tests whether eng.run for programs with tensor parameters
        successfully returns a Tensor."""
        eng, prog = setup_eng(1)

        alpha = prog.params("alpha")

        with prog.context as q:
            Dgate(alpha) | q

        state = eng.run(prog, args={"alpha": tf.Variable(0.5)}).state
        state_data = state.data

        assert isinstance(state_data, tf.Tensor)

    def test_eng_run_measurements_are_tensors(self, setup_eng):
        """Tests whether eng.run for programs with measurements
        successfully returns a tensors for samples."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q
            MeasureX | q

        res = eng.run(prog)
        assert isinstance(res.samples, tf.Tensor)

    def test_eng_run_state_ket(self, setup_eng, cutoff, pure, tol):
        """Tests whether the ket of the returned state is a
        Tensor object when executed with `eng.run`."""
        if not pure:
            pytest.skip("Tested only for pure states")

        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        ket = state.ket()
        assert isinstance(ket, tf.Tensor)

        coh_ket = coherent_state(ALPHA, cutoff)
        assert np.allclose(ket, coh_ket, atol=tol, rtol=0.0)

    def test_eng_run_state_dm(self, pure, cutoff, setup_eng, tol, batch_size):
        """Tests whether the density matrix of the returned state is an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)
        alpha = ALPHA * np.linspace(1, 0.5, batch_size) if batch_size else ALPHA

        with prog.context as q:
            Dgate(alpha) | q

        state = eng.run(prog).state
        dm = state.dm()
        assert state.is_pure == pure
        assert isinstance(dm, tf.Tensor)

        ket = coherent_state(np.asarray(alpha)[..., None], cutoff)
        coh_dm = np.einsum("...i,...j->...ij", ket, ket.conj())
        assert dm.shape == coh_dm.shape
        assert np.allclose(dm, coh_dm, atol=tol, rtol=0.0)

    def test_eng_run_state_trace(self, setup_eng, tol):
        """Tests whether the trace of the returned state is an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        tr = state.trace()
        assert isinstance(tr, tf.Tensor)
        assert np.allclose(tr, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_reduced_dm(self, setup_eng, cutoff, tol):
        """Tests whether the reduced_density matrix of the returned state
        is a Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        rho = state.reduced_dm([0])
        assert isinstance(rho, tf.Tensor)

        ket = coherent_state(ALPHA, cutoff)
        coh_dm = np.einsum("i,j->ij", ket, ket.conj())
        assert np.allclose(rho, coh_dm, atol=tol, rtol=0.0)

    def test_eng_run_state_fidelity_vacuum(self, setup_eng, tol):
        """Tests whether the fidelity_vacuum method of the state returns an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0) | q

        state = eng.run(prog).state
        fidel_vac = state.fidelity_vacuum()
        assert isinstance(fidel_vac, tf.Tensor)
        assert np.allclose(fidel_vac, 1.0, atol=tol, rtol=0.0)

    def test_eng_run_state_is_vacuum(self, setup_eng):
        """Tests whether the is_vacuum method of the state returns an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(0) | q

        state = eng.run(prog).state
        is_vac = state.is_vacuum()
        assert isinstance(is_vac, tf.Tensor)
        assert np.all(is_vac)

    def test_eng_run_state_fidelity_coherent(self, setup_eng, tol):
        """Tests whether the fidelity of the state with respect to coherent states is
        a Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity_coherent([ALPHA])
        assert isinstance(fidel_coh, tf.Tensor)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_fidelity(self, setup_eng, cutoff, tol):
        """Tests whether the fidelity of the state with respect to a local state is an
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        fidel_coh = state.fidelity(coherent_state(ALPHA, cutoff), 0)
        assert isinstance(fidel_coh, tf.Tensor)
        assert np.allclose(fidel_coh, 1, atol=tol, rtol=0.0)

    def test_eng_run_state_quad_expectation(self, setup_eng, tol, hbar):
        """Tests whether the local quadrature expectation of the state is
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        e, v = state.quad_expectation(0, 0)
        assert isinstance(e, tf.Tensor)
        assert isinstance(v, tf.Tensor)

        true_exp = np.sqrt(hbar / 2.0) * (ALPHA + np.conj(ALPHA))
        true_var = hbar / 2.0
        assert np.allclose(e, true_exp, atol=tol, rtol=0.0)
        assert np.allclose(v, true_var, atol=tol, rtol=0.0)

    def test_eng_run_state_mean_photon(self, setup_eng, tol):
        """Tests whether the local mean photon number of the state is
        Tensor object when executed with `eng.run`."""
        eng, prog = setup_eng(1)

        with prog.context as q:
            Dgate(ALPHA) | q

        state = eng.run(prog).state
        nbar, var = state.mean_photon(0)
        assert isinstance(nbar, tf.Tensor)
        assert isinstance(var, tf.Tensor)

        ref_nbar = np.abs(ALPHA) ** 2
        ref_var = np.abs(ALPHA) ** 2
        assert np.allclose(nbar, ref_nbar, atol=tol, rtol=0.0)
        assert np.allclose(var, ref_var, atol=tol, rtol=0.0)


class TestTwoModes:
    """Tests for workflows on two modes."""

    def test_eval_true_state_all_fock_probs(self, setup_eng, cutoff, batch_size, tol):
        """Tests whether the Fock-basis probabilities of the state return
        a Tensor with the correct value."""
        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        probs = state.all_fock_probs()

        assert isinstance(probs, tf.Tensor)

        probs = probs.numpy().flatten()
        ref_probs = (
            np.abs(
                np.outer(coherent_state(ALPHA, cutoff), coherent_state(-ALPHA, cutoff))
            ).flatten()
            ** 2
        )

        if batch_size is not None:
            ref_probs = np.tile(ref_probs, batch_size)

        assert np.allclose(probs, ref_probs, atol=tol, rtol=0.0)

    def test_eval_true_state_fock_prob(self, setup_eng, cutoff, tol):
        """Tests whether the probability of a Fock measurement outcome on the state returns
        a tensor with the correct value."""
        n1 = cutoff // 2
        n2 = cutoff // 3

        eng, prog = setup_eng(2)

        with prog.context as q:
            Dgate(ALPHA) | q[0]
            Dgate(-ALPHA) | q[1]

        state = eng.run(prog).state
        prob = state.fock_prob([n1, n2])
        assert isinstance(prob, tf.Tensor)

        ref_prob = np.abs(
            np.outer(coherent_state(ALPHA, cutoff), coherent_state(-ALPHA, cutoff)) ** 2
        )[n1, n2]
        assert np.allclose(prob, ref_prob, atol=tol, rtol=0.0)


def _gradient_values(first, last, batch_size):
    """Keep the original scalar input and use distinct inputs when batching."""
    return np.asarray(first) if batch_size is None else np.linspace(first, last, batch_size)


def _two_mode_preparation(occupation, cutoff, pure, batch_size):
    """Prepare the requested storage; Ket alone would force mixed runs to pure."""
    ket = np.zeros((cutoff, cutoff), dtype=np.complex64)
    ket[occupation] = 1
    data = ket if pure else np.einsum("ij,kl->ikjl", ket, ket.conj())
    if batch_size is not None:
        data = np.stack([data] * batch_size)
    return Ket(data) if pure else DensityMatrix(data)


@pytest.fixture
def check_gradient(batch_size, tol, assert_independent_jacobian):
    """Compare values and full Jacobians, including off-diagonal batch entries."""

    def check(tape, actual, expected, derivatives):
        expected = np.asarray(expected)
        assert tuple(actual.shape) == expected.shape
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)
        for parameter, derivative in derivatives:
            assert np.shape(derivative) == expected.shape
            jacobian = tape.jacobian(actual, parameter, experimental_use_pfor=False)
            assert_independent_jacobian(jacobian, derivative, batch_size)

    return check


class TestGradient:
    """Analytic values and derivatives for scalar or independent batched circuits."""

    def test_displacement_mean_photon_gradient(self, setup_eng, tol, batch_size, check_gradient):
        """A coherent state's photon mean and variance both have derivative 2a."""
        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(prog.params("alpha")) | q
        a = _gradient_values(ALPHA, 0.25, batch_size)
        alpha = tf.Variable(a, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"alpha": alpha}).state
            result = tf.stack(state.mean_photon(0), axis=-1)

        assert state.is_pure == eng.backend_options["pure"]
        check_gradient(
            tape,
            result,
            np.stack([a**2, a**2], axis=-1),
            [(alpha, np.stack([2 * a, 2 * a], axis=-1))],
        )

    def test_displaced_thermal_mean_photon_gradient(
        self, setup_eng, tol, batch_size, check_gradient
    ):
        """Check both photon moments against the thermal-state analytic formulas."""
        eng, prog = setup_eng(1)
        with prog.context as q:
            Thermal(prog.params("nbar")) | q
            Dgate(prog.params("alpha")) | q
        a = _gradient_values(0.2, 0.13, batch_size)
        n = _gradient_values(0.052, 0.035, batch_size)
        alpha, nbar = [tf.Variable(value, dtype=tf.float32) for value in (a, n)]

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"nbar": nbar, "alpha": alpha}).state
            result = tf.stack(state.mean_photon(0), axis=-1)

        # Thermal preparation is mixed even when the engine initially uses a ket.
        assert not state.is_pure
        check_gradient(
            tape,
            result,
            np.stack([a**2 + n, n**2 + n + a**2 * (1 + 2 * n)], axis=-1),
            [
                (alpha, np.stack([2 * a, 2 * a * (1 + 2 * n)], axis=-1)),
                (nbar, np.stack([np.ones_like(n), 2 * n + 1 + 2 * a**2], axis=-1)),
            ],
        )

    def test_coherent_ket_gradient(self, setup_eng, cutoff, tol, pure, batch_size, check_gradient):
        """Differentiate the |2> amplitude, preserving its leading batch axis."""
        if not pure:
            pytest.skip("Test skipped in mixed state mode")
        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(prog.params("alpha")) | q
        a = _gradient_values(ALPHA, 0.25, batch_size)
        alpha = tf.Variable(a, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"alpha": alpha}).state
            result = tf.math.real(state.ket()[..., 2])

        assert state.is_pure
        check_gradient(
            tape,
            result,
            np.exp(-(a**2) / 2) * a**2 / np.sqrt(2),
            [(alpha, -a * (a**2 - 2) * np.exp(-(a**2) / 2) / np.sqrt(2))],
        )

    def test_coherent_dm_gradient(self, setup_eng, cutoff, tol, batch_size, check_gradient):
        """Differentiate the |2><2| population in both state representations."""
        eng, prog = setup_eng(1)
        with prog.context as q:
            Dgate(prog.params("alpha")) | q
        a = _gradient_values(ALPHA, 0.25, batch_size)
        alpha = tf.Variable(a, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"alpha": alpha}).state
            result = tf.math.real(state.dm()[..., 2, 2])

        assert state.is_pure == eng.backend_options["pure"]
        check_gradient(
            tape,
            result,
            np.exp(-(a**2)) * a**4 / 2,
            [(alpha, -(a**3) * (a**2 - 2) * np.exp(-(a**2)))],
        )

    def test_displaced_squeezed_mean_photon_gradient(
        self, setup_eng, cutoff, tol, batch_size, check_gradient
    ):
        """Check displacement, squeezing and phase derivatives of photon mean."""
        eng, prog = setup_eng(1)
        with prog.context as q:
            Sgate(prog.params("r"), prog.params("phi")) | q
            Dgate(prog.params("a")) | q
        a = _gradient_values(ALPHA, 0.25, batch_size)
        r = _gradient_values(0.105, 0.08, batch_size)
        phi = _gradient_values(0.123, 0.31, batch_size)
        alpha, squeezing, phase = [tf.Variable(value, dtype=tf.float32) for value in (a, r, phi)]

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"a": alpha, "r": squeezing, "phi": phase}).state
            mean, _ = state.mean_photon(0)

        assert state.is_pure == eng.backend_options["pure"]
        check_gradient(
            tape,
            mean,
            a**2 + np.sinh(r) ** 2,
            [(alpha, 2 * a), (squeezing, 2 * np.sinh(r) * np.cosh(r)), (phase, np.zeros_like(phi))],
        )

    def test_BS_state_gradients(self, setup_eng, cutoff, tol, batch_size, check_gradient):
        """Check |1,1> and |0,2> probabilities after interfering two photons."""
        eng, prog = setup_eng(2)
        with prog.context as q:
            _two_mode_preparation((1, 1), cutoff, eng.backend_options["pure"], batch_size) | q
            BSgate(prog.params("theta"), prog.params("phi")) | q
        theta = _gradient_values(0.3, 0.19, batch_size)
        phi = _gradient_values(0.2, -0.15, batch_size)
        angle, phase = [tf.Variable(value, dtype=tf.float32) for value in (theta, phi)]

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"theta": angle, "phi": phase}).state
            result = tf.stack([state.fock_prob([1, 1]), state.fock_prob([0, 2])], axis=-1)

        assert state.is_pure == eng.backend_options["pure"]
        expected = np.stack([np.cos(2 * theta) ** 2, np.sin(2 * theta) ** 2 / 2], axis=-1)
        check_gradient(
            tape,
            result,
            expected,
            [
                (
                    angle,
                    np.stack(
                        [-4 * np.sin(2 * theta) * np.cos(2 * theta), np.sin(4 * theta)], axis=-1
                    ),
                ),
                (phase, np.zeros_like(expected)),
            ],
        )

    def test_MZ_state_gradients(self, setup_eng, cutoff, tol, batch_size, check_gradient):
        """Check both phase derivatives through a Mach-Zehnder interferometer."""
        eng, prog = setup_eng(2)
        with prog.context as q:
            _two_mode_preparation((1, 1), cutoff, eng.backend_options["pure"], batch_size) | q
            MZgate(prog.params("phi_in"), prog.params("phi_ex")) | q
        phi_in = _gradient_values(0.3, 0.19, batch_size)
        phi_ex = _gradient_values(0.2, -0.15, batch_size)
        internal, external = [tf.Variable(value, dtype=tf.float32) for value in (phi_in, phi_ex)]

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"phi_in": internal, "phi_ex": external}).state
            result = tf.stack([state.fock_prob([1, 1]), state.fock_prob([0, 2])], axis=-1)

        assert state.is_pure == eng.backend_options["pure"]
        expected = np.stack([np.cos(phi_in) ** 2, np.sin(phi_in) ** 2 / 2], axis=-1)
        check_gradient(
            tape,
            result,
            expected,
            [
                (
                    internal,
                    np.stack([-np.sin(2 * phi_in), np.sin(phi_in) * np.cos(phi_in)], axis=-1),
                ),
                (external, np.zeros_like(expected)),
            ],
        )

    def test_2mode_squeezed_vacuum_gradients(
        self, setup_eng, cutoff, tol, batch_size, check_gradient
    ):
        """Check the vacuum and paired-photon probabilities after an S2gate."""
        eng, prog = setup_eng(2)
        with prog.context as q:
            _two_mode_preparation((0, 0), cutoff, eng.backend_options["pure"], batch_size) | q
            S2gate(prog.params("r"), prog.params("phi")) | q
        r = _gradient_values(0.3, 0.19, batch_size)
        phi = _gradient_values(0.2, -0.15, batch_size)
        squeezing, phase = [tf.Variable(value, dtype=tf.float32) for value in (r, phi)]

        with tf.GradientTape(persistent=True) as tape:
            state = eng.run(prog, args={"r": squeezing, "phi": phase}).state
            result = tf.stack([state.fock_prob([0, 0]), state.fock_prob([1, 1])], axis=-1)

        assert state.is_pure == eng.backend_options["pure"]
        expected = np.stack([1 / np.cosh(r) ** 2, np.sinh(r) ** 2 / np.cosh(r) ** 4], axis=-1)
        check_gradient(
            tape,
            result,
            expected,
            [
                (
                    squeezing,
                    np.stack(
                        [
                            -2 * np.tanh(r) / np.cosh(r) ** 2,
                            2 * (np.sinh(r) - np.sinh(r) ** 3) / np.cosh(r) ** 5,
                        ],
                        axis=-1,
                    ),
                ),
                (phase, np.zeros_like(expected)),
            ],
        )
