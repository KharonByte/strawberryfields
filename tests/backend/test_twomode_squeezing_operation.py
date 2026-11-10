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
Unit tests for two-mode squeezing operation
Convention: The squeezing unitary is fixed to be
U(z) = \exp(0.5 (z^* \hat{a}_1\hat{a}_2 - z (\hat{a^\dagger}_1\hat{a}_2)))
where \hat{a} is the photon annihilation operator.
"""

# pylint: disable=too-many-arguments
import itertools as it

import pytest
import numpy as np

PHASE = np.linspace(0, 2 * np.pi, 3, endpoint=False)
MAG = np.linspace(0.0, 0.2, 5, endpoint=False)
MODES = list(it.combinations(range(4), 2))


def get_amplitude(k, r, p):
    """Get amplitude for two mode squeezing operation

    Args:
        k (tuple): fock state numbers for two modes
        r (float): two-mode squeezing magnitude
        p (float): two-mode squeezing phase

    Returns:
        float: the two-mode squeezed vacuum amplitude
    """
    if k[0] == k[1]:
        tmsv = (np.exp(1j * p) * np.tanh(r)) ** k[0] / np.cosh(r)
        return tmsv
    return 0.0


@pytest.mark.backends("fock")
class TestTwomodeSqueezing:
    """Tests two-mode squeezing operation"""

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("p", PHASE)
    @pytest.mark.backends("fock", "tf")
    def test_two_mode_squeezing(self, setup_backend, r, p, cutoff, pure, tol, batch_size):
        r"""Test two-mode squeezing on vacuum-state for both pure states and
        mixed states with the amplitude given by
                :math:`\delta_{kl} \frac{e^{in\phi} \tanh^n{r}}{\cosh{r}}`
        """

        backend = setup_backend(2)
        backend.two_mode_squeeze(r, p, 0, 1)

        state = backend.state()

        ket = np.array([get_amplitude(k, r, p) for k in it.product(range(cutoff), repeat=2)])
        ket = ket.reshape(cutoff, cutoff)
        expected = ket if pure else np.einsum("ij,kl->ikjl", ket, ket.conj())
        if batch_size is not None:
            expected = np.broadcast_to(expected, (batch_size,) + expected.shape)
        actual = np.asarray(state.data)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf")
    def test_distinct_squeezing_inputs(self, setup_backend, cutoff, pure, tol, batch_size):
        """A batch must preserve each member's independent squeezing parameters."""
        magnitudes = np.linspace(0.06, 0.16, batch_size or 1)
        phases = np.linspace(0.2, 2.0, batch_size or 1)
        backend = setup_backend(2)
        backend.two_mode_squeeze(
            magnitudes if batch_size else magnitudes[0],
            phases if batch_size else phases[0],
            0,
            1,
        )
        expected = []
        for magnitude, phase in zip(magnitudes, phases):
            ket = np.diag(
                (np.exp(1j * phase) * np.tanh(magnitude)) ** np.arange(cutoff) / np.cosh(magnitude)
            )
            expected.append(ket if pure else np.einsum("ij,kl->ikjl", ket, ket.conj()))
        expected = np.stack(expected) if batch_size else expected[0]
        actual = np.asarray(backend.state().data)
        assert actual.shape == expected.shape
        if batch_size and batch_size > 1:
            assert not np.allclose(expected[0], expected[-1], atol=tol, rtol=0)
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("r", MAG)
    @pytest.mark.parametrize("p", PHASE)
    @pytest.mark.parametrize("modes", MODES)
    @pytest.mark.backends("fock", "tf")
    def test_squeezing_on_mode_subset(self, setup_backend, r, p, modes, cutoff, batch_size, tol):
        r"""Test two-mode squeezing on vacuum-state for both pure states and
        mixed states on different mode subsets with the amplitude given by
                :math:`\delta_{kl} \frac{e^{in\phi} \tanh^n{r}}{\cosh{r}}`
        """

        backend = setup_backend(4)
        backend.two_mode_squeeze(r, p, *modes)

        state = backend.state()

        state_data = state.reduced_dm(list(modes), cutoff=cutoff)
        amplitudes = np.array(
            [[get_amplitude((i, j), r, p) for j in range(cutoff)] for i in range(cutoff)]
        )
        expected = np.einsum("ij,kl->ikjl", amplitudes, amplitudes.conj())
        expected_shape = ((batch_size,) if batch_size is not None else ()) + expected.shape
        assert state_data.shape == expected_shape
        assert np.allclose(state_data, expected, atol=tol, rtol=0)
