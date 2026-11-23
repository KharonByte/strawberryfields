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
r"""Unit tests for the states.py fock probabilities methods"""

import pytest

import numpy as np
from scipy.special import factorial as fac

MAG_ALPHAS = np.linspace(0, 0.8, 3)
PHASE_ALPHAS = np.linspace(0, 2 * np.pi, 3, endpoint=False)


def assert_probabilities(actual, expected, expected_shape, tol):
    """Check requested axes before allowing NumPy's value broadcasting."""
    actual = np.asarray(actual)
    assert actual.shape == expected_shape
    assert np.allclose(actual, expected, atol=tol, rtol=0)


@pytest.mark.frontend
@pytest.mark.parametrize(
    "corruption",
    [None, "wrong_axis", "flat", "extra_axis", "missing_member", "swapped", "duplicated", "value"],
)
def test_probability_array_assertions(corruption):
    """Equal flattened contents must not hide wrong axes or lost batch members."""
    expected = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    actual = expected.copy()
    if corruption == "wrong_axis":
        actual = actual.reshape(3, 2)
    elif corruption == "flat":
        actual = actual.flatten()
    elif corruption == "extra_axis":
        actual = actual[None, ...]
    elif corruption == "missing_member":
        actual = actual[:1]
    elif corruption == "swapped":
        actual = actual[::-1]
    elif corruption == "duplicated":
        actual = np.repeat(actual[:1], 2, axis=0)
    elif corruption == "value":
        actual[1, 0] += 1
    if corruption is None:
        assert_probabilities(actual, expected, (2, 3), 0.001)
    else:
        with pytest.raises(AssertionError):
            assert_probabilities(actual, expected, (2, 3), 0.001)


@pytest.mark.frontend
@pytest.mark.parametrize("corruption", [None, "scalar", "one_member", "wrong_second"])
def test_probability_scalar_reference_assertions(corruption):
    """A scalar reference still requires every requested batch output."""
    actual = np.array([0.25, 0.25])
    if corruption == "scalar":
        actual = 0.25
    elif corruption == "one_member":
        actual = actual[:1]
    elif corruption == "wrong_second":
        actual[1] += 1
    if corruption is None:
        assert_probabilities(actual, 0.25, (2,), 0.001)
    else:
        with pytest.raises(AssertionError):
            assert_probabilities(actual, 0.25, (2,), 0.001)


@pytest.mark.parametrize("a", MAG_ALPHAS)
@pytest.mark.parametrize("phi", PHASE_ALPHAS)
class TestFockProbabilities:
    """Tests for the fock_prob state method"""

    def test_gaussian(self, a, phi, setup_backend, cutoff, batch_size, tol):
        """Tests that probabilities of particular Fock states
        |n> are correct for a gaussian state."""
        backend = setup_backend(1)

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(np.abs(alpha), np.angle(alpha), 0)
        state = backend.state()
        shape = (batch_size,) if backend.short_name == "tf" and batch_size is not None else ()

        for n in range(cutoff):
            prob_n = state.fock_prob([n])
            assert_probabilities(prob_n, ref_probs[n], shape, tol)

    @pytest.mark.backends("fock", "tf")
    def test_nongaussian(self, a, phi, setup_backend, cutoff, batch_size, tol):
        """Tests that probabilities of particular Fock states |n> are
        correct for a nongaussian state."""
        backend = setup_backend(2)

        alpha = a * np.exp(1j * phi)
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(np.abs(alpha), np.angle(alpha), 0)
        backend.prepare_fock_state(cutoff // 2, 1)
        state = backend.state()
        shape = (batch_size,) if backend.short_name == "tf" and batch_size is not None else ()

        for n in range(cutoff):
            prob_n = state.fock_prob([n, cutoff // 2])
            assert_probabilities(prob_n, ref_probs[n], shape, tol)


@pytest.mark.backends("fock", "tf", "gaussian")
@pytest.mark.parametrize("a", MAG_ALPHAS)
@pytest.mark.parametrize("phi", PHASE_ALPHAS)
class TestAllFockProbs:
    """Tests for the all_fock_probs state method"""

    def test_pure(self, a, phi, setup_backend, cutoff, batch_size, tol):
        """Tests that the numeric probabilities in the full Fock basis are
        correct for a one-mode pure state."""
        backend = setup_backend(1)

        alpha = a * np.exp(1j * phi)
        batched = backend.short_name == "tf" and batch_size is not None
        if batched:
            alpha = alpha * np.linspace(1, 0.5, batch_size)
        amplitudes = np.asarray(alpha)[..., None]
        n = np.arange(cutoff)
        ref_state = np.exp(-0.5 * np.abs(amplitudes) ** 2) * amplitudes**n / np.sqrt(fac(n))
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(np.abs(alpha), np.angle(alpha), 0)
        state = backend.state()

        shape = (batch_size, cutoff) if batched else (cutoff,)
        assert_probabilities(state.all_fock_probs(cutoff=cutoff), ref_probs, shape, tol)

    def test_two_mode_gaussian(self, a, phi, setup_backend, batch_size, cutoff, tol):
        """Tests that the numeric probabilities in the full Fock basis are
        correct for a two-mode gaussian state."""
        backend = setup_backend(2)

        alpha = a * np.exp(1j * phi)
        if backend.short_name == "tf" and batch_size is not None:
            alpha = alpha * np.linspace(1, 0.5, batch_size)
        alpha = np.asarray(alpha)[..., None]

        n = np.arange(cutoff)
        ref_state1 = np.exp(-0.5 * np.abs(alpha) ** 2) * alpha**n / np.sqrt(fac(n))
        ref_state2 = np.exp(-0.5 * np.abs(-alpha) ** 2) * (-alpha) ** n / np.sqrt(fac(n))

        ref_state = ref_state1[..., :, None] * ref_state2[..., None, :]
        ref_probs = np.abs(ref_state) ** 2

        backend.prepare_coherent_state(np.abs(alpha[..., 0]), np.angle(alpha[..., 0]), 0)
        backend.prepare_coherent_state(np.abs(alpha[..., 0]), np.angle(alpha[..., 0]) + np.pi, 1)
        state = backend.state()

        probs = np.asarray(state.all_fock_probs(cutoff=cutoff))
        assert probs.shape == ref_probs.shape
        assert np.allclose(probs, ref_probs, atol=tol, rtol=0)
