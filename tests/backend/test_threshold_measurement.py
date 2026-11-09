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

r"""Unit tests for measurements in the Fock basis"""

import pytest

import numpy as np

NUM_REPEATS = 50


@pytest.mark.backends("gaussian")
class TestGaussianRepresentation:
    """Tests that make use of the Fock basis representation."""

    def measure_threshold_gaussian_warning(self, setup_backend):
        """Tests the warning message when MeasureThreshold is called."""

        backend = setup_backend(3)

        with pytest.warns(
            Warning,
            match="Cannot simulate non-Gaussian states. Conditional state after "
            "Threshold measurement has not been updated.",
        ):
            backend.measure_threshold([0, 1], shots=5)


@pytest.mark.backends("gaussian", "bosonic")
class TestRepresentationIndependent:
    """Basic implementation-independent tests."""

    def test_two_mode_squeezed_measurements(self, setup_backend, pure):
        """Tests Threshold measurement on the two mode squeezed vacuum state."""
        for _ in range(NUM_REPEATS):
            backend = setup_backend(2)
            backend.reset(pure=pure)

            r = 0.25
            # Circuit to prepare two mode squeezed vacuum
            backend.squeeze(r, np.pi, 0)
            backend.squeeze(r, 0, 1)
            backend.beamsplitter(np.pi / 4, np.pi, 0, 1)
            meas_modes = [0, 1]
            meas_results = backend.measure_threshold(meas_modes)
            assert np.all(meas_results[0][0] == meas_results[0][1])

    def test_vacuum_measurements(self, setup_backend, pure):
        """Tests Threshold measurement on the vacuum state."""
        backend = setup_backend(3)

        for _ in range(NUM_REPEATS):
            backend.reset(pure=pure)

            meas = backend.measure_threshold([0, 1, 2])[0]
            assert np.all(np.array(meas) == 0)

    @pytest.mark.parametrize("alpha", [0, 1, 0.5 + 0.5j])
    def test_binary_outcome(self, setup_backend, pure, alpha):
        """Test that the outcomes of a threshold measurement are zero or one."""
        num_modes = 2
        for _ in range(NUM_REPEATS):
            backend = setup_backend(num_modes)
            backend.reset(pure=pure)

            r = 0.5

            backend.squeeze(r, 0, 0)
            backend.displacement(abs(alpha), np.angle(alpha), 0)
            backend.beamsplitter(np.pi / 4, np.pi, 0, 1)
            meas_modes = [0, 1]
            meas_results = backend.measure_threshold(meas_modes)

            for i in range(num_modes):
                assert meas_results[0][i] == 0 or meas_results[0][i] == 1

    @pytest.mark.parametrize("alpha", [0, 1, 0.5 + 0.5j])
    def test_single_mode_measurement(self, setup_backend, pure, alpha):
        """Test that the outcomes of a threshold measurement are zero or one."""
        num_modes = 1
        for _ in range(NUM_REPEATS):
            backend = setup_backend(num_modes)
            backend.reset(pure=pure)

            r = 0.5
            backend.squeeze(r, 0, 0)
            backend.displacement(abs(alpha), np.angle(alpha), 0)
            meas_modes = [0]
            meas_results = backend.measure_threshold(meas_modes)

            for i in range(num_modes):
                assert meas_results[0][i] == 0 or meas_results[0][i] == 1


@pytest.mark.gaussian
@pytest.mark.parametrize("select", [None, 0, 1, [0], [1]])
@pytest.mark.parametrize("shots", [1, 2])
def test_gaussian_threshold_selection_contract(monkeypatch, select, shots):
    """Reject unsupported selection before sampling; retain unselected dispatch.

    Direct scalar selection cases are defensive checks; the documented backend
    input is a sequence. The frontend normalizes supported scalar selections.
    This controlled sampler checks the API, not the detector distribution.
    """
    import warnings

    import strawberryfields as sf
    from strawberryfields import ops
    from strawberryfields.backends.gaussianbackend import backend as gaussian

    monkeypatch.setattr(sf, "hbar", 1.7)
    preparation = sf.Program(2)
    with preparation.context as q:
        ops.Sgate(0.2, 0.3) | q[0]
        ops.Sgate(0.1, -0.4) | q[1]
        ops.Dgate(0.23, 0.4) | q[0]
        ops.Dgate(0.17, -0.2) | q[1]
        ops.BSgate(0.31, 0.27) | (q[0], q[1])
    engine = sf.Engine("gaussian")
    engine.run(preparation)
    backend = engine.backend
    modes = [1, 0] if select is None else [1]
    indices = modes + [m + 2 for m in modes]
    expected_mean = backend.circuit.smeanxp()[indices]
    expected_cov = backend.circuit.scovmatxp()[np.ix_(indices, indices)]
    value = select[0] if isinstance(select, list) else select
    output = (
        np.array([[0, 1], [1, 0]])[:shots]
        if select is None
        else np.full((shots, 1), 1 - value, dtype=int)
    )
    calls = []

    def sampler(**kwargs):
        calls.append(kwargs)
        return output.copy()

    def snapshot():
        state = backend.state()
        return [
            backend.circuit.mean.copy(),
            backend.circuit.nmat.copy(),
            backend.circuit.mmat.copy(),
            np.array(backend.circuit.active),
            state.means().copy(),
            state.cov().copy(),
        ]

    monkeypatch.setattr(gaussian, "torontonian_sample_state", sampler)
    before = snapshot()
    # Direct backend selection is sequence-valued in the public contract.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if select is not None:
            with pytest.raises(NotImplementedError, match="postselection"):
                backend.measure_threshold(modes, shots=shots, select=select)
            assert not calls
            assert not caught
        else:
            result = backend.measure_threshold(modes, shots=shots, select=select)
            np.testing.assert_array_equal(result, output)
            assert len(calls) == 1
            assert calls[0]["samples"] == shots
            np.testing.assert_array_equal(calls[0]["mu"], expected_mean)
            np.testing.assert_array_equal(calls[0]["cov"], expected_cov)
            assert len(caught) == (shots == 2)
            if shots == 2:
                assert str(caught[0].message) == (
                    "Cannot simulate non-Gaussian states. "
                    "Conditional state after Threshold measurement has not been updated."
                )
    for initial, final in zip(before, snapshot()):
        np.testing.assert_array_equal(final, initial)
    assert sf.hbar == backend.state().hbar == 1.7
