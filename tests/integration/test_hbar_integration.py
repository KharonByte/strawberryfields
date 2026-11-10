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
r"""Integration tests to make sure hbar values are set correctly in returned results"""

import pytest

import numpy as np

import strawberryfields as sf
from strawberryfields import ops

N_MEAS = 200

R = 0.3
X = 0.2
P = 0.1
HBAR = [0.5, 1]


@pytest.mark.parametrize("hbar", HBAR, indirect=True)
class TestIntegration:
    """Tests that go through the frontend and backend, and state object"""

    def test_squeeze_variance_frontend(self, setup_eng, hbar, batch_size, assert_normal_variance):
        """test homodyne measurement of a squeeze state is correct,
        returning a variance of np.exp(-2*r)*h/2, via the frontend"""
        eng, prog = setup_eng(1)

        with prog.context as q:
            ops.Sgate(R) | q
            ops.MeasureX | q

        res = []

        for i in range(N_MEAS):
            eng.run(prog)
            res.append(np.asarray(q[0].val).reshape(-1))
            eng.reset()

        # Each batch member must independently have the physical squeezed variance.
        samples = np.stack(res)
        assert samples.shape == (N_MEAS, batch_size or 1)
        assert_normal_variance(samples, np.exp(-2 * R) * hbar / 2)

    @pytest.mark.backends("fock", "tf", "gaussian", "bosonic")
    def test_x_displacement(self, setup_eng, hbar, tol, batch_size):
        """Check physical x displacement and every TensorFlow batch member."""
        eng, prog = setup_eng(1)
        batched = eng.backend_name == "tf" and batch_size is not None
        displacement = np.linspace(-X, X, batch_size) if batched else X

        with prog.context as q:
            ops.Xgate(displacement) | q

        state = eng.run(prog).state
        if eng.backend_name == "gaussian":
            mu_x = state.means()[0]
        elif eng.backend_name == "bosonic":
            mu_x = state.means()[0, 0]
        else:
            mu_x, variance = state.quad_expectation(0, 0)
            shape = (batch_size,) if batched else ()
            assert np.shape(mu_x) == np.shape(variance) == shape
            assert np.allclose(variance, hbar / 2, atol=tol, rtol=0)

        assert state.hbar == hbar
        assert np.allclose(mu_x, displacement, atol=tol, rtol=0)

    @pytest.mark.backends("fock", "tf", "gaussian", "bosonic")
    def test_z_displacement(self, setup_eng, hbar, tol, batch_size):
        """Check physical p displacement and every TensorFlow batch member."""
        eng, prog = setup_eng(1)
        batched = eng.backend_name == "tf" and batch_size is not None
        displacement = np.linspace(-P, P, batch_size) if batched else P

        with prog.context as q:
            ops.Zgate(displacement) | q

        state = eng.run(prog).state
        if eng.backend_name == "gaussian":
            mu_z = state.means()[1]
        elif eng.backend_name == "bosonic":
            mu_z = state.means()[0, 1]
        else:
            mu_z, variance = state.quad_expectation(0, np.pi / 2)
            shape = (batch_size,) if batched else ()
            assert np.shape(mu_z) == np.shape(variance) == shape
            assert np.allclose(variance, hbar / 2, atol=tol, rtol=0)

        assert state.hbar == hbar
        assert np.allclose(mu_z, displacement, atol=tol, rtol=0)


@pytest.mark.parametrize("hbar", HBAR, indirect=True)
class BackendOnly:
    """Tests that ignore the frontend. This test is currently not run."""

    def test_squeeze_variance(self, setup_backend, hbar, pure, assert_normal_variance):
        """test homodyne measurement of a squeeze state is correct,
        returning a variance of np.exp(-2*r) in the backend's normalized units"""
        # TODO: this test is a backend test that duplicates
        # the existing `test_squeeze_variance` integration test.
        # It should live in the backend folder, but currently takes too
        # long to run both.
        # We should revisit this test and decide what to do with it.
        backend = setup_backend(1)

        res = []

        for i in range(N_MEAS):
            backend.reset(pure=pure)
            backend.squeeze(R, 0, 0)
            x = backend.measure_homodyne(phi=0, mode=0)
            res.append(np.asarray(x).reshape(-1))

        # The backend API uses hbar=2; the frontend applies the physical-unit scaling.
        assert_normal_variance(np.stack(res), np.exp(-2 * R))
