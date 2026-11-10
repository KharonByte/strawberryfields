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
"""
Default parameters, environment variables, fixtures, and common routines for the unit tests.
"""

# pylint: disable=redefined-outer-name
import os
import pytest

import numpy as np
from scipy.stats import chi2

import strawberryfields as sf
from strawberryfields.engine import LocalEngine
from strawberryfields.program import Program
from strawberryfields.backends.base import BaseBackend
from strawberryfields.backends.fockbackend import FockBackend
from strawberryfields.backends.gaussianbackend import GaussianBackend
from strawberryfields.backends.bosonicbackend import BosonicBackend
from strawberryfields.backends.states import BaseGaussianState, BaseBosonicState

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError) as e:
    tf_available = False
    tf_version = False
else:
    tf_available = True
    tf_version = tf.__version__


backend_params = [
    pytest.param(FockBackend, marks=pytest.mark.fock),
    pytest.param(GaussianBackend, marks=pytest.mark.gaussian),
    pytest.param(BosonicBackend, marks=pytest.mark.bosonic),
]


eng_backend_params = [
    pytest.param("fock", marks=pytest.mark.fock),
    pytest.param("gaussian", marks=pytest.mark.gaussian),
    pytest.param("bosonic", marks=pytest.mark.bosonic),
]


if tf_available and tf.__version__[:2] == "2.":
    from strawberryfields.backends.tfbackend import TFBackend

    backend_params.append(pytest.param(TFBackend, marks=pytest.mark.tf))
    eng_backend_params.append(pytest.param("tf", marks=pytest.mark.tf))
else:
    tf_available = False


# defaults
TOL = 1e-3
CUTOFF = 6
ALPHA = 0.1
HBAR = 1.7
BATCHED = False
BATCHSIZE = 2


@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))


@pytest.fixture(scope="session")
def cutoff(request):
    """Fock cutoff: explicit test parameters override the environment/default."""
    return int(getattr(request, "param", os.environ.get("CUTOFF", CUTOFF)))


@pytest.fixture(scope="session")
def alpha():
    """Maximum magnitude of coherent states used in tests"""
    return float(os.environ.get("ALPHA", ALPHA))


@pytest.fixture
def hbar(request, monkeypatch):
    """Explicit units override the environment/default and never leak to another test."""
    value = float(getattr(request, "param", os.environ.get("HBAR", HBAR)))
    # Share the test's undo stack so later monkeypatches cannot restore stale units.
    monkeypatch.setattr(sf, "hbar", value)
    return value


@pytest.fixture(scope="session", autouse=True)
def default_hbar():
    """Keep the existing suite-wide default, restoring the caller's units afterwards."""
    previous = sf.hbar
    sf.hbar = float(os.environ.get("HBAR", HBAR))
    yield
    sf.hbar = previous


@pytest.fixture
def assert_normal_variance():
    """Check each Gaussian sample column with a two-sided 1e-6 chi-square test."""

    def check(samples, expected):
        samples = np.asarray(samples)
        assert samples.ndim in (1, 2) and samples.size > 0
        assert samples.shape[0] >= 2
        assert np.all(np.isfinite(samples))
        assert np.ndim(expected) == 0 and np.isfinite(expected) and expected > 0
        degrees = samples.shape[0] - 1
        bounds = chi2.ppf([5e-7, 1 - 5e-7], degrees) * expected / degrees
        variance = np.var(samples, axis=0, ddof=1)
        assert np.all(
            (bounds[0] <= variance) & (variance <= bounds[1])
        ), f"Sample variance {variance} outside Gaussian interval {bounds}"

    return check


@pytest.fixture
def state_trace(cutoff):
    """Use the native Fock trace or trace the finite Gaussian/Bosonic Fock tensor."""

    def trace(state):
        if not isinstance(state, (BaseGaussianState, BaseBosonicState)):
            return state.trace()
        modes = state.num_modes
        assert 1 <= modes <= 4
        density = np.asarray(state.reduced_dm(list(range(modes)), cutoff=cutoff))
        assert density.shape == (cutoff,) * (2 * modes)
        assert np.all(np.isfinite(density))
        # reduced_dm interleaves ket/bra axes; group them before taking the trace.
        axes = list(range(0, 2 * modes, 2)) + list(range(1, 2 * modes, 2))
        return np.trace(density.transpose(axes).reshape(cutoff**modes, cutoff**modes))

    return trace


@pytest.fixture
def assert_bosonic_even():
    """Bound cancellation roundoff at the original Bosonic dm() cutoff of ten."""

    def check(state):
        density = np.asarray(state.reduced_dm([0], cutoff=10))
        assert density.shape == (10, 10)
        assert np.all(np.isfinite(density))
        bound = 100 * np.finfo(np.float64).eps
        np.testing.assert_allclose(density[1::2, :], 0, atol=bound, rtol=0)
        np.testing.assert_allclose(density[:, 1::2], 0, atol=bound, rtol=0)

    return check


@pytest.fixture
def assert_fock_density(cutoff, tol):
    """Compare the full one/two-mode density matrix to independent Fock coefficients."""

    def check(state, ket):
        assert ket.ndim in (1, 2)
        modes = list(range(ket.ndim))
        expected = (
            np.outer(ket, ket.conj())
            if ket.ndim == 1
            else np.einsum("ij,kl->ikjl", ket, ket.conj())
        )
        actual = np.asarray(state.reduced_dm(modes, cutoff=cutoff))
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)

    return check


@pytest.fixture
def state_moments(cutoff, hbar, tol):
    """Return xxpp moments from native Gaussian data or finite Fock marginals."""

    def moments(state, batch_size=None):
        modes = state.num_modes
        assert 1 <= modes <= 3
        batch = () if batch_size is None else (batch_size,)
        if isinstance(state, BaseGaussianState):
            means, covariance = np.asarray(state.means()), np.asarray(state.cov())
        else:
            means = np.empty(batch + (2 * modes,))
            second = np.empty(batch + (2 * modes, 2 * modes))
            # Multiply before cropping: x^2 and p^2 include the |cutoff> boundary.
            ladder = np.diag(np.sqrt(np.arange(1, cutoff + 1)), 1)
            quadratures = np.sqrt(hbar / 2) * np.array(
                [ladder + ladder.T, -1j * (ladder - ladder.T)]
            )

            def marginal(selected):
                density = np.asarray(state.reduced_dm(selected))
                assert density.shape == batch + (cutoff,) * (2 * len(selected))
                assert np.all(np.isfinite(density))
                if len(selected) == 2:
                    # Group interleaved ket/bra axes into a matrix.
                    density = np.einsum("...abcd->...acbd", density).reshape(
                        batch + (cutoff**2, cutoff**2)
                    )
                np.testing.assert_allclose(
                    np.trace(density, axis1=-2, axis2=-1), 1, atol=tol, rtol=0
                )
                np.testing.assert_allclose(
                    density, density.swapaxes(-1, -2).conj(), atol=tol, rtol=0
                )
                return density

            for mode in range(modes):
                density = marginal([mode])
                for q, operator in enumerate(quadratures):
                    means[..., mode + q * modes] = np.einsum(
                        "...ij,ji->...", density, operator[:cutoff, :cutoff]
                    ).real
                    for p, other in enumerate(quadratures):
                        product = (operator @ other + other @ operator) / 2
                        second[..., mode + q * modes, mode + p * modes] = np.einsum(
                            "...ij,ji->...", density, product[:cutoff, :cutoff]
                        ).real
                for partner in range(mode + 1, modes):
                    density_pair = marginal([mode, partner])
                    for q, operator in enumerate(quadratures):
                        for p, other in enumerate(quadratures):
                            product = np.kron(operator[:cutoff, :cutoff], other[:cutoff, :cutoff])
                            value = np.einsum("...ij,ji->...", density_pair, product).real
                            first, last = mode + q * modes, partner + p * modes
                            second[..., first, last] = second[..., last, first] = value
            covariance = second - np.einsum("...i,...j->...ij", means, means)
        assert means.shape == batch + (2 * modes,)
        assert covariance.shape == batch + (2 * modes, 2 * modes)
        assert np.all(np.isfinite(means)) and np.all(np.isfinite(covariance))
        return means, covariance

    return moments


@pytest.fixture
def assert_finite_gaussian_moments(state_moments, batch_size, tol):
    """Compare finite-state moments with the full expected batch and covariance."""

    def check(state, expected_means, expected_covariance):
        means, covariance = state_moments(state, batch_size)
        batch = () if batch_size is None else (batch_size,)
        dimension = 2 * state.num_modes
        expected_means = np.broadcast_to(expected_means, batch + (dimension,))
        expected_covariance = np.broadcast_to(expected_covariance, batch + (dimension, dimension))
        assert means.shape == expected_means.shape
        assert covariance.shape == expected_covariance.shape
        np.testing.assert_allclose(means, expected_means, atol=tol, rtol=0)
        np.testing.assert_allclose(covariance, expected_covariance, atol=tol, rtol=0)

    return check


@pytest.fixture
def assert_passive_moments(tol):
    """Require analytic moments as well as agreement between two simulated states."""

    def check(initial, final, optical, expected_means, expected_cov, batch_size=None):
        dimension = optical.shape[0]
        assert optical.shape == (dimension, dimension)
        batch = () if batch_size is None else (batch_size,)
        expected_means = np.broadcast_to(expected_means, batch + (dimension,))
        expected_cov = np.broadcast_to(expected_cov, batch + (dimension, dimension))
        for (means, covariance), reference_means, reference_cov in (
            (initial, expected_means, expected_cov),
            (final, expected_means @ optical.T, optical @ expected_cov @ optical.T),
        ):
            assert means.shape == reference_means.shape
            assert covariance.shape == reference_cov.shape
            np.testing.assert_allclose(means, reference_means, atol=tol, rtol=0)
            np.testing.assert_allclose(covariance, reference_cov, atol=tol, rtol=0)
        # Retain the original covariance-propagation assertion too.
        assert np.allclose(final[1], optical @ initial[1] @ optical.T, atol=tol)

    return check


@pytest.fixture
def assert_marginal_fidelity(cutoff, tol):
    """Check complex marginal coherences before projecting onto the reference ket."""

    def check(state, mode, ket, batch_size=None):
        assert ket.shape == (cutoff,)
        batch = () if batch_size is None else (batch_size,)
        density = np.asarray(state.reduced_dm([mode], cutoff=cutoff))
        assert density.shape == batch + (cutoff, cutoff)
        expected = np.broadcast_to(np.outer(ket, ket.conj()), density.shape)
        np.testing.assert_allclose(density, expected, atol=tol, rtol=0)
        np.testing.assert_allclose(np.trace(density, axis1=-2, axis2=-1), 1, atol=tol, rtol=0)
        fidelity = np.einsum("i,...ij,j->...", ket.conj(), density, ket).real
        assert fidelity.shape == batch
        assert np.allclose(fidelity, 1, atol=tol)
        return fidelity

    return check


@pytest.fixture
def assert_independent_jacobian(tol):
    """Require each analytic derivative and zero coupling between batch members."""

    def check(actual, derivative, batch_size):
        derivative = np.asarray(derivative)
        assert np.all(np.isfinite(derivative))
        expected = derivative
        if batch_size is not None:
            assert derivative.ndim >= 1 and derivative.shape[0] == batch_size
            expected = np.zeros(derivative.shape + (batch_size,))
            for member in range(batch_size):
                expected[member, ..., member] = derivative[member]
        assert actual is not None, "Disconnected gradient"
        actual = np.asarray(actual)
        assert actual.shape == expected.shape, "Incorrect Jacobian shape"
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=0)

    return check


@pytest.fixture(
    params=[
        pytest.param(True, marks=pytest.mark.pure),
        pytest.param(False, marks=pytest.mark.mixed),
    ]
)
def pure(request):
    """Whether to run the backend in pure or mixed state mode"""
    return request.param  # bool(int(os.environ.get("PURE", PURE)))


@pytest.fixture(scope="session")
def batch_size():
    """Whether to run the backend in batched mode"""
    if "BATCHSIZE" in os.environ:
        # if user-specified BATCHSIZE provided, then batching is assumed (even if BATCHED=0)
        return int(os.environ["BATCHSIZE"])

    # check if batching is turned on
    batched = bool(int(os.environ.get("BATCHED", BATCHED)))

    if batched:
        # use the default batch size
        return BATCHSIZE

    return None  # no batching


@pytest.fixture
def skip_tf_batch(request, batch_size):
    """Keep tests with unbatched TensorFlow contracts out of batch configurations."""
    if batch_size is not None and request.node.get_closest_marker("tf") is not None:
        pytest.skip("This test only supports unbatched TensorFlow execution")


@pytest.fixture
def backend(monkeypatch):
    """Create a mocked out backend fixture for front-end only tests"""
    dummy_backend = BaseBackend()
    with monkeypatch.context() as m:
        # mock out the base backend
        m.setattr(dummy_backend, "add_mode", lambda n: None)
        m.setattr(dummy_backend, "del_mode", lambda n: None)
        m.setattr(dummy_backend, "displacement", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "prepare_coherent_state", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "squeeze", lambda r, phi, modes: None)
        m.setattr(dummy_backend, "rotation", lambda r, modes: None)
        m.setattr(dummy_backend, "beamsplitter", lambda t, r, m1, m2: None)
        m.setattr(
            dummy_backend, "measure_homodyne", lambda phi, modes, select, shots: np.array([[5]])
        )
        m.setattr(dummy_backend, "state", lambda modes, shots, crop=False: None)
        m.setattr(dummy_backend, "reset", lambda: None)
        dummy_backend.two_mode_squeeze = lambda r, phi, mode1, mode2: None
        dummy_backend.get_cutoff_dim = lambda: 6
        yield dummy_backend


@pytest.fixture
def print_fixtures(cutoff, hbar, batch_size):
    """Print the test configuration at the beginning of the session"""
    print(
        "FIXTURES: cutoff = {}, pure = {}, hbar = {}, batch_size = {}".format(
            cutoff, hbar, pure, batch_size
        )
    )


@pytest.fixture(params=backend_params)
def setup_backend(request, cutoff, pure, batch_size, hbar):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create a backend of certain number of modes.

    This fixture should only be used in backend tests, as it bypasses Engine and
    initializes the backend directly.

    Every test that uses this fixture, or a fixture that depends on it
    will be run three times, one for each backend.

    To explicitly mark a test to only work on certain backends, you may
    use the ``@pytest.mark.backends()`` fixture. For example, for a test that
    only works on the TF and Fock backends, ``@pytest.mark.backends('tf', 'fock').
    """

    def _setup_backend(num_subsystems):
        """Factory function"""
        backend = request.param()
        backend.begin_circuit(
            num_subsystems=num_subsystems,
            cutoff_dim=cutoff,
            pure=pure,
            batch_size=batch_size,
        )
        return backend

    return _setup_backend


@pytest.fixture(params=eng_backend_params)
def setup_backend_pars(
    request, cutoff, pure, batch_size, hbar
):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, a container for the backend parameters.

    Every test that uses this fixture, or a fixture that depends on it
    will be run three times, one for each backend.

    To explicitly mark a test to only work on certain backends, you may
    use the ``@pytest.mark.backends()`` fixture. For example, for a test that
    only works on the TF and Fock backends, ``@pytest.mark.backends('tf', 'fock').
    """
    return request.param, {"cutoff_dim": cutoff, "pure": pure, "batch_size": batch_size}


@pytest.fixture
def setup_eng(setup_backend_pars):  # pylint: disable=redefined-outer-name
    """Parameterized fixture, used to automatically create an Engine and a Program with a certain number of modes."""

    def _setup_eng(num_subsystems, **kwargs):
        """Factory function"""
        prog = Program(num_subsystems)
        backend, backend_options = setup_backend_pars
        backend_options.update(kwargs)  # override defaults with kwargs
        eng = LocalEngine(backend=backend, backend_options=backend_options)
        return eng, prog

    return _setup_eng


def pytest_runtest_call(item):
    """Record setup settings; tests may change units again inside their bodies."""
    if "setup_backend" in item.fixturenames or "setup_eng" in item.fixturenames:
        item.user_properties.extend(
            [
                ("setup_hbar", str(sf.hbar)),
                ("cutoff", str(item.funcargs["cutoff"])),
                ("batch_size", str(item.funcargs["batch_size"])),
            ]
        )
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        if "hbar" in parameters:
            item.user_properties.append(("requested_hbar", str(parameters["hbar"])))
        recorded = {"mesh"}
        if "assert_finite_gaussian_moments" in item.fixturenames:
            recorded.update(("a", "r", "phi"))
        item.user_properties.extend(
            ("parameter_" + name, str(parameters[name]))
            for name in sorted(recorded & parameters.keys())
        )


def pytest_runtest_setup(item):
    """Automatically skip tests if they are marked for only certain backends"""
    if tf_available:
        allowed_backends = {"gaussian", "tf", "fock", "bosonic"}
    else:
        allowed_backends = {"gaussian", "fock", "bosonic"}

    # load the marker specifying what the backend is
    marks = {mark.name for mark in item.iter_markers() if mark.name in allowed_backends}

    # load a marker specifying whether the test only works with certain backends
    test_backends = [mark.args for mark in item.iter_markers(name="backends")]

    if not test_backends:
        # if the test hasn't specified that it runs on particular backends,
        # assume it will work with all backends
        test_backends = allowed_backends
    else:
        # otherwise, extract the set of backend strings the test works with
        test_backends = test_backends[0]

    for b in marks:
        if b not in test_backends:
            pytest.skip(
                "\nTest {} only runs with {} backend(s), "
                "but {} backend provided".format(item.nodeid, test_backends, b)
            )

    # skip broken tests
    for mark in item.iter_markers(name="broken"):
        if mark.args:
            pytest.skip("Broken test skipped: {}".format(*mark.args))
        else:
            pytest.skip("Test skipped as corresponding code base is currently broken!")
