"""Independent analytic laws for terminal single-mode Gaussian detection."""

import math

import numpy as np
import pytest
from scipy.stats import poisson

import strawberryfields as sf
from strawberryfields.backends.gaussianbackend import GaussianBackend
from strawberryfields.backends.gaussianbackend import sampling


@pytest.mark.parametrize("nbar", [0, 0.1, 1, 4])
def test_thermal_distribution(nbar):
    probabilities = sampling.single_mode_probabilities(np.zeros(2), (2 * nbar + 1) * np.eye(2))
    expected = (nbar / (nbar + 1)) ** np.arange(len(probabilities)) / (nbar + 1)
    assert np.abs(probabilities - expected).sum() < 2e-10


@pytest.mark.parametrize("alpha", [0.3 + 0.7j, 2 - 1j, 1e-9j])
def test_coherent_distribution(alpha):
    probabilities = sampling.single_mode_probabilities(
        2 * np.array([alpha.real, alpha.imag]), np.eye(2)
    )
    expected = poisson.pmf(np.arange(len(probabilities)), abs(alpha) ** 2)
    assert np.abs(probabilities - expected).sum() < 2e-10


@pytest.mark.parametrize("eta", [0.2, 0.7, 1])
@pytest.mark.parametrize("phase", [0, 0.7])
def test_lossy_squeezed_distribution(eta, phase):
    # Squeezed-vacuum pair law followed by independent binomial loss.
    r = np.arctanh(0.5)
    backend = GaussianBackend()
    backend.begin_circuit(1)
    backend.prepare_squeezed_state(r, phase, 0)
    backend.loss(eta, 0)
    probabilities = sampling.single_mode_probabilities(
        backend.circuit.smeanxp(), backend.circuit.scovmatxp()
    )
    expected = np.zeros(len(probabilities))
    for k in range(100):
        pair_probability = math.comb(2 * k, k) * 0.25**k * 0.25**k * np.sqrt(0.75)
        for n in range(min(2 * k + 1, len(expected))):
            expected[n] += (
                pair_probability * math.comb(2 * k, n) * eta**n * (1 - eta) ** (2 * k - n)
            )
    assert np.abs(probabilities - expected).sum() < 2e-10
    if eta == 1:
        assert abs(probabilities[5:].sum() - (1 - 147 * np.sqrt(3) / 256)) < 2e-10


@pytest.mark.parametrize("engine", [False, True])
@pytest.mark.parametrize("threshold", [False, True])
def test_squeezed_terminal_law(engine, threshold):
    np.random.seed(1700000000)
    r = np.arctanh(0.5)
    with pytest.warns(UserWarning, match="Conditional state"):
        if engine:
            program = sf.Program(1)
            with program.context as q:
                sf.ops.Sgate(r, 0.7) | q[0]
                (sf.ops.MeasureThreshold() if threshold else sf.ops.MeasureFock()) | q[0]
            samples = sf.Engine("gaussian").run(program, shots=100000).samples
        else:
            backend = GaussianBackend()
            backend.begin_circuit(1)
            backend.prepare_squeezed_state(r, 0.7, 0)
            method = backend.measure_threshold if threshold else backend.measure_fock
            samples = method([0], shots=100000)
    assert samples.shape == (100000, 1)
    assert np.issubdtype(samples.dtype, np.integer)
    if threshold:
        assert set(np.unique(samples)) <= {0, 1}
        assert abs(samples.mean() - (1 - np.sqrt(3) / 2)) < 0.005
    else:
        assert np.all(samples % 2 == 0)
        assert abs(np.mean(samples >= 5) - (1 - 147 * np.sqrt(3) / 256)) < 0.0015


def test_resource_limit_precedes_density_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Density matrix allocated beyond resource limit")

    monkeypatch.setattr(sampling, "density_matrix", forbidden)
    with pytest.raises(ValueError, match="cutoff above 512"):
        sampling.single_mode_probabilities(np.zeros(2), 1000 * np.eye(2))


@pytest.mark.parametrize("cov", [np.full((2, 2), np.nan), 0.5 * np.eye(2)])
def test_invalid_moments(cov):
    with pytest.raises(ValueError):
        sampling.single_mode_probabilities(np.zeros(2), cov)


def test_probability_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(sampling, "density_matrix", lambda *a, **k: np.zeros((2, 2)))
    with pytest.raises(ValueError, match="probability mass"):
        sampling.single_mode_probabilities(np.zeros(2), np.eye(2))


def test_reduced_entangled_mode_is_thermal():
    backend = GaussianBackend()
    backend.begin_circuit(2)
    # Two equally squeezed modes at orthogonal angles, mixed 50:50.
    backend.prepare_squeezed_state(np.arcsinh(1), 0, 0)
    backend.prepare_squeezed_state(np.arcsinh(1), np.pi, 1)
    backend.beamsplitter(np.pi / 4, 0, 0, 1)
    cov = backend.circuit.scovmatxp()[np.ix_([1, 3], [1, 3])]
    probabilities = sampling.single_mode_probabilities(np.zeros(2), cov)
    assert np.abs(probabilities - 0.5 ** (np.arange(len(probabilities)) + 1)).sum() < 2e-10


def test_displaced_squeezed_moments():
    alpha, r, phi = 0.4 + 0.7j, 0.45, 0.3
    backend = GaussianBackend()
    backend.begin_circuit(1)
    backend.prepare_displaced_squeezed_state(abs(alpha), np.angle(alpha), r, phi, 0)
    probabilities = sampling.single_mode_probabilities(
        backend.circuit.smeanxp(), backend.circuit.scovmatxp()
    )
    n = np.arange(len(probabilities))
    mean = np.sinh(r) ** 2 + abs(alpha) ** 2
    variance = 2 * np.sinh(r) ** 2 * (1 + np.sinh(r) ** 2)
    variance += abs(alpha) ** 2 * (
        np.cosh(2 * r) - np.sinh(2 * r) * np.cos(2 * np.angle(alpha) - phi)
    )
    assert abs(n @ probabilities - mean) < 1e-8
    assert abs((n - mean) ** 2 @ probabilities - variance) < 1e-7


@pytest.mark.parametrize("family", ["coherent", "thermal", "squeezed_loss"])
def test_threshold_probability(family, monkeypatch):
    backend = GaussianBackend()
    backend.begin_circuit(1)
    if family == "coherent":
        backend.prepare_coherent_state(0.8, 0.7, 0)
        p0 = np.exp(-(0.8**2))
    elif family == "thermal":
        backend.prepare_thermal_state(0.4, 0)
        p0 = 1 / 1.4
    else:
        backend.prepare_squeezed_state(np.arctanh(0.5), 0.7, 0)
        backend.loss(0.6, 0)
        p0 = np.sqrt(0.75 / (1 - 0.25 * 0.4**2))

    def capture(n, p, size):
        assert n == 1
        assert abs(p - (1 - p0)) < 1e-12
        return np.zeros(size, dtype=int)

    monkeypatch.setattr(np.random, "binomial", capture)
    assert backend.measure_threshold([0]).shape == (1, 1)


def test_engine_nondefault_hbar(monkeypatch):
    monkeypatch.setattr(sf, "hbar", 8)
    program = sf.Program(1)
    with program.context as q:
        sf.ops.Coherent(0.8, 0.7) | q[0]
        sf.ops.MeasureThreshold() | q[0]

    def capture(n, p, size):
        assert abs(p - (1 - np.exp(-0.64))) < 1e-12
        return np.zeros(size, dtype=int)

    monkeypatch.setattr(np.random, "binomial", capture)
    assert sf.Engine("gaussian").run(program).samples.shape == (1, 1)
