"""Bounded terminal sampling of one Gaussian mode, in internal hbar=2 units.

The PNR truncation target is an exact-arithmetic Chernoff bound, not a
certificate for floating-point error. Multimode sampling is not implemented here.
"""

import numpy as np
from thewalrus.quantum import density_matrix

TAIL_TOLERANCE = 1e-10
MAX_CUTOFF = 512
NUMERICAL_TOLERANCE = 1e-9


def _validate(mean, cov):
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError("Expected one Gaussian mode")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(cov)):
        raise ValueError("Gaussian moments must be finite")
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-12):
        raise ValueError("Gaussian covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if eigenvalues[0] <= 0 or np.linalg.det(cov) < 1 - 1e-10:
        raise ValueError("Unphysical single-mode Gaussian covariance")
    return mean, cov, eigenvalues


def _cutoff(mean, cov, eigenvalues):
    """Bound P(N >= cutoff) using the photon-number generating function.

    A(z) = V + I - z(V - I); G(z) = 2 exp((z-1) mu^T A^-1 mu/2)
    / sqrt(det(A)). For real z>1 with A positive definite,
    P(N>=d) <= G(z)/z**d. A finite grid chooses an efficient valid bound.
    """
    radius = 2 / max(eigenvalues[-1] - 1, 1)
    best = MAX_CUTOFF + 1
    for fraction in (0.125, 0.25, 0.5, 0.75):
        step = radius * fraction
        matrix = 2 * np.eye(2) - step * (cov - np.eye(2))
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            continue
        log_g = np.log(2) - logdet / 2 + step * mean @ np.linalg.solve(matrix, mean) / 2
        required = (log_g - np.log(TAIL_TOLERANCE)) / np.log1p(step)
        if np.isfinite(required) and required <= MAX_CUTOFF:
            best = min(best, max(2, int(np.ceil(required))))
    if best > MAX_CUTOFF:
        raise ValueError("Single-mode PNR tail bound requires a cutoff above 512")
    return best


def single_mode_probabilities(mean, cov):
    """Return a checked, normalized finite PNR law with bounded omitted mass.

    In exact arithmetic the TV distance of this conditional law from the full
    law is its omitted mass, at most TAIL_TOLERANCE. Numerical checks below
    detect gross failures; they do not prove a floating-point forward bound.
    The density matrix has at most 512**2 complex entries.
    """
    mean, cov, eigenvalues = _validate(mean, cov)
    cutoff = _cutoff(mean, cov, eigenvalues)
    diagonal = np.diag(density_matrix(mean, cov, cutoff=cutoff, normalize=False, hbar=2))
    if not np.all(np.isfinite(diagonal)) or np.max(np.abs(diagonal.imag)) > NUMERICAL_TOLERANCE:
        raise ValueError("Invalid single-mode PNR probabilities")
    probabilities = diagonal.real.copy()
    if probabilities.min() < -NUMERICAL_TOLERANCE:
        raise ValueError("Negative single-mode PNR probability")
    probabilities = np.maximum(probabilities, 0)
    mass = probabilities.sum()
    if abs(mass - 1) > TAIL_TOLERANCE + NUMERICAL_TOLERANCE:
        raise ValueError("Single-mode PNR probability mass disagrees with tail bound")
    return probabilities / mass


def sample_single_mode(mean, cov, shots, threshold=False):
    """Draw terminal samples; no conditional state update is performed."""
    if threshold:
        mean, cov, _ = _validate(mean, cov)
        matrix = cov + np.eye(2)
        log_p0 = np.log(2) - np.linalg.slogdet(matrix)[1] / 2
        log_p0 -= mean @ np.linalg.solve(matrix, mean) / 2
        p0 = np.exp(log_p0)
        if not np.isfinite(p0) or p0 > 1 + NUMERICAL_TOLERANCE:
            raise ValueError("Invalid single-mode vacuum probability")
        return np.random.binomial(1, 1 - min(p0, 1), size=(shots, 1))
    probabilities = single_mode_probabilities(mean, cov)
    return np.random.choice(len(probabilities), size=(shots, 1), p=probabilities)
