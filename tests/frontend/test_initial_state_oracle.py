# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Independent finite-state controls for Gaussian utility moment comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.frontend


@pytest.fixture
def initial_moment_data(cutoff, batch_size, hbar):
    """Analytic moments of normalized, phased |0>+|1> superpositions."""
    phase = 0.3 + 0.4 * np.arange(batch_size or 1)
    ket = np.zeros((len(phase), cutoff), dtype=complex)
    ket[:, 0] = 0.8
    ket[:, 1] = 0.6 * np.exp(1j * phase)
    density = np.einsum("...i,...j->...ij", ket, ket.conj())
    mean_a = 0.48 * np.exp(1j * phase)
    means = np.sqrt(2 * hbar) * np.stack([mean_a.real, mean_a.imag], axis=-1)
    covariance = hbar * (0.36 + 0.5) * np.eye(2) - np.einsum("...i,...j->...ij", means, means)
    if batch_size is None:
        density, means, covariance = density[0], means[0], covariance[0]
    return density, means, covariance


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize("batch_size", [None, 2, 3])
@pytest.mark.parametrize(
    "corruption",
    [None, "conjugate", "diagonal_only", "normalization", "units", "covariance"],
)
def test_initial_moment_reference(assert_finite_gaussian_moments, initial_moment_data, corruption):
    """Retain phase, normalization, off-diagonal covariance and the requested units."""
    density, means, covariance = initial_moment_data
    if corruption == "conjugate":
        density = density.conj()
    elif corruption == "diagonal_only":
        density = density * np.eye(density.shape[-1])
    elif corruption == "normalization":
        density = density * 0.99
    elif corruption == "units":
        means, covariance = np.sqrt(2) * means, 2 * covariance
    elif corruption == "covariance":
        covariance[..., 0, 1] += 0.1
        covariance[..., 1, 0] += 0.1
    state = SimpleNamespace(num_modes=1, reduced_dm=lambda modes: density)
    if corruption is None:
        assert_finite_gaussian_moments(state, means, covariance)
    else:
        with pytest.raises(AssertionError):
            assert_finite_gaussian_moments(state, means, covariance)


@pytest.mark.parametrize("hbar", [0.5, 1.7, 2.0], indirect=True)
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("corruption", ["missing_batch", "duplicate_member", "swap_members"])
def test_initial_moments_require_every_batch_member(
    assert_finite_gaussian_moments, initial_moment_data, corruption
):
    density, means, covariance = initial_moment_data
    if corruption == "missing_batch":
        density = density[0]
    elif corruption == "duplicate_member":
        density[1:] = density[0]
    else:
        density = density[::-1]
    state = SimpleNamespace(num_modes=1, reduced_dm=lambda modes: density)
    with pytest.raises(AssertionError):
        assert_finite_gaussian_moments(state, means, covariance)
