# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Negative controls for the analytic passive-optics and rotated-state assertions."""

from math import factorial
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.frontend


@pytest.mark.parametrize("batch_size", [None, 2, 3])
@pytest.mark.parametrize("corruption", [None, "shared_bias", "wrong_phase", "wrong_means"])
def test_passive_analytic_reference(assert_passive_moments, hbar, batch_size, corruption):
    """Two matching but inaccurate covariances must not satisfy the optical oracle."""
    member = np.arange(batch_size or 1)
    angle = 0.37
    optical = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    means = np.array([[0.1 + 0.2 * b, -0.3 + 0.1 * b] for b in member])
    cov = np.array([np.diag(np.exp([-0.5 - b * 0.1, 0.5 + b * 0.1])) * hbar / 2 for b in member])
    if batch_size is None:
        means, cov = means[0], cov[0]
    initial = (means.copy(), cov.copy())
    final = (means @ optical.T, optical @ cov @ optical.T)
    if corruption == "shared_bias":
        initial = (initial[0], initial[1] * 0.99)
        final = (final[0], final[1] * 0.99)
        assert np.allclose(final[1], optical @ initial[1] @ optical.T)
    elif corruption == "wrong_phase":
        final = (final[0], optical.T @ cov @ optical)
    elif corruption == "wrong_means":
        final = (final[0] + 0.1, final[1])
    if corruption is None:
        assert_passive_moments(initial, final, optical, means, cov, batch_size)
    else:
        with pytest.raises(AssertionError):
            assert_passive_moments(initial, final, optical, means, cov, batch_size)


@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("corruption", ["missing_batch", "duplicate_member"])
def test_passive_batch_members(assert_passive_moments, batch_size, corruption):
    means = np.arange(2 * batch_size).reshape(batch_size, 2) * 0.1
    covariance = np.array([np.diag([1 + b * 0.1, 2 - b * 0.2]) for b in range(batch_size)])
    initial = (means.copy(), covariance.copy())
    final = (means.copy(), covariance.copy())
    if corruption == "missing_batch":
        final = (final[0][0], final[1][0])
    else:
        final[0][1:] = final[0][0]
        final[1][1:] = final[1][0]
    with pytest.raises(AssertionError):
        assert_passive_moments(initial, final, np.eye(2), means, covariance, batch_size)


@pytest.mark.parametrize("batch_size", [None, 2, 3])
@pytest.mark.parametrize("corruption", [None, "half_phase", "conjugate", "diagonal_only", "trace"])
def test_rotated_squeezed_coherences(assert_marginal_fidelity, cutoff, batch_size, corruption):
    """Independent coefficients expose the wrong phase hidden by near-unit fidelity."""
    r, rotation = 0.1, 0.2312

    def coefficients(phase):
        ket = np.zeros(cutoff, dtype=complex)
        for n in range((cutoff + 1) // 2):
            ket[2 * n] = (
                np.sqrt(factorial(2 * n))
                / factorial(n)
                * (-0.5 * np.exp(1j * phase) * np.tanh(r)) ** n
                / np.sqrt(np.cosh(r))
            )
        return ket

    expected = coefficients(2 * rotation)
    actual = coefficients(rotation if corruption == "half_phase" else 2 * rotation)
    density = np.outer(actual, actual.conj())
    if corruption == "half_phase":
        assert np.allclose(np.vdot(expected, density @ expected).real, 1, atol=1e-3)
    elif corruption == "conjugate":
        density = density.conj()
    elif corruption == "diagonal_only":
        density = np.diag(np.diag(density))
    elif corruption == "trace":
        density *= 0.99
    if batch_size is not None:
        density = np.broadcast_to(density, (batch_size, cutoff, cutoff)).copy()
    state = SimpleNamespace(reduced_dm=lambda modes, **kwargs: density)
    if corruption is None:
        assert_marginal_fidelity(state, 0, expected, batch_size)
    else:
        with pytest.raises(AssertionError):
            assert_marginal_fidelity(state, 0, expected, batch_size)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_marginal_requires_batch_axis(assert_marginal_fidelity, cutoff, batch_size):
    ket = np.eye(cutoff)[0]
    state = SimpleNamespace(reduced_dm=lambda modes, **kwargs: np.outer(ket, ket))
    with pytest.raises(AssertionError):
        assert_marginal_fidelity(state, 0, ket, batch_size)
