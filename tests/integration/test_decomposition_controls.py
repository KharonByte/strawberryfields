# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Distinct-input controls for passive decomposition in both TF storage modes."""

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops


@pytest.mark.parametrize(
    "setup_backend_pars,pure",
    [
        pytest.param("tf", True, marks=[pytest.mark.tf, pytest.mark.pure]),
        pytest.param("tf", False, marks=[pytest.mark.tf, pytest.mark.mixed]),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh", ["passive", "rectangular", "rectangular_compact", "triangular_compact", "sun_compact"]
)
def test_distinct_passive_inputs(
    setup_eng, batch_size, hbar, mesh, state_moments, assert_passive_moments
):
    """Low-energy controls retain cutoff 6 and exercise actual pure/mixed storage."""
    eng, initial_program = setup_eng(3)
    member = np.arange(batch_size or 1)
    magnitude = np.array([0.1, 0.13, 0.08]) + member[:, None] * 0.025
    phase = np.array([0.17, -0.43, 0.31]) + member[:, None] * 0.23
    alpha = (0.03 + 0.025 * member) * np.exp(1j * (-0.2 + 0.4 * member))
    means = np.zeros((len(member), 6))
    means[:, 0], means[:, 3] = np.sqrt(2 * hbar) * alpha.real, np.sqrt(2 * hbar) * alpha.imag
    covariance = np.zeros((len(member), 6, 6))
    for mode in range(3):
        cosh, sinh = np.cosh(2 * magnitude[:, mode]), np.sinh(2 * magnitude[:, mode])
        covariance[:, mode, mode] = (hbar / 2) * (cosh - np.cos(phase[:, mode]) * sinh)
        covariance[:, mode + 3, mode + 3] = (hbar / 2) * (cosh + np.cos(phase[:, mode]) * sinh)
        covariance[:, mode, mode + 3] = covariance[:, mode + 3, mode] = (
            -hbar / 2 * np.sin(phase[:, mode]) * sinh
        )
    if batch_size is None:
        magnitude, phase, alpha = magnitude[0], phase[0], alpha[0]
        means, covariance = means[0], covariance[0]
    with initial_program.context as q:
        for mode in range(3):
            ops.Sgate(magnitude[..., mode], phase[..., mode]) | q[mode]
        ops.Dgate(np.abs(alpha), np.angle(alpha)) | q[0]
    initial = eng.run(initial_program).state
    assert bool(initial.is_pure) == eng.backend_options["pure"]

    unitary = np.exp(2j * np.pi * np.outer(np.arange(3), np.arange(3)) / 3) / np.sqrt(3)
    unitary = unitary @ np.diag(np.exp(1j * np.array([0.37, -0.21, 0.19])))
    optical = np.block([[unitary.real, -unitary.imag], [unitary.imag, unitary.real]])
    continuation = sf.Program(initial_program)
    with continuation.context as q:
        if mesh == "passive":
            ops.GaussianTransform(optical) | q
        else:
            ops.Interferometer(unitary, mesh=mesh) | q
    final = eng.run(continuation).state
    assert bool(final.is_pure) == eng.backend_options["pure"]
    assert_passive_moments(
        state_moments(initial, batch_size),
        state_moments(final, batch_size),
        optical,
        means,
        covariance,
        batch_size,
    )
