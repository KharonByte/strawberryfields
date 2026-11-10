# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Analytic, distinct-input controls for finite Gaussian moments in TensorFlow."""

import numpy as np
import pytest

from strawberryfields import ops


def check_state(setup_eng, hbar, batch_size, assert_finite_gaussian_moments, kind):
    """Keep each batch member's phase and squeezing distinct, preserving storage."""
    member = np.arange(batch_size) if batch_size is not None else np.asarray(2)
    alpha = (-0.15 + 0.175 * member) + 1j * (-0.2 + 0.15 * member)
    r, phi = 0.14 + 0.035 * member, 0.4766666666666666 * (member + 1)
    if kind == "coherent":
        r = np.zeros_like(r)
    elif kind == "squeezed":
        alpha = np.zeros_like(alpha)
    means = np.sqrt(2 * hbar) * np.stack([alpha.real, alpha.imag], axis=-1)
    covariance = np.empty(means.shape[:-1] + (2, 2))
    covariance[..., 0, 0] = hbar / 2 * (np.cosh(2 * r) - np.cos(phi) * np.sinh(2 * r))
    covariance[..., 1, 1] = hbar / 2 * (np.cosh(2 * r) + np.cos(phi) * np.sinh(2 * r))
    covariance[..., 0, 1] = covariance[..., 1, 0] = -hbar / 2 * np.sin(phi) * np.sinh(2 * r)
    engine, program = setup_eng(1)
    with program.context as q:
        ops.Sgate(r, phi) | q[0]
        ops.Dgate(np.abs(alpha), np.angle(alpha)) | q[0]
    state = engine.run(program).state
    assert bool(state.is_pure) == engine.backend_options["pure"]
    assert_finite_gaussian_moments(state, means, covariance)


@pytest.mark.parametrize(
    "setup_backend_pars,pure",
    [
        pytest.param("tf", True, marks=[pytest.mark.tf, pytest.mark.pure]),
        pytest.param("tf", False, marks=[pytest.mark.tf, pytest.mark.mixed]),
    ],
    indirect=True,
)
class TestDistinctInitialMoments:
    """Actual scalar and batched engines, with independent analytic references."""

    def test_coherent(self, setup_eng, hbar, batch_size, assert_finite_gaussian_moments):
        check_state(setup_eng, hbar, batch_size, assert_finite_gaussian_moments, "coherent")

    @pytest.mark.parametrize("cutoff", [8], indirect=True)
    def test_squeezed(self, setup_eng, hbar, batch_size, assert_finite_gaussian_moments):
        check_state(setup_eng, hbar, batch_size, assert_finite_gaussian_moments, "squeezed")

    @pytest.mark.parametrize("cutoff", [8], indirect=True)
    def test_displaced_squeezed(self, setup_eng, hbar, batch_size, assert_finite_gaussian_moments):
        check_state(
            setup_eng,
            hbar,
            batch_size,
            assert_finite_gaussian_moments,
            "displaced_squeezed",
        )
