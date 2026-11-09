# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Independent analytic controls for the finite-density moment test adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.frontend


@pytest.fixture
def state_from_kets():
    """Trace spectator axes directly from small normalized test kets."""

    def build(kets, batch_size):
        modes = kets.ndim - 1
        cutoff = kets.shape[1]

        def reduced_dm(selected):
            remainder = [mode for mode in range(modes) if mode not in selected]
            marginals = []
            for ket in kets:
                amplitudes = ket.transpose(selected + remainder).reshape(
                    cutoff ** len(selected), -1
                )
                density = (amplitudes @ amplitudes.conj().T).reshape(
                    (cutoff,) * (2 * len(selected))
                )
                interleaved = [
                    axis for mode in range(len(selected)) for axis in (mode, mode + len(selected))
                ]
                marginals.append(density.transpose(interleaved))
            return np.asarray(marginals) if batch_size is not None else marginals[0]

        return SimpleNamespace(num_modes=modes, reduced_dm=reduced_dm)

    return build


@pytest.mark.parametrize("modes", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [None, 2, 3])
def test_moments_at_cutoff_boundary(
    state_moments, state_from_kets, cutoff, hbar, modes, batch_size
):
    """The top Fock level still has variance hbar*(n+1/2), including its boundary."""
    kets = np.zeros((batch_size or 1,) + (cutoff,) * modes, dtype=complex)
    covariance = []
    for member, ket in enumerate(kets):
        occupations = (np.arange(modes) + member) % cutoff
        occupations[0] = cutoff - 1 - member
        ket[tuple(occupations)] = 1
        covariance.append(np.diag(hbar * np.tile(occupations + 0.5, 2)))
    means, actual = state_moments(state_from_kets(kets, batch_size), batch_size)
    expected = np.asarray(covariance) if batch_size else covariance[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(means, np.zeros(expected.shape[:-1]))


@pytest.mark.parametrize("modes", [2, 3])
@pytest.mark.parametrize("batch_size", [None, 2, 3])
def test_complex_pair_correlations(state_moments, state_from_kets, cutoff, hbar, modes, batch_size):
    """A phased |00>+|11> pair has nonzero xx, pp and xp cross-mode moments."""
    kets = np.zeros((batch_size or 1,) + (cutoff,) * modes, dtype=complex)
    covariance = []
    for member, ket in enumerate(kets):
        phase = 0.4 + member * 0.3
        spectator = [member % 2] if modes == 3 else []
        ket[tuple([0, 0] + spectator)] = 0.6
        ket[tuple([1, 1] + spectator)] = 0.8 * np.exp(1j * phase)
        cov = np.diag(hbar * np.tile(np.array([0.64, 0.64] + spectator) + 0.5, 2))
        xx, xp = hbar * 0.48 * np.array([np.cos(phase), np.sin(phase)])
        cov[0, 1] = cov[1, 0] = xx
        cov[modes, modes + 1] = cov[modes + 1, modes] = -xx
        cov[0, modes + 1] = cov[modes + 1, 0] = xp
        cov[1, modes] = cov[modes, 1] = xp
        covariance.append(cov)
    means, actual = state_moments(state_from_kets(kets, batch_size), batch_size)
    expected = np.asarray(covariance) if batch_size else covariance[0]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(means, 0, atol=1e-12, rtol=0)


@pytest.fixture
def displaced_pair(state_from_kets, cutoff, hbar):
    """Two distinct product superpositions with analytic first/second moments."""
    kets, means, covariance = [], [], []
    for member in range(2):
        amplitudes = np.array([0.2 + 0.3j, -0.4 + 0.1j]) * (member + 1)
        photons = abs(amplitudes) ** 2 / (1 + abs(amplitudes) ** 2)
        ladder_mean = amplitudes / (1 + abs(amplitudes) ** 2)
        mean = np.sqrt(2 * hbar) * np.r_[ladder_mean.real, ladder_mean.imag]
        cov = np.diag(hbar * np.tile(photons + 0.5, 2) - mean**2)
        for mode in range(2):
            cov[mode, mode + 2] = cov[mode + 2, mode] = -mean[mode] * mean[mode + 2]
        factors = np.zeros((2, cutoff), dtype=complex)
        factors[:, 0] = 1
        factors[:, 1] = amplitudes
        factors /= np.sqrt(1 + abs(amplitudes) ** 2)[:, None]
        kets.append(np.outer(*factors))
        means.append(mean)
        covariance.append(cov)
    return state_from_kets(np.asarray(kets), 2), np.asarray(means), np.asarray(covariance)


@pytest.mark.parametrize("corruption", [None, "duplicate_member", "conjugate", "swap_modes"])
def test_distinct_means_and_covariance(state_moments, displaced_pair, corruption):
    """Check xxpp order, complex coherences, covariance centering and every member."""
    state, expected_means, expected_cov = displaced_pair
    original = state.reduced_dm

    def reduced_dm(selected):
        density = original(
            [1 - mode for mode in selected] if corruption == "swap_modes" else selected
        )
        if corruption == "duplicate_member":
            density[1] = density[0]
        elif corruption == "conjugate":
            density = density.conj()
        return density

    state.reduced_dm = reduced_dm

    def check():
        means, covariance = state_moments(state, 2)
        np.testing.assert_allclose(means, expected_means, atol=1e-12, rtol=0)
        np.testing.assert_allclose(covariance, expected_cov, atol=1e-12, rtol=0)

    if corruption is None:
        check()
    else:
        with pytest.raises(AssertionError):
            check()


@pytest.mark.parametrize("selected_modes", [1, 2])
@pytest.mark.parametrize(
    "corruption", ["shape", "nonfinite", "normalization", "hermiticity", "missing_batch"]
)
def test_invalid_marginals_are_rejected(state_moments, displaced_pair, selected_modes, corruption):
    """Do not normalize away cutoff loss or broadcast a missing batch axis."""
    state, _, _ = displaced_pair
    original = state.reduced_dm

    def reduced_dm(selected):
        density = original(selected)
        if len(selected) == selected_modes:
            if corruption == "shape":
                density = density[..., :-1]
            elif corruption == "nonfinite":
                density.flat[0] = np.nan
            elif corruption == "normalization":
                density *= 0.9
            elif corruption == "hermiticity":
                density.flat[1] += 0.1j
            elif corruption == "missing_batch":
                density = density[0]
        return density

    state.reduced_dm = reduced_dm
    with pytest.raises(AssertionError):
        state_moments(state, 2)
