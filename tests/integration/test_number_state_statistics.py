# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exact finite-support number-state moments and their batch shape contract."""

from types import SimpleNamespace

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

OCCUPATIONS = ((0, 1, 2), (2, 0, 1))


def assert_number_state_moments(state, occupations, atol):
    """Check shapes before values so scalar broadcasting cannot hide lost members."""
    expected = np.asarray(occupations)
    expected_shape = expected.shape[:-1]
    maximum_error = 0.0
    for mode in range(3):
        mean, variance = map(np.asarray, state.mean_photon(mode))
        assert mean.shape == expected_shape, "mean shape"
        assert variance.shape == expected_shape, "variance shape"
        assert np.all(np.isfinite(mean))
        assert np.all(np.isfinite(variance))
        # N_j |n_0,n_1,n_2> = n_j |n_0,n_1,n_2>, hence Var(N_j) = 0.
        np.testing.assert_allclose(mean, expected[..., mode], atol=atol, rtol=0)
        np.testing.assert_allclose(variance, np.zeros(expected_shape), atol=atol, rtol=0)
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(mean - expected[..., mode]))),
            float(np.max(np.abs(variance))),
        )
    return maximum_error


@pytest.mark.parametrize(
    "backend_name,occupations",
    [
        pytest.param("fock", OCCUPATIONS[0], marks=pytest.mark.fock, id="fock-A"),
        pytest.param("fock", OCCUPATIONS[1], marks=pytest.mark.fock, id="fock-B"),
        pytest.param("tf", OCCUPATIONS[0], marks=pytest.mark.tf, id="tf-A"),
        pytest.param("tf", OCCUPATIONS[1], marks=pytest.mark.tf, id="tf-B"),
        pytest.param("tf", OCCUPATIONS, marks=pytest.mark.tf, id="tf-distinct-batch"),
    ],
)
@pytest.mark.parametrize("cutoff", [3, 4], indirect=True)
@pytest.mark.parametrize("hbar", [1.0, 1.7], indirect=True)
def test_distinct_number_state_moments(backend_name, occupations, cutoff, hbar, pure, request):
    """Both storage forms retain each mode/member, with no cutoff tail or unit dependence.

    Zero number-state variances do not detect copying a correct zero variance
    between members, or establish correctness for nonzero-variance states.
    """
    if backend_name == "tf":
        pytest.importorskip("tensorflow")

    expected = np.asarray(occupations)
    batched = expected.ndim == 2
    batch_size = len(expected) if batched else None
    leading_shape = (batch_size,) if batched else ()
    members = expected if batched else expected[None, :]
    assert np.all((members >= 0) & (members < cutoff))
    data_shape = leading_shape + (cutoff,) * (3 if pure else 6)
    reference = np.zeros(data_shape, dtype=np.complex128)
    for member, numbers in enumerate(members):
        indices = tuple(numbers) if pure else tuple(n for n in numbers for _ in range(2))
        reference[((member,) if batched else ()) + indices] = 1

    engine = sf.Engine(
        backend_name,
        backend_options={"cutoff_dim": cutoff, "pure": pure, "batch_size": batch_size},
    )
    program = sf.Program(3)
    # Whole-state preparation preserves the requested representation. Replacing
    # individual modes can turn ket storage into density storage even for |n>.
    with program.context as q:
        (ops.Ket(reference.copy()) if pure else ops.DensityMatrix(reference.copy())) | q
    state = engine.run(program).state
    data = np.asarray(state.data)
    assert state.is_pure == pure
    assert state.hbar == hbar
    assert state.num_modes == 3
    assert state.cutoff_dim == cutoff
    assert state.mode_names == {0: "q[0]", 1: "q[1]", 2: "q[2]"}
    assert getattr(state, "batched", False) == batched
    assert data.shape == data_shape
    assert data.dtype in (np.dtype("complex64"), np.dtype("complex128"))
    assert np.all(np.isfinite(data))
    # All coefficients and number-diagonal contractions are exact small integers.
    # Eight eps is a predeclared diagnostic allowance, not truncation error.
    atol = 8 * np.finfo(data.real.dtype).eps
    np.testing.assert_allclose(data, reference, atol=atol, rtol=0)
    trace = np.asarray(state.trace())
    assert trace.shape == leading_shape
    np.testing.assert_allclose(trace, np.ones(leading_shape), atol=atol, rtol=0)
    for mode in range(3):
        reduced = np.asarray(state.reduced_dm(mode))
        expected_reduced = np.zeros(leading_shape + (cutoff, cutoff))
        for member, numbers in enumerate(members):
            indices = (numbers[mode], numbers[mode])
            expected_reduced[((member,) if batched else ()) + indices] = 1
        assert reduced.shape == expected_reduced.shape
        np.testing.assert_allclose(reduced, expected_reduced, atol=atol, rtol=0)
    maximum_error = assert_number_state_moments(state, occupations, atol)
    request.node.user_properties.extend(
        [
            ("effective_dtype", str(data.dtype)),
            ("effective_storage", "ket" if state.is_pure else "density"),
            ("effective_hbar", str(state.hbar)),
            ("effective_cutoff", str(state.cutoff_dim)),
            ("effective_batch_size", str(batch_size)),
            ("maximum_moment_error", str(maximum_error)),
            ("maximum_state_error", str(float(np.max(np.abs(data - reference))))),
            ("atol", str(atol)),
        ]
    )


@pytest.mark.parametrize("occupations", [OCCUPATIONS[0], OCCUPATIONS[1], OCCUPATIONS])
def test_number_state_moment_oracle_accepts_reference(occupations):
    """Valid scalar and distinct-member references pass the same assertion path."""
    expected = np.asarray(occupations)
    state = SimpleNamespace(
        mean_photon=lambda mode: (expected[..., mode], np.zeros(expected.shape[:-1]))
    )
    assert assert_number_state_moments(state, occupations, 8 * np.finfo(float).eps) == 0


@pytest.mark.parametrize("observable", ["mean", "variance"])
@pytest.mark.parametrize("shape_error", ["scalar", "extra_axis", "unbatched_vector"])
def test_number_state_moment_oracle_rejects_shape(observable, shape_error):
    """Independently corrupt either observable's shape without changing its values."""
    expected = np.asarray(OCCUPATIONS[0] if shape_error == "unbatched_vector" else OCCUPATIONS)

    def mean_photon(mode):
        results = [expected[..., mode], np.zeros(expected.shape[:-1])]
        index = 0 if observable == "mean" else 1
        value = np.asarray(results[index])
        if shape_error == "scalar":
            results[index] = value[0]
        else:
            results[index] = value[..., None]
        return results

    with pytest.raises(AssertionError, match=observable + " shape"):
        assert_number_state_moments(SimpleNamespace(mean_photon=mean_photon), expected, 1e-15)


@pytest.mark.parametrize("error", ["copied_member", "reordered_members", "wrong_mode", "variance"])
def test_number_state_moment_oracle_rejects_values(error):
    """Correct shapes cannot hide copied/reordered means or an erroneous variance."""
    expected = np.asarray(OCCUPATIONS)

    def mean_photon(mode):
        mean = expected[:, mode].copy()
        variance = np.zeros(2)
        if error == "copied_member":
            mean[:] = mean[0]
        elif error == "reordered_members":
            mean = mean[::-1]
        elif error == "wrong_mode":
            mean = expected[:, (mode + 1) % 3]
        else:
            variance[1] = 1
        return mean, variance

    with pytest.raises(AssertionError):
        assert_number_state_moments(SimpleNamespace(mean_photon=mean_photon), expected, 1e-15)
