# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Finite-support states with distinct nonzero photon variances in every member."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

# A = sqrt(3)/2 |01> + 1/2 |20>; B = 1/2 |10> + sqrt(3)/2 |02>.
# Density storage deliberately uses the incoherent mixtures, not ket projectors.
SUPPORT = {"A": (((0, 1), (2, 0)), (0.75, 0.25)), "B": (((1, 0), (0, 2)), (0.25, 0.75))}
# Independent N|n> = n|n> calculation: E[N^2] - E[N]^2.
MEANS = {"A": (0.5, 0.75), "B": (0.25, 1.5)}
VARIANCES = {"A": (0.75, 0.1875), "B": (0.1875, 0.75)}
SECONDS = {"A": (1.0, 0.75), "B": (0.25, 3.0)}


def reference(labels, pure, cutoff):
    """Construct explicit input tensors and independent full/marginal densities."""
    if not isinstance(labels, str):
        return tuple(
            np.stack(parts) for parts in zip(*(reference(x, pure, cutoff) for x in labels))
        )
    occupations, weights = SUPPORT[labels]
    ket = np.zeros((cutoff, cutoff), dtype=np.complex128)
    density = np.zeros((cutoff,) * 4, dtype=np.complex128)
    reduced = np.zeros((2, cutoff, cutoff), dtype=np.complex128)
    for numbers, weight in zip(occupations, weights):
        ket[numbers] = np.sqrt(3) / 2 if weight == 0.75 else 0.5
        density[tuple(n for n in numbers for _ in range(2))] = weight
        for mode, n in enumerate(numbers):
            reduced[mode, n, n] += weight
    if pure:
        left, right = occupations
        density[left[0], right[0], left[1], right[1]] = np.sqrt(3) / 4
        density[right[0], left[0], right[1], left[1]] = np.sqrt(3) / 4
    return ket if pure else density.copy(), density, reduced


def expected_moments(labels):
    if isinstance(labels, str):
        return np.asarray(MEANS[labels]), np.asarray(VARIANCES[labels])
    return np.asarray([MEANS[x] for x in labels]), np.asarray([VARIANCES[x] for x in labels])


def assert_photon_array(actual, expected, atol, label):
    """Do not let broadcasting or a shape repair hide a lost mode or member."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, label + " shape"
    assert np.all(np.isfinite(actual)), label + " finite"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=label)
    return float(np.max(np.abs(actual - expected)))


def assert_variance_moments(state, labels, atol):
    means, variances = expected_moments(labels)
    errors = [0.0, 0.0]
    for mode in range(2):
        mean, variance = state.mean_photon(mode)
        for index, (actual, expected, label) in enumerate(
            zip((mean, variance), (means[..., mode], variances[..., mode]), ("mean", "variance"))
        ):
            errors[index] = max(errors[index], assert_photon_array(actual, expected, atol, label))
    return errors


def check_coefficient_precision(real_dtype):
    """Certify the sqrt coefficient's casting bound without simulator output."""
    coefficient = Fraction(float(real_dtype(np.sqrt(3) / 2)))
    u = Fraction(float(np.finfo(real_dtype).eps)) / 2
    assert coefficient > 0
    assert Fraction(3, 4) * (1 - 2 * u) ** 2 <= coefficient**2
    assert coefficient**2 <= Fraction(3, 4) * (1 + 2 * u) ** 2


@pytest.mark.parametrize(
    "backend_name,labels",
    [
        pytest.param("fock", "A", marks=pytest.mark.fock, id="fock-A"),
        pytest.param("fock", "B", marks=pytest.mark.fock, id="fock-B"),
        pytest.param("tf", "A", marks=pytest.mark.tf, id="tf-A"),
        pytest.param("tf", "B", marks=pytest.mark.tf, id="tf-B"),
        pytest.param("tf", ("A", "B"), marks=pytest.mark.tf, id="tf-distinct-batch"),
    ],
)
@pytest.mark.parametrize("cutoff", [3, 4], indirect=True)
@pytest.mark.parametrize("hbar", [1.0, 1.7], indirect=True)
def test_distinct_photon_variance(backend_name, labels, cutoff, hbar, pure, request):
    """No-gate finite support checks nonzero variances without a cutoff tail."""
    if backend_name == "tf":
        pytest.importorskip("tensorflow")
    real_dtype = np.float32 if backend_name == "tf" else np.float64
    check_coefficient_precision(real_dtype)
    # Let u=eps/2. The coefficient bound gives squared-weight error <4u.
    # Only one nonzero occupation contributes to each N/N^2 contraction;
    # powers of two scale exactly. Weight sensitivity plus the rounded mean
    # square and subtraction bounds variance error by <12u; 8eps = 16u.
    # This bound is specific to these sparse real states and the direct path.
    atol = 8 * np.finfo(real_dtype).eps
    data_reference, density_reference, reduced_reference = reference(labels, pure, cutoff)
    batched = not isinstance(labels, str)
    batch_size = len(labels) if batched else None
    leading = (batch_size,) if batched else ()
    engine = sf.Engine(
        backend_name,
        backend_options={"cutoff_dim": cutoff, "pure": pure, "batch_size": batch_size},
    )
    program = sf.Program(2)
    with program.context as q:
        (ops.Ket(data_reference.copy()) if pure else ops.DensityMatrix(data_reference.copy())) | q
    state = engine.run(program).state
    data = np.asarray(state.data)
    assert data.real.dtype == np.dtype(real_dtype)
    assert state.is_pure == pure
    assert state.num_modes == 2
    assert state.mode_names == {0: "q[0]", 1: "q[1]"}
    assert state.hbar == hbar
    assert state.cutoff_dim == cutoff
    assert getattr(state, "batched", False) == batched
    errors = {"state": assert_photon_array(data, data_reference, atol, "state")}
    # Full projectors retain coherence; the mixed inputs have only diagonal
    # populations (physical purity 5/8). Equal moments cannot distinguish them.
    errors["density"] = assert_photon_array(state.dm(), density_reference, atol, "density")
    errors["trace"] = assert_photon_array(state.trace(), np.ones(leading), atol, "trace")
    errors["reduced"] = max(
        assert_photon_array(
            state.reduced_dm(mode), reduced_reference[..., mode, :, :], atol, "reduced density"
        )
        for mode in range(2)
    )
    errors["mean"], errors["variance"] = assert_variance_moments(state, labels, atol)
    request.node.user_properties.extend(
        [
            ("effective_dtype", str(data.dtype)),
            ("effective_storage", "ket" if state.is_pure else "density"),
            ("effective_hbar", str(state.hbar)),
            ("effective_cutoff", str(state.cutoff_dim)),
            ("effective_batch_size", str(batch_size)),
            ("atol", str(atol)),
        ]
        + [("maximum_" + key + "_error", str(value)) for key, value in errors.items()]
    )


@pytest.mark.parametrize("labels", ["A", "B", ("A", "B")])
def test_variance_oracle_accepts_reference(labels):
    mean, variance = expected_moments(labels)
    state = SimpleNamespace(mean_photon=lambda mode: (mean[..., mode], variance[..., mode]))
    assert assert_variance_moments(state, labels, 1e-15) == [0.0, 0.0]


@pytest.mark.parametrize("observable", [0, 1], ids=["mean", "variance"])
@pytest.mark.parametrize("shape_error", ["scalar", "extra_axis", "unbatched_vector"])
def test_variance_oracle_rejects_shape(observable, shape_error):
    labels = "A" if shape_error == "unbatched_vector" else ("A", "B")
    expected = expected_moments(labels)

    def moments(mode):
        values = [x[..., mode] for x in expected]
        value = np.asarray(values[observable])
        values[observable] = value[0] if shape_error == "scalar" else value[..., None]
        return values

    with pytest.raises(AssertionError, match=("mean", "variance")[observable] + " shape"):
        assert_variance_moments(SimpleNamespace(mean_photon=moments), labels, 1e-15)


@pytest.mark.parametrize("observable", [0, 1], ids=["mean", "variance"])
@pytest.mark.parametrize("corruption", ["copy", "reverse", "mode", "zero", "second", "nan"])
def test_variance_oracle_rejects_member_errors(observable, corruption):
    labels = ("A", "B")
    expected = expected_moments(labels)

    def moments(mode):
        values = [x[:, mode].copy() for x in expected]
        value = values[observable]
        if corruption == "copy":
            value[:] = value[0]
        elif corruption == "reverse":
            value[:] = value[::-1]
        elif corruption == "mode":
            value[:] = expected[observable][:, 1 - mode]
        elif corruption == "zero":
            value[:] = 0
        elif corruption == "second":
            value[:] = [SECONDS[x][mode] for x in labels]
        else:
            value[1] = np.nan
        return values

    with pytest.raises(AssertionError):
        assert_variance_moments(SimpleNamespace(mean_photon=moments), labels, 1e-15)


@pytest.mark.parametrize("pure", [True, False])
def test_variance_density_oracle_rejects_wrong_coherence(pure):
    """Moment-equal mixtures and projectors are not interchangeable states."""
    _, correct, _ = reference("A", pure, 3)
    _, wrong, _ = reference("A", not pure, 3)
    assert assert_photon_array(correct.copy(), correct, 1e-15, "density") == 0
    with pytest.raises(AssertionError):
        assert_photon_array(wrong, correct, 1e-15, "density")
