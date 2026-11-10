# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Bounded correlated Gaussian preparation, distinct from the random-state tests.

References use the canonical symplectic moment transformation (Weedbrook et al.,
arXiv:1110.3234v1, Eqs. 24-25). These finite-output moment checks do not establish
arbitrary state fidelity, native covariance batching, or wider backend support.
"""

from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache
import json
import os

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

CUTOFF = 20
REFERENCE_ERROR = 1e-12
MOMENT_LIMIT = 1e-4 - REFERENCE_ERROR


@dataclass(frozen=True)
class GaussianCase:
    backend: str
    target: str
    initial_pure: bool
    hbar: float
    batch: object = None

    @property
    def identity(self):
        profile = "ordinary" if self.batch is None else "replicated2"
        return (
            f"{self.backend}-{self.target}-initial{self.initial_pure}" f"-hbar{self.hbar}-{profile}"
        )


def cases():
    """Follow the existing environment selection; never accept covariance arrays."""
    batch = int(os.environ["BATCHSIZE"]) if "BATCHSIZE" in os.environ else None
    if batch is None and bool(int(os.environ.get("BATCHED", "0"))):
        batch = 2
    assert batch in (None, 2)
    for backend in ("gaussian", "fock", "tf"):
        for target in ("pure", "mixed"):
            for initial in (True, False):
                for units in (1.0, 1.7):
                    case = GaussianCase(
                        backend, target, initial, units, batch if backend == "tf" else None
                    )
                    marks = [
                        getattr(pytest.mark, backend),
                        getattr(pytest.mark, "pure" if initial else "mixed"),
                    ]
                    yield pytest.param(case, id=case.identity, marks=marks)


def decimal_reference(target, units, precision):
    """Scalar reference with exact decimal inputs and bounded Taylor remainders."""
    with localcontext() as context:
        context.prec = precision
        r, angle, h = Decimal("0.12"), Decimal("0.31"), Decimal(str(units))
        c, s = (r.exp() + (-r).exp()) / 2, (r.exp() - (-r).exp()) / 2
        cosine = cosine_term = Decimal(1)
        sine = sine_term = angle
        for k in range(1, 51):
            cosine_term *= -(angle * angle) / Decimal((2 * k - 1) * (2 * k))
            sine_term *= -(angle * angle) / Decimal((2 * k) * (2 * k + 1))
            cosine += cosine_term
            sine += sine_term
        n0, n1 = (
            (Decimal(0), Decimal(0)) if target == "pure" else (Decimal("0.03"), Decimal("0.07"))
        )
        v0, v1 = h * (n0 + Decimal("0.5")), h * (n1 + Decimal("0.5"))
        a, b, cross = c * c * v0 + s * s * v1, s * s * v0 + c * c * v1, c * s * (v0 + v1)
        x, y, zero = cross * cosine, cross * sine, Decimal(0)
        covariance = [[a, x, zero, y], [x, b, y, zero], [zero, y, a, -x], [y, zero, -x, b]]
        means = [(2 * h).sqrt() * Decimal(value) for value in ("0.08", "-0.05", "0.03", "0.06")]
        return means, covariance


@lru_cache(maxsize=None)
def certified_reference(target, units):
    """Check the scalar oracle against higher precision and an explicit S T S^T."""
    assert target in ("pure", "mixed") and units in (1.0, 1.7)
    low = decimal_reference(target, units, 80)
    high = decimal_reference(target, units, 120)
    low_flat = low[0] + sum(low[1], [])
    high_flat = high[0] + sum(high[1], [])
    with localcontext() as context:
        context.prec = 120
        discrepancy = max(abs(a - b) for a, b in zip(low_flat, high_flat))
    assert discrepancy <= Decimal("1e-60")
    means, covariance = (np.asarray(value, dtype=np.float64) for value in high)
    c, s, t, u = np.cosh(0.12), np.sinh(0.12), np.cos(0.31), np.sin(0.31)
    squeeze = np.array([[c, s, 0, 0], [s, c, 0, 0], [0, 0, c, -s], [0, 0, -s, c]])
    rotation = np.array([[t, 0, -u, 0], [0, 1, 0, 0], [u, 0, t, 0], [0, 0, 0, 1]])
    transform = rotation @ squeeze
    populations = np.array([0.0, 0.0] if target == "pure" else [0.03, 0.07])
    thermal = units * np.diag(np.tile(populations + 0.5, 2))
    input_covariance = transform @ thermal @ transform.T
    input_means = np.sqrt(2 * units) * np.array([0.08, -0.05, 0.03, 0.06])
    omega = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    np.testing.assert_allclose(transform @ omega @ transform.T, omega, atol=REFERENCE_ERROR, rtol=0)
    assert np.linalg.eigvalsh(input_covariance + 0.5j * units * omega).min() >= -REFERENCE_ERROR
    mean_error = float(np.max(np.abs(input_means - means)) / np.sqrt(units))
    covariance_error = float(np.max(np.abs(input_covariance - covariance)) / units)
    assert max(mean_error, covariance_error) <= REFERENCE_ERROR
    receipt = {
        "decimal_disagreement": str(discrepancy),
        "input_mean_error": mean_error,
        "input_covariance_error": covariance_error,
    }
    return input_means, input_covariance, means, covariance, receipt


def expected_moments(case):
    _, _, means, covariance, _ = certified_reference(case.target, case.hbar)
    if case.batch is not None:
        means = np.broadcast_to(means, (case.batch, 4))
        covariance = np.broadcast_to(covariance, (case.batch, 4, 4))
    return means, covariance


def assert_moments(actual, expected, units):
    """Check each observable separately, with no implicit shape broadcasting."""
    assert len(actual) == len(expected) == 2
    for value, reference, scale in zip(actual, expected, (np.sqrt(units), units)):
        assert value.shape == reference.shape
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value / scale, reference / scale, atol=MOMENT_LIMIT, rtol=0)


def density_diagnostics(density, cutoff, units, batch):
    """Inspect the raw density and complex moments before any real-part extraction."""
    shape = (() if batch is None else (batch,)) + (cutoff,) * 4
    assert density.shape == shape and np.all(np.isfinite(density))
    members = density[None] if batch is None else density
    matrix = members.transpose(0, 1, 3, 2, 4).reshape((-1, cutoff**2, cutoff**2))
    difference = matrix - matrix.swapaxes(-1, -2).conj()
    traces = np.trace(matrix, axis1=-2, axis2=-1)
    hermitian = (matrix + matrix.swapaxes(-1, -2).conj()) / 2
    eigenvalues = np.linalg.eigvalsh(hermitian)
    ladder = np.diag(np.sqrt(np.arange(1, cutoff + 1)), 1)
    quadratures = np.sqrt(units / 2) * np.array([ladder + ladder.T, -1j * (ladder - ladder.T)])
    first, second = [], []
    marginals = (np.einsum("...abcc->...ab", members), np.einsum("...aacd->...cd", members))
    for marginal in marginals:
        for left in quadratures:
            first.append(np.einsum("...ij,ji->...", marginal, left[:cutoff, :cutoff]))
            for right in quadratures:
                product = (left @ right + right @ left) / 2
                second.append(np.einsum("...ij,ji->...", marginal, product[:cutoff, :cutoff]))
    for left in quadratures:
        for right in quadratures:
            product = np.kron(left[:cutoff, :cutoff], right[:cutoff, :cutoff])
            second.append(np.einsum("...ij,ji->...", matrix, product))
    return {
        "trace_real": traces.real.tolist(),
        "trace_imag": traces.imag.tolist(),
        "trace_deficiency": np.maximum(1 - traces.real, 0).tolist(),
        "trace_excess": np.maximum(traces.real - 1, 0).tolist(),
        "hermiticity_error": float(np.max(np.abs(difference))),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "imaginary_mean": float(np.max(np.abs(np.asarray(first).imag)) / np.sqrt(units)),
        "imaginary_second": float(np.max(np.abs(np.asarray(second).imag)) / units),
    }


def assert_density_diagnostics(values, final):
    assert values["hermiticity_error"] <= 1e-7
    assert values["minimum_eigenvalue"] >= -1e-6
    assert values["imaginary_mean"] <= 1e-6
    assert values["imaginary_second"] <= 1e-6
    if final:
        assert np.max(np.abs(values["trace_imag"])) <= 1e-7
        assert np.max(np.abs(np.asarray(values["trace_real"]) - 1)) <= 1e-6


def evaluate_case(engine, program, case, cutoff, moment_reader, final=True, receipt=None):
    """Shared fixed preparation for the maintained test and retained cutoff study."""
    record = {} if receipt is None else receipt
    assert sf.hbar == case.hbar and case.batch in (None, 2)
    input_means, input_cov, _, _, reference_receipt = certified_reference(case.target, case.hbar)
    record.update(case_id=case.identity, cutoff=cutoff, reference=reference_receipt)
    with program.context as q:
        ops.Gaussian(input_cov.copy(), input_means.copy(), decomp=True) | q
    state = engine.run(program, compile_options={"optimize": False}).state
    names = [type(command.op).__name__ for command in engine.run_progs[-1].circuit]
    record["compiled_operations"] = names
    assert not {"Gaussian", "GaussianTransform", "Interferometer"}.intersection(names)
    assert {"Sgate", "BSgate", "Dgate"}.issubset(names)
    assert names.count("Vacuum" if case.target == "pure" else "Thermal") == 2
    phases = [
        float(command.op.p[1])
        for command in engine.run_progs[-1].circuit
        if type(command.op).__name__ == "Dgate"
    ]
    np.testing.assert_allclose(
        sorted(phases), [0, 0, np.pi / 2, np.pi / 2], atol=REFERENCE_ERROR, rtol=0
    )
    assert state.hbar == case.hbar and state.num_modes == 2
    assert list(state.mode_names) == [0, 1]
    if case.backend == "gaussian":
        assert case.batch is None and state.is_pure == (case.target == "pure")
        assert np.asarray(state.means()).dtype == np.float64
        assert np.asarray(state.cov()).dtype == np.float64
        representation, dtype = "gaussian", "float64"
    else:
        # General Gaussian preparation replaces individual modes, forcing density storage.
        assert not state.is_pure and state.cutoff_dim == cutoff
        density = np.asarray(state.dm())
        dtype = "complex64" if case.backend == "tf" else "complex128"
        assert str(density.dtype) == dtype
        assert np.asarray(state.data).shape == density.shape
        if case.backend == "tf":
            assert state.batched == (case.batch is not None)
        record["density"] = density_diagnostics(density, cutoff, case.hbar, case.batch)
        assert_density_diagnostics(record["density"], final)
        representation = "density"
    actual = moment_reader(state, case.batch)
    expected = expected_moments(case)
    record.update(
        representation=representation,
        dtype=dtype,
        observed_pure=bool(state.is_pure),
        means=np.asarray(actual[0]).tolist(),
        covariance=np.asarray(actual[1]).tolist(),
        mean_error=float(np.max(np.abs(actual[0] - expected[0])) / np.sqrt(case.hbar)),
        covariance_error=float(np.max(np.abs(actual[1] - expected[1])) / case.hbar),
    )
    if final:
        assert_moments(actual, expected, case.hbar)
    return record


@pytest.fixture
def pure(case):
    return case.initial_pure


@pytest.fixture
def hbar(case, monkeypatch):
    monkeypatch.setattr(sf, "hbar", case.hbar)
    return case.hbar


@pytest.fixture
def cutoff():
    return CUTOFF


@pytest.fixture
def batch_size(case):
    return case.batch


@pytest.fixture
def setup_backend_pars(case, hbar, cutoff, pure, batch_size):
    return case.backend, {"cutoff_dim": cutoff, "pure": pure, "batch_size": batch_size}


@pytest.mark.backends("gaussian", "fock", "tf")
@pytest.mark.parametrize("case", list(cases()))
def test_correlated_gaussian_preparation(case, setup_eng, state_moments, cutoff, tol, request):
    assert tol == 0.001
    engine, program = setup_eng(2)
    result = evaluate_case(engine, program, case, cutoff, state_moments)
    request.node.user_properties.extend(
        [
            ("requested_hbar", str(case.hbar)),
            ("physical_target", case.target),
            ("requested_initial_pure", str(case.initial_pure)),
            ("observed_representation", result["representation"]),
            ("observed_state_pure", str(result["observed_pure"])),
            ("observed_dtype", result["dtype"]),
            ("observed_num_modes", "2"),
            ("case_id", case.identity),
            ("numerical_receipt", json.dumps(result, sort_keys=True)),
        ]
    )


@pytest.mark.frontend
@pytest.mark.parametrize("target", ["pure", "mixed"])
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "mean_shape",
        "covariance_shape",
        "zero_mean",
        "mean_sign",
        "mean_units",
        "diagonal",
        "xp_sign",
        "pp_sign",
        "quadrature_order",
        "mode_order",
        "covariance_units",
        "nan",
        "inf",
        "replica",
        "mean_replica",
    ],
)
def test_gaussian_moment_oracle(target, batch, corruption):
    case = GaussianCase("tf", target, True, 1.7, batch)
    expected = expected_moments(case)
    actual = tuple(value.copy() for value in expected)
    if corruption == "mean_shape":
        actual = (actual[0][..., :1], actual[1])
    elif corruption == "covariance_shape":
        actual = (actual[0], actual[1][..., :1, :1])
    elif corruption == "zero_mean":
        actual[0][...] = 0
    elif corruption == "mean_sign":
        actual[0][..., 2:] *= -1
    elif corruption == "mean_units":
        actual[0][...] /= np.sqrt(case.hbar)
    elif corruption == "diagonal":
        actual[1][...] *= np.eye(4)
    elif corruption in ("xp_sign", "pp_sign"):
        i, j = (0, 3) if corruption == "xp_sign" else (2, 3)
        actual[1][..., i, j] *= -1
        actual[1][..., j, i] *= -1
    elif corruption == "quadrature_order":
        order = [0, 2, 1, 3]
        actual = (actual[0], actual[1][..., order, :][..., :, order])
    elif corruption == "mode_order":
        order = [1, 0, 3, 2]
        # Pure covariance is invariant; its asymmetric means expose the whole-state swap.
        actual = (
            actual[0][..., order] if target == "pure" else actual[0],
            actual[1][..., order, :][..., :, order],
        )
    elif corruption == "covariance_units":
        actual[1][...] /= case.hbar
    elif corruption in ("nan", "inf"):
        actual[1].flat[0] = np.nan if corruption == "nan" else np.inf
    elif corruption == "replica":
        actual[1][0 if batch is not None else ...] *= 0.9
    elif corruption == "mean_replica":
        actual[0][0 if batch is not None else ...] *= 0.9
    if corruption is None:
        assert_moments(actual, expected, case.hbar)
    else:
        with pytest.raises(AssertionError):
            assert_moments(actual, expected, case.hbar)


@pytest.mark.frontend
@pytest.mark.parametrize("observable", [0, 1])
@pytest.mark.parametrize("shape", ["missing", "extra"])
def test_gaussian_replica_shape_controls(observable, shape):
    case = GaussianCase("tf", "mixed", False, 1.7, 2)
    expected = expected_moments(case)
    actual = [value.copy() for value in expected]
    actual[observable] = (
        actual[observable][0]
        if shape == "missing"
        else np.repeat(actual[observable][:1], 3, axis=0)
    )
    with pytest.raises(AssertionError):
        assert_moments(actual, expected, case.hbar)


@pytest.mark.frontend
@pytest.mark.parametrize(
    "field",
    [
        None,
        "trace_real",
        "trace_imag",
        "hermiticity_error",
        "minimum_eigenvalue",
        "imaginary_mean",
        "imaginary_second",
    ],
)
def test_gaussian_density_diagnostic_controls(field):
    density = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    density[0, 0, 0, 0] = 1
    values = density_diagnostics(density, 3, 1.7, None)
    if field is None:
        assert_density_diagnostics(values, True)
        return
    values[field] = {"trace_real": [0.99], "trace_imag": [1e-4], "minimum_eigenvalue": -1e-3}.get(
        field, 1e-3
    )
    with pytest.raises(AssertionError):
        assert_density_diagnostics(values, True)


@pytest.mark.frontend
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize(
    "corruption", [None, "first", "second", "cross", "trace", "imaginary_trace", "negative"]
)
def test_gaussian_raw_density_controls(batch, corruption):
    """Known raw matrix elements test extraction, including the second replica.

    For rho[0,1]=i*e, Im<x>/sqrt(hbar)=e/sqrt(2); rho[0,2]=i*e gives
    Im<x^2>/hbar=e/sqrt(2). The two-mode element |00><11| gives e/2.
    The 1e-15 allowance covers double arithmetic on these small three-level
    tensors; it is not the finite-backend accuracy allowance.
    """
    density = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    # Exact density of (|00> + i|10>)/sqrt(2), including relative coherence.
    density[0, 0, 0, 0] = density[1, 1, 0, 0] = 0.5
    density[0, 1, 0, 0], density[1, 0, 0, 0] = -0.5j, 0.5j
    if batch is not None:
        density = np.stack([density, density])
    member = density if batch is None else density[1]
    epsilon = 1e-4
    if corruption in ("first", "second", "cross"):
        index = {"first": (0, 1, 0, 0), "second": (0, 2, 0, 0), "cross": (0, 1, 0, 1)}[corruption]
        member[index] += 1j * epsilon
    elif corruption == "trace":
        member *= 0.99
    elif corruption == "imaginary_trace":
        member[0, 0, 0, 0] += 1j * epsilon
    elif corruption == "negative":
        member[...] = 0
        member[0, 0, 0, 0], member[1, 1, 0, 0] = 1.01, -0.01
    values = density_diagnostics(density, 3, 1.7, batch)
    if corruption is None:
        assert_density_diagnostics(values, True)
        np.testing.assert_allclose(values["trace_real"], 1, atol=1e-15, rtol=0)
        assert values["imaginary_mean"] <= 1e-15
        assert values["imaginary_second"] <= 1e-15
    else:
        with pytest.raises(AssertionError):
            assert_density_diagnostics(values, True)
        if corruption in ("first", "second", "cross"):
            field = "imaginary_mean" if corruption == "first" else "imaginary_second"
            expected = epsilon / (2 if corruption == "cross" else np.sqrt(2))
            np.testing.assert_allclose(values[field], expected, atol=1e-15, rtol=0)
            np.testing.assert_allclose(values["hermiticity_error"], epsilon, atol=1e-15, rtol=0)
        elif corruption == "trace":
            np.testing.assert_allclose(values["trace_real"][-1], 0.99, atol=1e-15, rtol=0)
        elif corruption == "imaginary_trace":
            np.testing.assert_allclose(values["trace_imag"][-1], epsilon, atol=1e-15, rtol=0)
        elif corruption == "negative":
            np.testing.assert_allclose(values["minimum_eigenvalue"], -0.01, atol=1e-15, rtol=0)
