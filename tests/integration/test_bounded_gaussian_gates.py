# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Bounded finite-state P/CX/CZ checks, separate from the existing r=3 tests.

The references follow the declared unitary generators and [x, p] = i*hbar.
These small forward-moment cases do not establish general Fock/TF support.
"""

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

STRENGTHS = (-0.25, 0.0, 0.25)
UNITS = (1.0, 2.0)
CUTOFF = 20
DISPLACEMENTS = np.array([[0.24, -0.18, 0.16, 0.27], [-0.11, 0.21, 0.09, -0.14]])
SQUEEZING = np.array([0.12, -0.08])


def gate_map(gate, strength):
    """CCR-derived xxpp map, independent of the simulator decompositions."""
    transform = np.eye(2 if gate == "Pgate" else 4)
    if gate == "Pgate":
        transform[1, 0] = strength
    elif gate == "CXgate":
        transform[1, 0] = strength
        transform[2, 3] = -strength
    elif gate == "CZgate":
        transform[2, 1] = strength
        transform[3, 0] = strength
    else:
        raise ValueError(f"Unknown gate {gate}")
    return transform


def gate_reference(gate, strength, hbar, batch_size):
    """Analytic physical-displacement and symmetrized covariance references."""
    assert batch_size in (None, 2)
    modes = 1 if gate == "Pgate" else 2
    means = DISPLACEMENTS[:, [0, 2]] if modes == 1 else DISPLACEMENTS
    means = means[0] if batch_size is None else means.copy()
    r = SQUEEZING[:modes]
    covariance = (hbar / 2) * np.diag(np.exp(np.r_[-2 * r, 2 * r]))
    transform = gate_map(gate, strength)
    expected_means = means @ transform.T
    expected_covariance = transform @ covariance @ transform.T
    if batch_size is not None:
        expected_covariance = np.broadcast_to(expected_covariance, (batch_size,) + covariance.shape)
    return expected_means, expected_covariance


def prepare_gate_program(program, gate, strength, batch_size):
    """Use bounded physical X/Z displacements and unequal TF batch members."""
    assert batch_size in (None, 2)
    modes = 1 if gate == "Pgate" else 2
    means = DISPLACEMENTS[0] if batch_size is None else DISPLACEMENTS.T
    with program.context as q:
        for mode in range(modes):
            ops.Sgate(SQUEEZING[mode]) | q[mode]
            ops.Xgate(means[mode]) | q[mode]
            ops.Zgate(means[mode + 2]) | q[mode]
        getattr(ops, gate)(strength) | (q[0] if modes == 1 else (q[0], q[1]))


def assert_gate_moments(actual, expected, tolerance):
    """Require full shapes and every member; do not accept implicit broadcasting."""
    assert len(actual) == len(expected) == 2
    for value, reference in zip(actual, expected):
        assert value.shape == reference.shape
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value, reference, atol=tolerance, rtol=0)


def density_diagnostics(state, batch_size, cutoff):
    """Check the raw finite physical density without trace renormalization."""
    density = np.asarray(state.dm())
    modes = state.num_modes
    batch = () if batch_size is None else (batch_size,)
    assert density.shape == batch + (cutoff,) * (2 * modes)
    assert np.all(np.isfinite(density))
    members = density[None] if batch_size is None else density
    traces = []
    for member in members:
        matrix = (
            member if modes == 1 else member.transpose(0, 2, 1, 3).reshape(cutoff**2, cutoff**2)
        )
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=0.001, rtol=0)
        trace = np.trace(matrix)
        np.testing.assert_allclose(trace, 1, atol=1e-5, rtol=0)
        assert np.linalg.eigvalsh((matrix + matrix.conj().T) / 2).min() >= -0.001
        traces.append(trace)
    return density, np.asarray(traces)


@pytest.mark.backends("fock", "tf")
@pytest.mark.parametrize("cutoff", [CUTOFF], indirect=True)
@pytest.mark.parametrize("hbar", UNITS, indirect=True)
@pytest.mark.parametrize("strength", STRENGTHS)
class TestBoundedGaussianGates:
    """Low-energy counterparts, not replacements for the high-squeezing cases."""

    @staticmethod
    def check(
        gate,
        setup_eng,
        state_moments,
        hbar,
        cutoff,
        pure,
        batch_size,
        strength,
        tol,
        request,
    ):
        assert tol == 0.001
        assert sf.hbar == hbar
        modes = 1 if gate == "Pgate" else 2
        eng, program = setup_eng(modes)
        prepare_gate_program(program, gate, strength, batch_size)
        state = eng.run(program).state
        assert state.hbar == hbar
        assert state.cutoff_dim == cutoff
        assert state.is_pure == pure
        density, _ = density_diagnostics(state, batch_size, cutoff)
        actual = state_moments(state, batch_size)
        expected = gate_reference(gate, strength, hbar, batch_size)
        assert_gate_moments(actual, expected, tol)
        omega = np.block(
            [
                [np.zeros((modes, modes)), np.eye(modes)],
                [-np.eye(modes), np.zeros((modes, modes))],
            ]
        )
        assert np.linalg.eigvalsh(actual[1] + 0.5j * hbar * omega).min() >= -0.001
        request.node.user_properties.extend(
            [
                ("parameter_strength", str(strength)),
                ("parameter_hbar", str(hbar)),
                ("observed_density_dtype", str(density.dtype)),
                ("observed_state_pure", str(state.is_pure)),
            ]
        )

    def test_Pgate(
        self,
        setup_eng,
        state_moments,
        hbar,
        cutoff,
        pure,
        batch_size,
        strength,
        tol,
        request,
    ):
        self.check(
            "Pgate",
            setup_eng,
            state_moments,
            hbar,
            cutoff,
            pure,
            batch_size,
            strength,
            tol,
            request,
        )

    def test_CXgate(
        self,
        setup_eng,
        state_moments,
        hbar,
        cutoff,
        pure,
        batch_size,
        strength,
        tol,
        request,
    ):
        self.check(
            "CXgate",
            setup_eng,
            state_moments,
            hbar,
            cutoff,
            pure,
            batch_size,
            strength,
            tol,
            request,
        )

    def test_CZgate(
        self,
        setup_eng,
        state_moments,
        hbar,
        cutoff,
        pure,
        batch_size,
        strength,
        tol,
        request,
    ):
        self.check(
            "CZgate",
            setup_eng,
            state_moments,
            hbar,
            cutoff,
            pure,
            batch_size,
            strength,
            tol,
            request,
        )


@pytest.mark.frontend
@pytest.mark.parametrize("gate", ["Pgate", "CXgate", "CZgate"])
@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "sign",
        "covariance_transpose",
        "conjugate",
        "copy_member",
        "missing_batch",
        "nonfinite",
        "empty",
        "truncated",
    ],
)
def test_bounded_gate_oracle_rejects_corruption(gate, corruption):
    """Negative controls test observable sensitivity, not simulator correctness."""
    expected = gate_reference(gate, 0.25, 1.0, 2)
    actual = tuple(value.copy() for value in expected)
    if corruption == "sign":
        actual = gate_reference(gate, -0.25, 1.0, 2)
    elif corruption == "covariance_transpose":
        transform = gate_map(gate, 0.25)
        modes = transform.shape[0] // 2
        r = SQUEEZING[:modes]
        initial = 0.5 * np.diag(np.exp(np.r_[-2 * r, 2 * r]))
        actual = (
            actual[0],
            np.broadcast_to(transform.T @ initial @ transform, actual[1].shape),
        )
    elif corruption == "conjugate":
        actual[0][..., actual[0].shape[-1] // 2 :] *= -1
    elif corruption == "copy_member":
        actual[0][1] = actual[0][0]
    elif corruption == "missing_batch":
        actual = (actual[0][0], actual[1][0])
    elif corruption == "nonfinite":
        actual[1].flat[0] = np.nan
    elif corruption == "empty":
        actual = ()
    elif corruption == "truncated":
        actual = actual[:1]
    if corruption is None:
        assert_gate_moments(actual, expected, 0.001)
    else:
        with pytest.raises(AssertionError):
            assert_gate_moments(actual, expected, 0.001)


@pytest.mark.frontend
@pytest.mark.parametrize("gate", ["CXgate", "CZgate"])
def test_bounded_gate_oracle_rejects_mode_swap(gate):
    expected = gate_reference(gate, 0.25, 1.0, 2)
    order = [1, 0, 3, 2]
    actual = (expected[0][..., order], expected[1][..., order, :][..., :, order])
    with pytest.raises(AssertionError):
        assert_gate_moments(actual, expected, 0.001)
