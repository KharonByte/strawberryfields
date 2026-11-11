# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Exact finite-array contracts for bosonic vacuum insertion at internal hbar=2."""

from copy import deepcopy
import numpy as np
import pytest
from strawberryfields.backends.bosonicbackend import BosonicBackend
from strawberryfields.backends.bosonicbackend.bosoniccircuit import BosonicModes

pytestmark = pytest.mark.bosonic


def make(route, modes):
    if route == "backend":
        obj = BosonicBackend()
        obj.begin_circuit(modes)
        return obj, obj.circuit
    obj = BosonicModes(modes)
    return obj, obj


def insert(obj, route, operation):
    if operation == "default":
        return obj.add_mode()
    if operation == "implicit-zero":
        return obj.add_mode(0)
    peaks = {"zero": [], "one": [1], "two": [1, 1], "split": [2], "split-two": [2, 2]}[operation]
    if route == "backend":
        return obj.add_mode(len(peaks), peaks=peaks)
    return obj.add_mode(peaks)


def snapshot(circuit):
    return deepcopy(
        {
            key: getattr(circuit, key)
            for key in ("active", "nlen", "hbar", "to_xp", "from_xp", "weights", "means", "covs")
        }
    )


def check(circuit, before, modes, copies=1):
    """Indexed oracle: old-major components tensor independent vacuum copies.

    Each old term is multiplied by copies identical terms of weight 1/copies.
    No insertion implementation, Kronecker product or block-diagonal helper
    supplies the reference. Dyadic entries make every comparison exact.
    """
    if modes == 0:
        for key, expected in before.items():
            actual = getattr(circuit, key)
            if isinstance(expected, np.ndarray):
                assert actual.shape == expected.shape
                assert actual.dtype == expected.dtype
                assert actual.tobytes() == expected.tobytes()
                np.testing.assert_array_equal(actual, expected)
            else:
                assert actual == expected
        assert circuit.get_modes() == [x for x in before["active"] if x is not None]
        return
    old = before["nlen"]
    total = old + modes
    terms = len(before["weights"])
    weights = np.empty(terms * copies, dtype=np.complex128)
    means = np.zeros((terms * copies, 2 * total), dtype=np.complex128)
    covs = np.zeros((terms * copies, 2 * total, 2 * total), dtype=np.complex128)
    for a in range(terms):
        for b in range(copies):
            row = a * copies + b
            weights[row] = before["weights"][a] / copies
            for i in range(2 * old):
                means[row, i] = before["means"][a, i]
                for j in range(2 * old):
                    covs[row, i, j] = before["covs"][a, i, j]
            for i in range(2 * old, 2 * total):
                covs[row, i, i] = 1
    for name, expected in (("weights", weights), ("means", means), ("covs", covs)):
        actual = getattr(circuit, name)
        assert actual.shape == expected.shape
        assert actual.dtype == np.dtype("complex128")
        np.testing.assert_array_equal(actual, expected)
    active = before["active"] + list(range(old, total))
    assert circuit.active == active
    assert circuit.nlen == total
    assert circuit.hbar == before["hbar"] == 2
    assert circuit.get_modes() == [x for x in active if x is not None]
    assert len(set(circuit.get_modes())) == len(circuit.get_modes())
    for index in range(old, total):
        assert circuit.active[index] == index
    to_xp = list(range(0, 2 * total, 2)) + list(range(1, 2 * total, 2))
    from_xp = [index for i in range(total) for index in (i, total + i)]
    np.testing.assert_array_equal(circuit.to_xp, to_xp)
    np.testing.assert_array_equal(circuit.from_xp, from_xp)
    np.testing.assert_array_equal(np.asarray(circuit.to_xp)[circuit.from_xp], range(2 * total))
    np.testing.assert_array_equal(np.asarray(circuit.from_xp)[circuit.to_xp], range(2 * total))


VACUUM_CASES = [
    (route, initial, operation)
    for route in ("backend", "low")
    for initial in (1, 2)
    for operation in (
        ("default", "implicit-zero", "zero", "one", "two")
        if route == "backend"
        else ("default", "zero", "one", "two")
    )
]


@pytest.mark.parametrize(
    "route,initial,operation", VACUUM_CASES, ids=[f"{r}-{n}-{op}" for r, n, op in VACUUM_CASES]
)
def test_vacuum_insertion(route, initial, operation):
    obj, circuit = make(route, initial)
    before = snapshot(circuit)
    assert insert(obj, route, operation) is None
    count = 0 if operation in ("zero", "implicit-zero") else 2 if operation == "two" else 1
    check(circuit, before, count)


def witness(circuit, hole):
    """F1 convex displaced Gaussian mixture, or F2 with a retired vacuum slot.

    V=S S^T for the symplectic S specified in the reviewed fixture. These
    literal dyadic entries are independent of simulator preparation routines.
    """
    v = [[4, 0, 0, 1], [0, 5 / 16, 1 / 4, 0], [0, 1 / 4, 1, 0], [1, 0, 0, 5 / 4]]
    mu = [[1 / 2, -1 / 4, 3 / 4, 1 / 8], [-1 / 2, 3 / 4, -1 / 4, -1 / 8]]
    circuit.weights = np.array([1] if hole else [1 / 4, 3 / 4], dtype=np.complex128)
    circuit.means = np.zeros((1, 6) if hole else (2, 4), dtype=np.complex128)
    circuit.covs = np.zeros((1, 6, 6) if hole else (2, 4, 4), dtype=np.complex128)
    positions = [0, 1, 4, 5] if hole else [0, 1, 2, 3]
    for a in range(len(circuit.weights)):
        for i, qi in enumerate(positions):
            circuit.means[a, qi] = mu[a][i]
            for j, qj in enumerate(positions):
                circuit.covs[a, qi, qj] = v[i][j]
    if hole:
        circuit.active = [0, None, 2]
        circuit.covs[0, 2, 2] = circuit.covs[0, 3, 3] = 1


WITNESS_CASES = [
    (r, f, op)
    for r in ("backend", "low")
    for f, ops in (
        ("F1", ("zero", "default", "one", "split", "split-two")),
        ("F2", ("zero", "two")),
    )
    for op in ops
]


@pytest.mark.parametrize(
    "route,fixture,operation", WITNESS_CASES, ids=[f"{r}-{f}-{op}" for r, f, op in WITNESS_CASES]
)
def test_correlated_insertion(route, fixture, operation):
    obj, circuit = make(route, 3 if fixture == "F2" else 2)
    witness(circuit, fixture == "F2")
    before = snapshot(circuit)
    assert insert(obj, route, operation) is None
    count = 0 if operation == "zero" else 2 if operation in ("two", "split-two") else 1
    copies = 4 if operation == "split-two" else 2 if operation == "split" else 1
    check(circuit, before, count, copies)


@pytest.mark.parametrize("route", ["backend", "low"])
def test_delete_then_insert(route):
    obj, circuit = make(route, 3)
    obj.del_mode([1])
    assert circuit.active == [0, None, 2]
    for operation, count in (("zero", 0), ("two", 2), ("one", 1)):
        before = snapshot(circuit)
        insert(obj, route, operation)
        check(circuit, before, count)
    assert circuit.active == [0, None, 2, 3, 4, 5]
    obj.del_mode([3, 4, 5])
    assert circuit.active == [0, None, 2, None, None, None]
    assert circuit.get_modes() == [0, 2]


@pytest.mark.parametrize("route", ["backend", "low"])
def test_consecutive_correlated_insertions(route):
    obj, circuit = make(route, 2)
    witness(circuit, False)
    for operation, count, copies in (("split", 1, 2), ("split-two", 2, 4), ("default", 1, 1)):
        before = snapshot(circuit)
        insert(obj, route, operation)
        check(circuit, before, count, copies)
