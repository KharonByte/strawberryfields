# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Bounded structural compatibility for XIR definition expansion."""

from decimal import Decimal
import sys

import pytest
import xir

from strawberryfields import program_utils
from strawberryfields.io import xir_io

pytestmark = pytest.mark.frontend


def snapshot(value):
    """Record values and container/object identities without retaining mutable views."""
    if isinstance(value, dict):
        return id(value), tuple((snapshot(k), snapshot(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return type(value).__name__, id(value), tuple(snapshot(v) for v in value)
    if hasattr(value, "__dict__"):
        return type(value).__name__, id(value), snapshot(vars(value))
    return type(value).__name__, repr(value)


def expand(prog):
    """The helper preserves its input on success and failure, including repeat calls."""
    before = snapshot(prog)
    try:
        return xir_io.get_expanded_statements(prog)
    finally:
        assert snapshot(prog) == before


def values(statements):
    return [(s.name, s.params, s.wires) for s in statements]


def chain(parameterized):
    """Exactly 1100 single-child definitions and one finite leaf."""
    prog = xir.Program()
    params = ["x"] if parameterized else []
    for i in range(1100):
        name = f"G{i + 1}" if i < 1099 else ("Rgate" if parameterized else "Vacuum")
        prog.add_gate(f"G{i}", params, ["w"], [xir.Statement(name, params, ("w",))])
    prog.add_statement(xir.Statement("G0", [0.25] if parameterized else [], (0,)))
    return prog


@pytest.mark.parametrize("parameterized", [False, True])
@pytest.mark.parametrize("route", ["helper", "ordinary", "tdm"])
def test_deep_single_leaf(parameterized, route):
    """Deep finite definitions expand without changing the interpreter recursion limit."""
    limit = sys.getrecursionlimit()
    assert limit < 1100, "Fixture must exceed the unchanged normal recursion limit"
    prog = chain(parameterized)
    expected_name = "Rgate" if parameterized else "Vacuum"
    expected_params = [0.25] if parameterized else []
    if route == "helper":
        try:
            for _ in range(2):
                assert values(expand(prog)) == [(expected_name, expected_params, (0,))]
        finally:
            assert sys.getrecursionlimit() == limit
        return
    converter = xir_io.from_xir if route == "ordinary" else xir_io.from_xir_to_tdm
    if route == "tdm":
        prog.add_option("N", 1)
        prog.add_constant("p0", [0.0])
    assert program_utils.Program_current_context is None
    try:
        result = converter(prog)
        assert len(result.circuit) == 1
        command = result.circuit[0]
        assert type(command.op).__name__ == expected_name
        assert command.op.p == expected_params
        assert [r.ind for r in command.reg] == [0]
    finally:
        assert program_utils.Program_current_context is None
        assert sys.getrecursionlimit() == limit
        recovery = xir.Program()
        recovery.add_statement(xir.Statement("Vacuum", [], (0,)))
        if route == "tdm":
            recovery.add_option("N", 1)
            recovery.add_constant("p0", [0.0])
        assert len(converter(recovery).circuit) == 1
        assert program_utils.Program_current_context is None


def test_asymmetric_repeated_dag():
    """Literal hand expansion distinguishes order, parameter and wire substitutions."""
    prog = xir.Program()
    prog.add_gate("Leaf", ["x", "y"], ["w"], [xir.Statement("Sgate", ["y", "x"], ("w",))])
    prog.add_gate(
        "Pair",
        ["a", "b"],
        ["u", "v"],
        [xir.Statement("Leaf", ["a", "b"], ("v",)), xir.Statement("Rgate", ["a"], ("u",))],
    )
    prog.add_gate(
        "Outer",
        ["p", "q"],
        ["left", "right"],
        [
            xir.Statement("Pair", ["q", "p"], ("right", "left")),
            xir.Statement("Pair", ["p", "q"], ("left", "right")),
        ],
    )
    first = xir.Statement("Vacuum", [], (9,))
    last = xir.Statement("MeasureFock", {"shots": 1}, (8,))
    for stmt in [
        first,
        xir.Statement("Outer", [0.25, 0.75], (2, 5)),
        xir.Statement("Outer", [1.25, 1.75], (6, 7)),
        last,
    ]:
        prog.add_statement(stmt)
    expected = [
        ("Vacuum", [], (9,)),
        ("Sgate", [0.25, 0.75], (2,)),
        ("Rgate", [0.75], (5,)),
        ("Sgate", [0.75, 0.25], (5,)),
        ("Rgate", [0.25], (2,)),
        ("Sgate", [1.25, 1.75], (6,)),
        ("Rgate", [1.75], (7,)),
        ("Sgate", [1.75, 1.25], (7,)),
        ("Rgate", [1.25], (6,)),
        ("MeasureFock", {"shots": 1}, (8,)),
    ]
    one, two = expand(prog), expand(prog)
    assert values(one) == values(two) == expected
    assert one[0] is two[0] is first
    assert one[-1] is two[-1] is last
    assert len({id(s) for s in one[1:-1] + two[1:-1]}) == 16


def test_empty_program():
    prog = xir.Program()
    assert expand(prog) == expand(prog) == []


def test_empty_body_and_nested_empty():
    prog = xir.Program()
    prog.add_gate("Empty", ["x"], ["w"], [])
    direct = xir.Statement("Empty", [0.25], (3,), inverse=True, ctrl_wires=(7,))
    prog.add_gate("Wrapper", ["p"], ["v"], [xir.Statement("Empty", ["p"], ("v",))])
    prog.add_statement(direct)
    prog.add_statement(xir.Statement("Wrapper", [0.75], (5,)))
    for _ in range(2):
        result = expand(prog)
        assert result[0] is direct
        assert values(result) == [("Empty", [0.25], (3,)), ("Empty", [0.75], (5,))]
        assert result[1] is not prog.gates["Wrapper"][0]


def test_leaf_identity_and_reconstructed_defaults():
    prog = xir.Program()
    direct = xir.Statement(
        "Leaf", [Decimal("0.25")], (3,), inverse=True, ctrl_wires=(4,), use_floats=False
    )
    inner = xir.Statement("Leaf", ["x"], ("w",), inverse=True, ctrl_wires=("c",), use_floats=False)
    prog.add_gate("Macro", ["x"], ["w"], [inner])
    prog.add_statement(direct)
    for _ in range(2):
        prog.add_statement(xir.Statement("Macro", [Decimal("0.75")], (5,), use_floats=False))
    result = expand(prog)
    assert result[0] is direct
    assert direct.is_inverse and direct.ctrl_wires == (4,) and not direct.use_floats
    assert isinstance(direct.params[0], Decimal)
    for stmt in result[1:]:
        assert stmt is not inner
        assert not stmt.is_inverse and stmt.ctrl_wires == () and stmt.use_floats
        assert values([stmt]) == [("Leaf", [0.75], (5,))]
    assert result[1] is not result[2]
    assert result[1]._params is not result[2]._params


@pytest.mark.parametrize("extra", [False, True])
def test_zip_truncation_and_dictionary_iteration(extra):
    prog = xir.Program()
    prog.add_gate("M", ["a", "unused"], ["u", "unused"], [xir.Statement("Leaf", ["a"], ("u",))])
    # Legacy iteration uses dictionary keys, not its values; no keyword interpretation.
    params = {"first": 0.25, "second": 0.75, "extra": 1.25} if extra else {"first": 0.25}
    wires = (4, 5, 6) if extra else (4,)
    prog.add_statement(xir.Statement("M", params, wires))
    assert values(expand(prog)) == [("Leaf", ["first"], (4,))]


@pytest.mark.parametrize(
    "case, error, args",
    [
        ("wire_first", KeyError, ("missing_wire",)),
        ("param", KeyError, ("missing_param",)),
        ("unhashable", TypeError, ("unhashable type: 'list'",)),
        (
            "missing_declaration",
            ValueError,
            ("No gate declarations with the name 'M' were found.",),
        ),
        ("postorder", KeyError, ("inner_missing",)),
    ],
)
def test_mapping_failures(case, error, args):
    prog = xir.Program()
    if case == "postorder":
        prog.add_gate("Inner", [], ["i"], [xir.Statement("Leaf", [], ("inner_missing",))])
        children = [
            xir.Statement("Leaf", [], ("outer_missing",)),
            xir.Statement("Inner", [], ("w",)),
        ]
    else:
        wires = ("missing_wire",) if case == "wire_first" else ("w",)
        params = [["x"]] if case == "unhashable" else ["missing_param"]
        children = [xir.Statement("Leaf", params, wires)]
    prog.add_gate("M", ["x"], ["w"], children)
    if case == "missing_declaration":
        prog.declarations["gate"].clear()
    prog.add_statement(xir.Statement("M", [0.25], (0,)))
    for _ in range(2):
        with pytest.raises(error) as exc:
            expand(prog)
        assert exc.value.args == args


@pytest.mark.parametrize("cycle", ["self", "mutual", "deep"])
def test_cycle_precedes_mapping(cycle):
    prog = xir.Program()
    prog.add_gate("Bad", [], ["w"], [xir.Statement("Leaf", [], ("missing",))])
    prog.add_statement(xir.Statement("Bad", [], (0,)))
    depth = 1100 if cycle == "deep" else (2 if cycle == "mutual" else 1)
    for i in range(depth):
        prog.add_gate(f"C{i}", [], ["w"], [xir.Statement(f"C{(i + 1) % depth}", [], ("w",))])
    prog.add_statement(xir.Statement("C0", [], (0,)))
    for _ in range(2):
        with pytest.raises(ValueError) as exc:
            expand(prog)
        assert exc.value.args == ("Recursive XIR gate definition detected: 'C0'.",)


def test_unused_cycle_and_repeated_leaf():
    prog = xir.Program()
    prog.add_gate("Loop", [], ["w"], [xir.Statement("Loop", [], ("w",))])
    leaf = xir.Statement("Leaf", [], (0,))
    prog.add_statement(leaf)
    prog.add_statement(leaf)
    result = expand(prog)
    assert result[0] is result[1] is leaf
