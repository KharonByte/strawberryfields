# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rejection and compatibility checks for serialized circuit imports."""

import blackbird
import pytest
import xir

from strawberryfields import io, ops, program_utils
from strawberryfields.program import Program
from strawberryfields.io import blackbird_io, xir_io

pytestmark = pytest.mark.frontend


def script(body, ir, tdm):
    """Wrap a tiny circuit in the requested serialization metadata."""
    if ir == "xir":
        prefix = "options: _type_: tdm; N: 2; end; constants: p0: [0.0]; end; "
        return (prefix if tdm else "") + body
    prefix = "name test\nversion 1.0\n"
    if tdm:
        prefix += "type tdm (temporal_modes=1)\nfloat array p0 =\n    0.0\n"
    return prefix + body + "\n"


def convert(text, ir, tdm, entry):
    """Exercise the public loader, dispatcher or direct converter."""
    if entry == "loads":
        return io.loads(text, ir=ir)
    parsed = xir.parse_script(text) if ir == "xir" else blackbird.loads(text)
    if entry == "to_program":
        return io.to_program(parsed)
    if ir == "xir":
        converter = xir_io.from_xir_to_tdm if tdm else xir_io.from_xir
    else:
        converter = blackbird_io.from_blackbird_to_tdm if tdm else blackbird_io.from_blackbird
    return converter(parsed)


def assert_recovered(ir, tdm, entry):
    """Failed conversion must not leave a context active."""
    assert program_utils.Program_current_context is None
    body = "Vacuum | [0];" if ir == "xir" else "Vacuum() | 0"
    assert len(convert(script(body, ir, tdm), ir, tdm, entry).circuit) == 1
    assert program_utils.Program_current_context is None


@pytest.mark.parametrize("entry", ["loads", "to_program", "direct"])
@pytest.mark.parametrize("tdm", [False, True])
@pytest.mark.parametrize(
    "definitions",
    [
        "gate A[w]: A | [w]; end;",
        "gate A[w]: B | [w]; end; gate B[w]: A | [w]; end;",
        "gate A[w]: B | [w]; end; gate B[w]: C | [w]; end; gate C[w]: B | [w]; end;",
    ],
)
def test_recursive_xir_rejected(definitions, tdm, entry):
    """Finite expansion cannot represent a reachable recursive definition."""
    text = script(definitions + " A | [0];", "xir", tdm)
    with pytest.raises(ValueError, match="Recursive XIR gate definition"):
        convert(text, "xir", tdm, entry)
    assert_recovered("xir", tdm, entry)


def test_cycle_preflight_uses_explicit_stack():
    """A cycle beyond the normal recursion depth still has the same error."""
    prog = xir.Program()
    for i in range(1100):
        prog.add_gate(f"G{i}", [], ["w"], [xir.Statement(f"G{i + 1}", [], ("w",))])
    prog.add_gate("G1100", [], ["w"], [xir.Statement("G0", [], ("w",))])
    prog.add_statement(xir.Statement("G0", [], (0,)))
    with pytest.raises(ValueError, match="Recursive XIR gate definition"):
        xir_io.get_expanded_statements(prog)


@pytest.mark.parametrize("tdm", [False, True])
@pytest.mark.parametrize("use_floats", [False, True])
def test_shared_gates_and_unused_cycle(tdm, use_floats):
    """Hand-expanded leaves preserve order, values and swapped wire mapping."""
    text = script(
        "gate Loop[w]: Loop | [w]; end; "
        "gate Leaf(x)[w]: Rgate(x) | [w]; end; "
        "gate Pair(a, b)[u, v]: Leaf(a) | [v]; Leaf(b) | [u]; end; "
        "Pair(0.25, 0.5) | [0, 1]; Pair(0.75, 1.0) | [1, 0];",
        "xir",
        tdm,
    )
    prog = io.to_program(xir.parse_script(text, use_floats=use_floats))
    assert [cmd.op.__class__.__name__ for cmd in prog.circuit] == ["Rgate"] * 4
    assert [cmd.op.p for cmd in prog.circuit] == [[0.25], [0.5], [0.75], [1.0]]
    assert [[reg.ind for reg in cmd.reg] for cmd in prog.circuit] == [[1], [0], [0], [1]]


@pytest.mark.parametrize("entry", ["loads", "to_program", "direct"])
@pytest.mark.parametrize("tdm", [False, True])
@pytest.mark.parametrize("ir", ["blackbird", "xir"])
@pytest.mark.parametrize("keyword", [False, True])
def test_serialized_new_rejected(mocker, entry, tdm, ir, keyword):
    """Reject New before calling the helper or allocating extra registers."""
    new = mocker.spy(ops, "New")
    add = mocker.spy(Program, "_add_subsystems")
    argument = ("n: 2" if ir == "xir" else "n=2") if keyword else "2"
    body = f"New({argument}) | " + ("[0];" if ir == "xir" else "0")
    with pytest.raises(ValueError, match="New is not supported in serialized programs"):
        convert(script(body, ir, tdm), ir, tdm, entry)
    new.assert_not_called()
    assert add.call_count == 1
    assert add.call_args.args[1] == (2 if ir == "xir" and tdm else 1)
    assert_recovered(ir, tdm, entry)


@pytest.mark.parametrize("entry", ["loads", "to_program", "direct"])
@pytest.mark.parametrize("tdm", [False, True])
def test_nested_serialized_new_rejected(mocker, entry, tdm):
    """Expanding a macro must not make the allocation helper importable."""
    new = mocker.spy(ops, "New")
    add = mocker.spy(Program, "_add_subsystems")
    body = "gate Allocate(n)[w]: New(n) | [w]; end; Allocate(2) | [0];"
    with pytest.raises(ValueError, match="New is not supported in serialized programs"):
        convert(script(body, "xir", tdm), "xir", tdm, entry)
    new.assert_not_called()
    assert add.call_count == 1
    assert_recovered("xir", tdm, entry)


def test_trusted_python_new_remains_supported():
    """The restriction applies only to serialized input."""
    prog = Program(1)
    with prog.context:
        refs = ops.New(2)
    assert [ref.ind for ref in refs] == [1, 2]
    assert prog.num_subsystems == 3


@pytest.mark.parametrize("tdm", [False, True])
def test_macro_named_new_can_expand_to_supported_operation(tdm):
    """A user-defined macro name is not a call to the Python helper."""
    text = script("gate New(x)[w]: Rgate(x) | [w]; end; New(0.5) | [0];", "xir", tdm)
    prog = io.loads(text, ir="xir")
    assert len(prog.circuit) == 1
    assert prog.circuit[0].op.__class__.__name__ == "Rgate"
    assert prog.circuit[0].op.p == [0.5]
