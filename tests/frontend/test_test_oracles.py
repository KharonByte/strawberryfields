# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Test infrastructure must reject incorrect units, statistics and coherences."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

pytestmark = pytest.mark.frontend
pytest_plugins = ["pytester"]


@pytest.mark.parametrize(
    "batch_size,derivative,jacobian",
    [
        (None, 2.0, 2.0),
        (None, [2.0, -1.0], [2.0, -1.0]),
        (1, [2.0], [[2.0]]),
        (2, [2.0, 4.0], [[2.0, 0.0], [0.0, 4.0]]),
        (2, [[2.0, -1.0], [4.0, -3.0]], [[[2.0, 0.0], [-1.0, 0.0]], [[0.0, 4.0], [0.0, -3.0]]]),
        (
            3,
            [[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]],
            [
                [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 4.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]],
            ],
        ),
    ],
)
def test_independent_jacobian_accepts_analytic_entries(
    assert_independent_jacobian, batch_size, derivative, jacobian
):
    """Scalar, multiple-output and unequal output/batch axes retain their shape."""
    assert_independent_jacobian(jacobian, derivative, batch_size)


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_output_batch",
        "summed_gradient",
        "duplicated_member",
        "cross_member",
        "swapped_members",
        "transposed_axes",
        "nan",
        "zero",
        "disconnected",
    ],
)
def test_independent_jacobian_rejects_incorrect_batches(assert_independent_jacobian, corruption):
    """The Jacobian of (x**2, x) at x=[1,2,3] has no cross-member entries."""
    derivative = np.array([[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])
    jacobian = np.array(
        [
            [[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 4.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 6.0], [0.0, 0.0, 1.0]],
        ]
    )
    if corruption == "missing_output_batch":
        jacobian = jacobian[0]
    elif corruption == "summed_gradient":
        jacobian = jacobian.sum(axis=(0, 1))
    elif corruption == "duplicated_member":
        jacobian[1, 0, 1] = jacobian[2, 0, 2] = 2.0
    elif corruption == "cross_member":
        jacobian[0, 0, 1] = 1.0
    elif corruption == "swapped_members":
        jacobian = jacobian[::-1, :, ::-1]
    elif corruption == "transposed_axes":
        jacobian = jacobian.transpose(1, 0, 2)
    elif corruption == "nan":
        jacobian[0, 0, 0] = np.nan
    elif corruption == "zero":
        jacobian[:] = 0
    else:
        jacobian = None
    with pytest.raises(AssertionError):
        assert_independent_jacobian(jacobian, derivative, 3)


@pytest.mark.parametrize("jacobian", [None, 0.0, [2.0], np.nan])
def test_independent_jacobian_rejects_incorrect_scalar(assert_independent_jacobian, jacobian):
    """Enabling batches must preserve the scalar derivative contract."""
    with pytest.raises(AssertionError):
        assert_independent_jacobian(jacobian, 2.0, None)


def test_hbar_fixture_precedence_and_restoration(pytester, monkeypatch):
    """Exercise the real fixture through pytest, including failing test teardown."""
    fixture_path = Path(__file__).resolve().parents[1] / "conftest.py"
    monkeypatch.setenv("HBAR", "3.0")
    pytester.makeconftest(f"""
import importlib.util
import pytest
import strawberryfields as sf
spec = importlib.util.spec_from_file_location("unit_fixture_source", {str(fixture_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
hbar = module.hbar
sf.hbar = 4.0

@pytest.fixture(autouse=True)
def check_restoration():
    assert sf.hbar == 4.0
    yield
    assert sf.hbar == 4.0
""")
    pytester.makepyfile("""
import pytest
import strawberryfields as sf

@pytest.mark.parametrize("hbar", [0.5, 1.0], indirect=True)
def test_parameter(hbar, request):
    assert hbar == sf.hbar == request.node.callspec.params["hbar"]

def test_environment(hbar):
    assert hbar == sf.hbar == 3.0

def test_default(monkeypatch, request):
    monkeypatch.delenv("HBAR")
    assert request.getfixturevalue("hbar") == sf.hbar == 1.7

@pytest.mark.parametrize("hbar", [0.5], indirect=True)
def test_failure(hbar):
    assert hbar == sf.hbar == 0.5
    raise RuntimeError("intentional fixture teardown probe")
""")
    result = pytester.runpytest_subprocess("-q", "-p", "no:randomly", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=4, failed=1)
    result.stdout.fnmatch_lines(["*intentional fixture teardown probe*"])


@pytest.mark.parametrize("arguments", ["monkeypatch, hbar", "hbar, monkeypatch"])
def test_hbar_restoration_with_later_monkeypatch(pytester, arguments):
    """Fixture and test-body mutations share one undo stack, in either argument order."""
    fixture_path = Path(__file__).resolve().parents[1] / "conftest.py"
    pytester.makeconftest(f"""
import importlib.util
import strawberryfields as sf
spec = importlib.util.spec_from_file_location("unit_fixture_source", {str(fixture_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
hbar = module.hbar
sf.hbar = 1.7
""")
    pytester.makepyfile(f"""
import pytest
import strawberryfields as sf

@pytest.mark.parametrize("hbar", [0.5], indirect=True)
def test_mutation({arguments}):
    assert hbar == sf.hbar == 0.5
    monkeypatch.setattr(sf, "hbar", 4.0)

def test_after_mutation():
    assert sf.hbar == 1.7

@pytest.mark.parametrize("hbar", [0.5], indirect=True)
def test_failure({arguments}):
    monkeypatch.setattr(sf, "hbar", 4.0)
    raise RuntimeError("intentional fixture teardown probe")

def test_after_failure():
    assert sf.hbar == 1.7
""")
    result = pytester.runpytest_subprocess("-q", "-p", "no:randomly", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(["*intentional fixture teardown probe*"])


@pytest.mark.parametrize("hbar", [0.5, 1.0], indirect=True)
def test_requested_units_reach_global(hbar, request):
    assert hbar == sf.hbar == request.node.callspec.params["hbar"]


def test_report_distinguishes_setup_units_from_body_units(pytester):
    """Setup metadata must not claim to describe later states, even on failure."""
    fixture_path = Path(__file__).resolve().parents[1] / "conftest.py"
    pytester.makeconftest(fixture_path.read_text(encoding="utf-8"))
    pytester.makepyfile("""
import pytest
import strawberryfields as sf

def check_units(make_state, hbar, monkeypatch, request, completion):
    assert sf.hbar == hbar
    observed = []
    for value in (1.0, 2.0):
        monkeypatch.setattr(sf, "hbar", value)
        state = make_state()
        assert state.hbar == value
        observed.append(str(state.hbar))
    request.node.user_properties.append(("observed_state_units", ",".join(observed)))
    if completion == "restored":
        monkeypatch.setattr(sf, "hbar", hbar)
    elif completion == "failed":
        raise RuntimeError("intentional reporting probe")

@pytest.mark.parametrize("hbar", [0.5, 1.7], indirect=True)
@pytest.mark.parametrize("completion", ["changed", "restored", "failed"])
def test_engine_units(setup_eng, hbar, monkeypatch, request, completion):
    def make_state():
        engine, program = setup_eng(1)
        return engine.run(program).state
    check_units(make_state, hbar, monkeypatch, request, completion)

@pytest.mark.parametrize("hbar", [0.5, 1.7], indirect=True)
@pytest.mark.parametrize("completion", ["changed", "restored", "failed"])
def test_backend_units(setup_backend, hbar, monkeypatch, request, completion):
    check_units(lambda: setup_backend(1).state(), hbar, monkeypatch, request, completion)
""")
    report = pytester.path / "units.xml"
    result = pytester.runpytest_subprocess(
        "-q",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "-m",
        "gaussian and pure",
        f"--junitxml={report}",
    )
    result.assert_outcomes(passed=8, failed=4)
    cases = list(ET.parse(report).getroot().iter("testcase"))
    assert len(cases) == 12
    for case in cases:
        properties = {
            prop.get("name"): prop.get("value") for prop in case.findall("./properties/property")
        }
        assert "sf_hbar" not in properties
        assert properties["setup_hbar"] == properties["requested_hbar"]
        assert float(properties["setup_hbar"]) in (0.5, 1.7)
        assert properties["observed_state_units"] == "1.0,2.0"


@pytest.mark.parametrize("variance", [0.137202909, 0.274405818])
def test_normal_variance_accepts_correct_samples(assert_normal_variance, variance):
    samples = np.random.default_rng(12).normal(size=(200, 3)) * np.sqrt(variance)
    assert_normal_variance(samples, variance)


@pytest.mark.parametrize("bad_column", [0, 1, 2])
@pytest.mark.parametrize("corruption", ["constant", "wrong_scale", "nonfinite"])
def test_normal_variance_rejects_one_bad_batch(assert_normal_variance, bad_column, corruption):
    variance = 0.274405818
    samples = np.random.default_rng(12).normal(size=(200, 3)) * np.sqrt(variance)
    if corruption == "constant":
        samples[:, bad_column] = 0
    elif corruption == "wrong_scale":
        samples[:, bad_column] *= 4
    else:
        samples[0, bad_column] = np.nan
    with pytest.raises(AssertionError):
        assert_normal_variance(samples, variance)


def test_actual_homodyne_test_rejects_zero_sampler(monkeypatch, hbar, assert_normal_variance):
    """Run the real integration assertion with the previously undetected faulty sampler."""
    path = Path(__file__).resolve().parents[1] / "integration" / "test_hbar_integration.py"
    spec = importlib.util.spec_from_file_location("homodyne_unit_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def zero_sampler(*args, **kwargs):
        calls.append(1)
        return np.zeros((1, 1))

    monkeypatch.setattr(ops.MeasureHomodyne, "_apply", zero_sampler)
    with pytest.raises(AssertionError, match="outside Gaussian interval"):
        module.TestIntegration().test_squeeze_variance_frontend(
            lambda modes: (sf.Engine("gaussian"), sf.Program(modes)),
            hbar,
            None,
            assert_normal_variance,
        )
    assert len(calls) == module.N_MEAS


@pytest.mark.parametrize("shape", [(0,), (200, 0), (1, 2)])
def test_variance_oracle_requires_samples(assert_normal_variance, shape):
    with pytest.raises(AssertionError):
        assert_normal_variance(np.zeros(shape), 0.5)


@pytest.mark.parametrize("modes", [1, 2])
@pytest.mark.parametrize("corruption", ["coherence", "shape"])
def test_density_oracle_rejects_corruption(assert_fock_density, cutoff, modes, corruption):
    """Equal photon probabilities cannot conceal a wrong off-diagonal density entry."""
    ket = np.zeros((cutoff,) * modes, dtype=complex)
    ket.flat[:2] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    density = np.outer(ket, ket.conj()) if modes == 1 else np.einsum("ij,kl->ikjl", ket, ket.conj())
    if corruption == "shape":
        density = density[0]
    else:
        density.flat[1] = 0
    state = SimpleNamespace(reduced_dm=lambda *args, **kwargs: density)
    with pytest.raises(AssertionError):
        assert_fock_density(state, ket)


@pytest.mark.parametrize("backend", ["gaussian", "bosonic"])
@pytest.mark.parametrize("modes", [1, 2, 4])
@pytest.mark.parametrize("corruption", [None, "shape", "nonfinite", "normalization"])
def test_finite_density_trace(state_trace, cutoff, tol, monkeypatch, backend, modes, corruption):
    """Interleaved ket/bra axes must preserve trace and expose lost population."""
    state = sf.Engine(backend).run(sf.Program(modes)).state
    ket = np.arange(1, cutoff**modes + 1).reshape((cutoff,) * modes).astype(complex)
    ket *= np.exp(0.31j * ket.real)
    ket /= np.linalg.norm(ket)
    density = np.outer(ket, ket.conj()).reshape((cutoff,) * (2 * modes))
    interleaved = [axis for mode in range(modes) for axis in (mode, mode + modes)]
    density = density.transpose(interleaved)
    if corruption == "shape":
        density = density[0]
    elif corruption == "nonfinite":
        density.flat[0] = np.nan
    elif corruption == "normalization":
        density *= 0.9

    def reduced_dm(selected, cutoff):
        assert selected == list(range(modes))
        assert cutoff == ket.shape[0]
        return density

    monkeypatch.setattr(state, "reduced_dm", reduced_dm)
    if corruption is None:
        assert np.allclose(state_trace(state), 1, atol=tol, rtol=0)
    else:
        with pytest.raises(AssertionError):
            assert np.allclose(state_trace(state), 1, atol=tol, rtol=0)


@pytest.mark.parametrize("corruption", [None, "row", "column", "shape", "nonfinite"])
def test_bosonic_parity_roundoff_bound(assert_bosonic_even, monkeypatch, corruption):
    """Accept cancellation roundoff but reject a small nonzero odd component."""
    state = sf.Engine("bosonic").run(sf.Program(1)).state
    density = np.zeros((10, 10), dtype=complex)
    density[0, 0] = 1
    density[1, 1] = np.finfo(np.float64).eps
    if corruption == "row":
        density[1, 0] = 1e-6
    elif corruption == "column":
        density[0, 1] = 1e-6
    elif corruption == "shape":
        density = density[0]
    elif corruption == "nonfinite":
        density[0, 0] = np.nan

    def reduced_dm(selected, cutoff):
        assert selected == [0] and cutoff == 10
        return density

    monkeypatch.setattr(state, "reduced_dm", reduced_dm)
    if corruption is None:
        assert_bosonic_even(state)
    else:
        with pytest.raises(AssertionError):
            assert_bosonic_even(state)


@pytest.mark.parametrize("modes,vectorize", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("representation", ["choi", "liouville", "kraus"])
@pytest.mark.parametrize("corruption", [None, "duplicate_member", "missing_batch"])
def test_channel_assertions_check_every_batch(modes, vectorize, representation, corruption):
    """The real channel tests must reject duplicated values and broadcastable shapes."""
    path = Path(__file__).resolve().parents[1] / "integration" / "test_utils_integration.py"
    spec = importlib.util.spec_from_file_location("channel_assertions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cutoff = 3
    ket = np.zeros((2,) + (cutoff,) * modes, dtype=complex)
    ket.reshape(2, -1)[0, :2] = [np.sqrt(0.7), 1j * np.sqrt(0.3)]
    ket.reshape(2, -1)[1, :2] = [np.sqrt(0.2), -1j * np.sqrt(0.8)]
    density = (
        np.einsum("bi,bj->bij", ket, ket.conj())
        if modes == 1
        else np.einsum("bij,bkl->bikjl", ket, ket.conj())
    )
    reference = density.copy()
    if corruption == "duplicate_member":
        reference[1] = reference[0]
    elif corruption == "missing_batch":
        reference = reference[0]
    program = sf.Program(modes)
    with program.context as q:
        ops.LossChannel(1.0) | q[0]
    instance = (
        module.TestExtractChannelOneMode() if modes == 1 else module.TestExtractChannelTwoMode()
    )
    suffix = "_vectorize" if vectorize else ""
    check = getattr(instance, f"test_extract_{representation}_channel{suffix}")
    arguments = ((program, reference, density), cutoff, 1e-3)
    if corruption is None:
        check(*arguments)
    else:
        with pytest.raises(AssertionError):
            check(*arguments)
