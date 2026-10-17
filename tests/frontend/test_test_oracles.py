# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Test infrastructure must reject incorrect units, statistics and coherences."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import strawberryfields as sf
from strawberryfields import ops

pytestmark = pytest.mark.frontend
pytest_plugins = ["pytester"]


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


@pytest.mark.parametrize("hbar", [0.5, 1.0], indirect=True)
def test_requested_units_reach_global(hbar, request):
    assert hbar == sf.hbar == request.node.callspec.params["hbar"]


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
