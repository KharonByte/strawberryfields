# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""The remote client is optional, with actionable and accurate import errors."""

import builtins
from importlib import metadata

import pytest
import strawberryfields as sf

pytestmark = pytest.mark.frontend


@pytest.mark.parametrize("missing", ["xcc", "pydantic"])
def test_remote_engine_missing_module(monkeypatch, missing):
    """Only a missing client is translated into installation guidance."""
    original_import = builtins.__import__
    error = ModuleNotFoundError(f"No module named '{missing}'", name=missing)

    def import_module(name, *args, **kwargs):
        if name == "xcc":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ImportError) as caught:
        sf.RemoteEngine("X8")
    if missing == "xcc":
        assert ".[cloud]" in str(caught.value)
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error


def test_remote_engine_preserves_broken_client_error(monkeypatch):
    """An installed but incompatible client must not look like a missing extra."""
    original_import = builtins.__import__
    error = ImportError("incompatible cloud dependency")

    def import_module(name, *args, **kwargs):
        if name == "xcc":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(ImportError) as caught:
        sf.RemoteEngine("X8")
    assert caught.value is error


def test_about_without_cloud_distribution(monkeypatch, capfd):
    """The diagnostic report works when the optional distribution is absent."""
    original_version = metadata.version

    def version(name):
        if name == "xanadu-cloud-client":
            raise metadata.PackageNotFoundError(name)
        return original_version(name)

    monkeypatch.setattr(metadata, "version", version)
    sf.about()
    out, _ = capfd.readouterr()
    assert "XCC version:               Not installed (optional)" in out
