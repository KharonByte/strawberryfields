# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Explicit cutoff parameters must reach backend setup rather than silently using six."""

import pytest


@pytest.mark.parametrize("cutoff", [4, 12], indirect=True)
def test_requested_cutoff_is_used(cutoff, request):
    assert cutoff == request.node.callspec.params["cutoff"]
