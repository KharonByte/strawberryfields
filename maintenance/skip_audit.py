# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Opt-in pytest exclusion ledger; records observations, never classifies support.

Load with ``-p maintenance.skip_audit --skip-audit=PATH --audit-profile=NAME``.
JSONL ends with session_finish only on orderly shutdown. Missing termination is
incomplete evidence, including when collection or execution was interrupted.
"""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest


def describe(value):
    """Keep parameter identity without evaluating or expanding arbitrary objects."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, type):
        return {"type": value.__module__ + "." + value.__qualname__}
    # This is explicitly partial identity. Do not collapse distinct objects into
    # a claim of duplicate parametrization merely because their types agree.
    return {"unserialized_type": type(value).__module__ + "." + type(value).__qualname__}


def pytest_addoption(parser):
    group = parser.getgroup("skip-audit")
    group.addoption("--skip-audit", help="Write an observation-only JSONL exclusion ledger")
    group.addoption("--audit-profile", default="unspecified", help="Requested CI profile label")


def pytest_configure(config):
    path = config.getoption("skip_audit")
    if path:
        config.pluginmanager.register(Ledger(config, Path(path)), "skip-audit-ledger")


class Ledger:
    """Write before and after selection, and preserve all three runtime phases."""

    def __init__(self, config, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("w", encoding="utf-8")
        self.config = config
        self.collected = 0
        self.selected = 0
        self.reports = 0

    def emit(self, event, **fields):
        self.stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
        self.stream.flush()

    def pytest_sessionstart(self, session):
        root = str(session.config.rootpath)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
        )
        packages = {}
        for name in (
            "pytest",
            "numpy",
            "scipy",
            "thewalrus",
            "tensorflow",
            "tensorflow-cpu",
            "xanadu-cloud-client",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        self.emit(
            "session_start",
            schema=1,
            profile=self.config.getoption("audit_profile"),
            revision=revision.stdout.strip() if revision.returncode == 0 else None,
            tracked_and_untracked_changes=(
                dirty.stdout.splitlines() if dirty.returncode == 0 else None
            ),
            python=sys.version,
            platform=platform.platform(),
            packages=packages,
            arguments=list(self.config.invocation_params.args),
            requested_environment={
                key: os.environ.get(key)
                for key in (
                    "BATCHED",
                    "BATCHSIZE",
                    "HBAR",
                    "CUTOFF",
                    "TOL",
                    "ALPHA",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                    "PYTHONNOUSERSITE",
                    "SYMPY_USE_CACHE",
                    "CUDA_VISIBLE_DEVICES",
                )
            },
        )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        for item in items:
            self.collected += 1
            self.emit(
                "collected",
                nodeid=item.nodeid,
                location=list(item.location),
                parameters=describe(getattr(getattr(item, "callspec", None), "params", {})),
                fixtures=list(getattr(item, "fixturenames", [])),
                markers=[
                    {
                        "name": mark.name,
                        "args": describe(mark.args),
                        "kwargs": describe(mark.kwargs),
                    }
                    for mark in item.iter_markers()
                ],
            )
        yield
        self.selected = len(items)
        self.emit("selection", nodeids=[item.nodeid for item in items])

    def pytest_deselected(self, items):
        self.emit("deselected", nodeids=[item.nodeid for item in items])

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.emit(
                "collection_outcome",
                nodeid=report.nodeid,
                outcome=report.outcome,
                reason=str(report.longrepr),
            )

    def pytest_runtest_logreport(self, report):
        self.reports += 1
        self.emit(
            "runtime",
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            wasxfail=getattr(report, "wasxfail", None),
            reason=str(report.longrepr) if report.longrepr else None,
            properties=describe(report.user_properties),
        )

    def pytest_sessionfinish(self, session, exitstatus):
        self.emit(
            "session_finish",
            exitstatus=int(exitstatus),
            collected=self.collected,
            selected=self.selected,
            reports=self.reports,
            collectonly=bool(self.config.getoption("collectonly")),
        )

    def pytest_unconfigure(self):
        self.stream.close()
