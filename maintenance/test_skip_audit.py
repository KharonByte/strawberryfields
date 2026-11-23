# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Negative evidence controls and an isolated real pytest collection/execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from maintenance.summarize_skip_audit import junit_identity, reconcile


def test_parameter_delimiters_are_not_class_separators():
    assert junit_identity("apps/test_a.py::TestA::test_a[Language :: Python :: 3.10]") == (
        "apps.test_a.TestA",
        "test_a[Language :: Python :: 3.10]",
    )


def test_collection_skips_are_not_collected_test_items():
    events, root = example()
    events.insert(
        1,
        {
            "event": "collection_outcome",
            "nodeid": "test_absent.py",
            "outcome": "skipped",
            "reason": "missing dependency",
        },
    )
    case = ET.SubElement(root[0], "testcase", classname="", name="test_absent")
    ET.SubElement(case, "skipped", message="collection skipped")
    result = reconcile(events, root)
    assert result["collection_skipped"] == 1 and result["selected"] == 1
    assert result["outcomes"] == {"passed": 1, "skipped": 1}
    case.remove(case.find("skipped"))
    with pytest.raises(ValueError, match="contradicts"):
        reconcile(events, root)


def example():
    events = [
        {"event": "session_start"},
        {"event": "collected", "nodeid": "test_sample.py::test_a"},
        {"event": "selection", "nodeids": ["test_sample.py::test_a"]},
        {
            "event": "session_finish",
            "exitstatus": 0,
            "collected": 1,
            "selected": 1,
            "reports": 0,
            "collectonly": True,
        },
    ]
    root = ET.fromstring(
        '<testsuites><testsuite><testcase classname="test_sample" '
        'name="test_a"/></testsuite></testsuites>'
    )
    return events, root


@pytest.mark.parametrize(
    "fault",
    [
        "truncated",
        "duplicate_collection",
        "duplicate_junit",
        "missing_junit",
        "bad_partition",
        "bad_counter",
        "unsuccessful",
        "failure",
        "no_runtime",
    ],
)
def test_rejects_incomplete_or_ambiguous_evidence(fault):
    events, root = example()
    if fault == "truncated":
        events.pop()
    elif fault == "duplicate_collection":
        events.insert(1, deepcopy(events[1]))
    elif fault == "duplicate_junit":
        root[0].append(deepcopy(root[0][0]))
    elif fault == "missing_junit":
        root[0].remove(root[0][0])
    elif fault == "bad_partition":
        events.insert(2, {"event": "deselected", "nodeids": ["test_sample.py::test_a"]})
    elif fault == "bad_counter":
        events[-1]["selected"] = 2
    elif fault == "unsuccessful":
        events[-1]["exitstatus"] = 2
    elif fault == "failure":
        ET.SubElement(root[0][0], "failure")
    elif fault == "no_runtime":
        events[-1]["collectonly"] = False
    with pytest.raises(ValueError):
        reconcile(events, root)


def test_preserves_skip_xfail_and_duplicate_properties():
    for kind, outcome in (("pytest.skip", "skipped"), ("pytest.xfail", "xfail")):
        events, root = example()
        ET.SubElement(root[0][0], "skipped", type=kind, message="contract needs review")
        props = ET.SubElement(root[0][0], "properties")
        ET.SubElement(props, "property", name="batch_size", value="2")
        ET.SubElement(props, "property", name="batch_size", value="None")
        result = reconcile(events, root)
        assert result["outcomes"] == {outcome: 1}
        assert result["exclusions"][0]["classification"] == "unreviewed"
        assert len(result["exclusions"][0]["reported_properties"]) == 2
        assert not result["runtime_evidence_in_ledger"]


def test_real_collection_and_runtime(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        """import pytest
@pytest.fixture
def unavailable():
    pytest.skip("fixture unavailable")
def test_setup_skip(unavailable): pass
def test_call_skip(): pytest.skip("call unavailable")
@pytest.mark.xfail(strict=True, reason="known failure")
def test_xfail(): assert False
def test_pass(): assert True
def test_deselected(): raise AssertionError("must not execute")
class Dormant:
    def test_uncollected(self): raise AssertionError("must not execute")
""",
        encoding="utf-8",
    )
    repo = str(Path(__file__).resolve().parents[1])
    ledger, junit = tmp_path / "ledger.jsonl", tmp_path / "tests.xml"
    env = {**os.environ, "PYTHONPATH": repo, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(sample),
            "-q",
            "-p",
            "maintenance.skip_audit",
            "--skip-audit=" + str(ledger),
            "--audit-profile=control",
            "--junitxml=" + str(junit),
            "-k",
            "not deselected",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    report = reconcile(events, ET.parse(junit).getroot())
    assert report["outcomes"] == {"skipped": 2, "xfail": 1, "passed": 1}
    assert report["collected"] == 5 and report["selected"] == 4 and report["deselected"] == 1
    assert report["runtime_evidence_in_ledger"]
    runtime = [event for event in events if event["event"] == "runtime"]
    assert not any(
        event["phase"] == "call" and "setup_skip" in event["nodeid"] for event in runtime
    )
    altered = deepcopy(events)
    next(
        event
        for event in altered
        if event["event"] == "runtime"
        and "call_skip" in event["nodeid"]
        and event["phase"] == "call"
    )["outcome"] = "passed"
    with pytest.raises(ValueError, match="contradicts"):
        reconcile(altered, ET.parse(junit).getroot())
