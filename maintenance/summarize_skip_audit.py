# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0
"""Reconcile a collection ledger and JUnit without equating skips with support."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def junit_identity(nodeid):
    """Match pytest's JUnit classname/name convention, retaining parameter IDs."""
    address, bracket, parameters = nodeid.partition("[")
    path, *parts = address.split("::")
    path = path.removesuffix(".py").replace("/", ".")
    if not parts:
        return "", path
    return ".".join([path, *parts[:-1]]), parts[-1] + bracket + parameters


def reconcile(events, root):
    """Fail on missing/duplicate identities; preserve observed properties verbatim."""
    if not events or events[0].get("event") != "session_start":
        raise ValueError("Missing session_start")
    if events[-1].get("event") != "session_finish":
        raise ValueError("Incomplete ledger: missing session_finish")
    finish = events[-1]
    if finish["exitstatus"] != 0:
        raise ValueError("Unsuccessful collection/execution")
    collected = [event for event in events if event["event"] == "collected"]
    ids = [event["nodeid"] for event in collected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate collected node identity")
    selections = [event for event in events if event["event"] == "selection"]
    if len(selections) != 1:
        raise ValueError("Expected exactly one selection receipt")
    selected = selections[0]["nodeids"]
    deselected = [
        nodeid for event in events if event["event"] == "deselected" for nodeid in event["nodeids"]
    ]
    if Counter(selected + deselected) != Counter(ids):
        raise ValueError("Selection and deselection do not partition collection")
    if finish["collected"] != len(ids) or finish["selected"] != len(selected):
        raise ValueError("Incorrect collection/selection counters")
    collection_outcomes = [event for event in events if event["event"] == "collection_outcome"]
    if any(event["outcome"] == "failed" for event in collection_outcomes):
        raise ValueError("Collection failed")
    by_id = {junit_identity(event["nodeid"]): event for event in collected}
    if len(by_id) != len(collected):
        raise ValueError("Ambiguous JUnit identity projection")
    collection_skips = {
        junit_identity(event["nodeid"]): event
        for event in collection_outcomes
        if event["outcome"] == "skipped"
    }
    if len(collection_skips) != len(collection_outcomes) or set(collection_skips) & set(by_id):
        raise ValueError("Ambiguous collection outcome identity")
    by_id.update(collection_skips)
    cases = root.findall(".//testcase")
    case_ids = [(case.get("classname"), case.get("name")) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate JUnit identity")
    if set(case_ids) != {junit_identity(nodeid) for nodeid in selected} | set(collection_skips):
        raise ValueError("JUnit identities differ from selection (including collection-only skips)")
    counts = Counter()
    exclusions = []
    for case, identity in zip(cases, case_ids):
        skipped = case.find("skipped")
        if identity in collection_skips and skipped is None:
            raise ValueError("Collection skip contradicts JUnit")
        outcome = "passed"
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError("JUnit contains failures or errors")
        if skipped is not None:
            outcome = "xfail" if skipped.get("type") == "pytest.xfail" else "skipped"
            exclusions.append(
                {
                    **by_id[identity],
                    "outcome": outcome,
                    "classification": "unreviewed",
                    "reason": skipped.get("message"),
                    "source": skipped.text,
                    "reported_properties": [
                        dict(prop.attrib) for prop in case.findall("./properties/property")
                    ],
                }
            )
        counts[outcome] += 1
    runtime = [event for event in events if event["event"] == "runtime"]
    if finish["reports"] != len(runtime):
        raise ValueError("Incorrect runtime report counter")
    if not finish["collectonly"]:
        reports_by_node = defaultdict(list)
        for event in runtime:
            reports_by_node[event["nodeid"]].append(event)
        cases_by_id = dict(zip(case_ids, cases))
        phases = Counter((event["nodeid"], event["phase"]) for event in runtime)
        if any(count != 1 for count in phases.values()):
            raise ValueError("Duplicate runtime phase")
        if {event["nodeid"] for event in runtime} != set(selected):
            raise ValueError("Missing or unexpected runtime identities")
        for nodeid in selected:
            if phases[(nodeid, "setup")] != 1 or phases[(nodeid, "teardown")] != 1:
                raise ValueError("Missing setup/teardown evidence")
            reports = reports_by_node[nodeid]
            if any(event["outcome"] == "failed" for event in reports):
                raise ValueError("Runtime failure contradicts passing JUnit")
            observed = "passed"
            for event in reports:
                if event["wasxfail"] is not None and event["outcome"] == "passed":
                    raise ValueError("Unexpected pass needs explicit assessment")
                if event["outcome"] == "skipped":
                    observed = "xfail" if event["wasxfail"] is not None else "skipped"
            if observed == "passed" and phases[(nodeid, "call")] != 1:
                raise ValueError("Missing call evidence")
            case = cases_by_id[junit_identity(nodeid)]
            skip = case.find("skipped")
            expected = (
                "passed"
                if skip is None
                else "xfail" if skip.get("type") == "pytest.xfail" else "skipped"
            )
            if observed != expected:
                raise ValueError("Runtime outcome contradicts JUnit")
    return {
        "schema": 1,
        "collection_session": events[0],
        "runtime_evidence_in_ledger": not finish["collectonly"],
        "collected": len(ids),
        "selected": len(selected),
        "deselected": len(deselected),
        "collection_outcomes": collection_outcomes,
        "collection_skipped": len(collection_skips),
        "outcomes": dict(counts),
        "exclusions": exclusions,
        "limits": [
            "Reasons and markers are observations, not reviewed support classifications.",
            "Requested profiles do not establish actual per-test batching or storage.",
            "Unserialized parameters require source inspection; semantic duplicates remain open.",
            "Absent dependency parameters and dormant definitions require a source inventory.",
            "Passing outcomes do not prove independent assertions or every batch member.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("junit", type=Path)
    parser.add_argument(
        "--report-revision", required=True, help="Externally verified report revision"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines()]
    result = reconcile(events, ET.parse(args.junit).getroot())
    result["evidence"] = {
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "junit_sha256": hashlib.sha256(args.junit.read_bytes()).hexdigest(),
        "externally_verified_report_revision": args.report_revision,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("collected", "selected", "deselected", "outcomes")}
        )
    )


if __name__ == "__main__":
    main()
