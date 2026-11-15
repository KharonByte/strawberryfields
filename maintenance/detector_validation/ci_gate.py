"""Hosted dispatch admission and separate integrity/scientific comparison gate."""

import argparse
import os
from pathlib import Path
import re
import subprocess

from .ci import context, require, validate_release
from .fresh import load, save
from .provenance import digest, parse_json
from .ci_collect import summarize


def admit(path, sha):
    require(
        re.fullmatch(r"maintenance/detector_validation/releases/[A-Za-z0-9_-]+\.json", path),
        "release must be a committed policy file",
    )
    release = validate_release(path, sha)
    current = context(sha, "base")
    require(
        os.environ.get("GITHUB_REF") == "refs/heads/" + os.environ.get("DEFAULT_BRANCH", ""),
        "full collection requires owner-merged default branch workflow",
    )
    repository = os.environ["GITHUB_REPOSITORY"]
    # Serial release concurrency plus complete, bounded history inspection prevents a
    # second dispatch from quietly becoming a seed reroll. Unknown history fails closed.
    pages = 0
    seen = set()
    total = None
    for page in range(1, 11):
        raw = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/{repository}/actions/workflows/detector-study.yml/runs?per_page=100&page={page}",
            ],
            timeout=30,
        )
        result = parse_json(raw)
        pages += 1
        if total is None:
            total = result["total_count"]
        require(
            result["total_count"] == total,
            "history changed during admission; retain failed dispatch",
        )
        require(
            result["total_count"] <= 1000, "history exceeds admission bound; new policy required"
        )
        for run in result["workflow_runs"]:
            require(run["id"] not in seen, "duplicate history row")
            seen.add(run["id"])
            if run["display_title"] == "detector-" + sha:
                require(str(run["id"]) == current["run_id"], "release already dispatched; no retry")
        if len(result["workflow_runs"]) < 100:
            break
    require(len(seen) == total and int(current["run_id"]) in seen, "incomplete run history")
    return {
        "status": "admitted",
        "release_sha256": sha,
        "run_id": current["run_id"],
        "history_pages_checked": pages,
        "budget_scope": release["budget_scope"],
    }


def comparison(base, candidate, sha):
    for role, report in (("base", base), ("candidate", candidate)):
        for key, value in context(sha, role).items():
            require(report.get(key) == value, "comparison binding mismatch: " + key)
        require(
            report.get("integrity_status") == "PASS"
            and report.get("coverage_status") == "COMPLETE",
            "missing or invalid revision evidence",
        )
        require(
            report.get("scientific_verdict") in ("PASS", "FAIL", "INCOMPLETE"),
            "invalid scientific verdict",
        )
        computed = summarize(report.get("cases", []), report.get("errors", []))
        require(
            computed["integrity_status"] == "PASS"
            and computed["scientific_verdict"] == report["scientific_verdict"],
            "aggregate report is incomplete or inconsistent",
        )
    require(
        base.get("runtime_content") and base["runtime_content"] == candidate.get("runtime_content"),
        "base/candidate runtime content mismatch",
    )
    return {
        "release_sha256": sha,
        "run_id": base["run_id"],
        "integrity_status": "PASS",
        "base_verdict": base["scientific_verdict"],
        "candidate_verdict": candidate["scientific_verdict"],
        "scientific_verdict": (
            "PASS"
            if base["scientific_verdict"] == candidate["scientific_verdict"] == "PASS"
            else (
                "FAIL"
                if "FAIL" in (base["scientific_verdict"], candidate["scientific_verdict"])
                else "INCOMPLETE"
            )
        ),
        "confidence_scope": "99% within each revision; no 99% simultaneous two-revision claim",
        "criterion_changes": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("admit", "compare"))
    p.add_argument("--release")
    p.add_argument("--sha256", required=True)
    p.add_argument("--base")
    p.add_argument("--candidate")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        if args.command == "admit":
            result = admit(args.release, args.sha256)
        else:
            result = comparison(load(args.base), load(args.candidate), args.sha256)
            result["report_sha256"] = {
                "base": digest(args.base),
                "candidate": digest(args.candidate),
            }
    except Exception as error:
        result = {
            "status": "rejected",
            "integrity_status": "INCOMPLETE",
            "scientific_verdict": "INCOMPLETE",
            "release_sha256": args.sha256,
            "error": type(error).__name__ + ": " + str(error)[:2000],
        }
    save(Path(args.out), result)
    return 0 if result.get("scientific_verdict", "PASS") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
