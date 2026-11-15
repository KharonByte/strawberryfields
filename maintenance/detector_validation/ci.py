"""Prospective CI release policy and fixed shards. Never grants review approval."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .fresh import load, save, raw_bytes
from .fresh_profile import profile
from .provenance import ROOT, digest
from .storage import encoded

SHARDS = 17
LIMITS = {
    "cpu_seconds": 2500,
    "wall_seconds": 3300,
    "worker_cpu_seconds": 120,
    "worker_wall_seconds": 180,
    "capacity_bytes": 40 * 1024**2,
    "reserve_bytes": 2 * 1024**2,
    "retention_days": 30,
    "max_parallel": 4,
}
ROLES = ("qa", "scientific", "security")


def hash_value(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, timeout=30).strip()


def suite_files(root=ROOT):
    paths = list((root / "maintenance/detector_validation").rglob("*.py"))
    paths += [root / "maintenance/detector_validation/cases.json"]
    paths += list((root / "maintenance").glob("requirements-*.txt"))
    paths += list((root / ".github/workflows").glob("detector*.yml"))
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def cases_for(shard):
    require(type(shard) is int and 0 <= shard < SHARDS, "invalid shard index")
    return profile("scientific")["cases"][shard * 4 : (shard + 1) * 4]


def binding(release):
    return hash_value({k: v for k, v in release.items() if k != "reviews"})


def draft(base, candidate):
    for ref in (base, candidate):
        require(re.fullmatch(r"[0-9a-f]{40}", ref), "full immutable revision required")
        require(git("rev-parse", ref + "^{commit}") == ref, "revision unavailable")
    value = {
        "schema": "detector-ci-release-v1",
        "base": base,
        "candidate": candidate,
        "suite_files": suite_files(),
        "profile_sha256": hash_value(profile("scientific")),
        "shards": [[c["id"] for c in cases_for(i)] for i in range(SHARDS)],
        "limits": LIMITS,
        "runtime_policy": "exact maintenance pins; Python 3.12.3 Linux x86_64; identical content fingerprints across shards",
        "budget_scope": "REQUIRED: new prospective CI budget identity and retained prior evidence reference",
        "archive_plan": "REQUIRED: controlled durable destination and responsible custodian before 30-day expiry",
        "execution_policy": "one dispatch per approved release; run_attempt=1; no retry, resume, pooling or seed changes",
        "reviews": {},
    }
    for role in ROLES:
        value["reviews"][role] = {
            "reviewer": "",
            "decision": "pending",
            "binding_sha256": binding(value),
            "evidence": "",
        }
    return value


def validate_release(path, expected):
    require(re.fullmatch(r"[0-9a-f]{64}", expected or ""), "release SHA256 required")
    require(digest(path) == expected, "release checksum mismatch")
    value = load(path)
    require(value.get("schema") == "detector-ci-release-v1", "wrong CI release schema")
    for key in ("base", "candidate"):
        require(re.fullmatch(r"[0-9a-f]{40}", value.get(key, "")), "immutable refs required")
    require(value.get("suite_files") == suite_files(), "suite content drift")
    require(value.get("profile_sha256") == hash_value(profile("scientific")), "profile drift")
    require(
        value.get("shards") == [[c["id"] for c in cases_for(i)] for i in range(SHARDS)],
        "shard ownership drift",
    )
    require(value.get("limits") == LIMITS, "resource policy drift")
    for key in ("budget_scope", "archive_plan"):
        require(
            isinstance(value.get(key), str)
            and value[key].strip()
            and not value[key].startswith("REQUIRED:"),
            "unreviewed " + key,
        )
    reviews = value.get("reviews", {})
    require(set(reviews) == set(ROLES), "three independent reviews required")
    identities = []
    for role in ROLES:
        review = reviews[role]
        require(review.get("decision") == "approved", "pending review: " + role)
        require(review.get("binding_sha256") == binding(value), "stale review: " + role)
        for key in ("reviewer", "evidence"):
            require(
                isinstance(review.get(key), str) and review[key].strip(),
                "review identity/evidence required",
            )
        identities.append(review["reviewer"].strip())
    require(len(set(identities)) == 3, "distinct independent reviewers required")
    return value


def context(release_sha, role, shard=None):
    run = os.environ.get("GITHUB_RUN_ID", "")
    require(run.isdigit() and int(run) > 0, "hosted immutable run ID required")
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "reruns forbidden; preserve prior attempt")
    require(role in ("base", "candidate"), "unknown revision role")
    return {
        "release_sha256": release_sha,
        "run_id": run,
        "run_attempt": 1,
        "role": role,
        "shard": shard,
    }


def prepare(release, role, target):
    """Overlay only the reviewed harness and requirements onto an immutable simulator ref."""
    require(role in ("base", "candidate"), "unknown revision role")
    target = Path(target).absolute()
    require(not target.exists(), "target already exists")
    git("worktree", "add", "--detach", str(target), release[role])
    # The working simulator tree stays at the named revision. No cherry-picks or ref mutations.
    for name in release["suite_files"]:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    require(suite_files(target) == release["suite_files"], "overlay mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("draft")
    p.add_argument("--base", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--release", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--role", choices=("base", "candidate"), required=True)
    p.add_argument("--target", required=True)
    args = parser.parse_args()
    if args.command == "draft":
        save(Path(args.out), draft(args.base, args.candidate))
    else:
        release = validate_release(args.release, args.sha256)
        context(args.sha256, args.role)
        prepare(release, args.role, args.target)


if __name__ == "__main__":
    main()
