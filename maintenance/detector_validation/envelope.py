"""Prospective resource declarations; integrity checks do not confer authorization."""

from pathlib import Path
import hashlib
import json

from .budget import SETTINGS

SCHEMA = "detector-resource-envelope-v6"


def validate_envelope(release, explicit_path=None, explicit_sha256=None):
    if release.get("workflow_schema") != "detector-workflow-release-v6":
        raise ValueError("new v6 release required; historical freezes cannot authorize this runner")
    if (
        release.get("purpose") != "full_scientific_collection"
        or release.get("test_only_amendment") is not False
    ):
        raise ValueError("test-only amendments cannot authorize full collection")
    if release.get("collection_authorized") is not True:
        raise ValueError("new operational collection release required")
    record = release.get("resource_envelope", {})
    path = Path(record.get("path", ""))
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute nonsymlink resource envelope required")
    if not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("bounded regular resource envelope required")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != record.get("sha256"):
        raise ValueError("resource envelope digest mismatch")
    if explicit_path is not None and (
        Path(explicit_path).absolute() != path or explicit_sha256 != digest
    ):
        raise ValueError("explicit resource envelope does not match new release")
    value = json.loads(data)
    if value.get("schema") != SCHEMA or value.get("settings") != SETTINGS:
        raise ValueError("resource envelope settings mismatch")
    if (
        value.get("strict_cpu_guarantee") is not False
        or value.get("overshoot_policy") != "record_and_fail_no_retry"
    ):
        raise ValueError("honest nominal CPU and overshoot policy required")
    if value.get("study_ceiling") != {
        "cpu_seconds": 43200,
        "wall_seconds": 57600,
        "bytes": 4294967296,
    }:
        raise ValueError("unchanged study ceilings required")
    if value.get("prior_debits") != {
        k: release["cumulative_ledger"].get(k)
        for k in ("prior_cpu_debit_seconds", "prior_wall_debit_seconds")
    }:
        raise ValueError("envelope cumulative debit mismatch")
    if set(value.get("review_records", {})) != {"qa", "scientific", "security"}:
        raise ValueError("prospective independent resource reviews required")
    if value["review_records"] != release.get("reviewers"):
        raise ValueError("envelope review identities must match release reviewers")
    if len({r.get("task") for r in value["review_records"].values()}) != 3:
        raise ValueError("distinct resource reviewers required")
    for role, reviewed in value["review_records"].items():
        review = Path(reviewed["path"])
        if (
            not review.is_file()
            or review.stat().st_size > 1024**2
            or hashlib.sha256(review.read_bytes()).hexdigest() != reviewed["sha256"]
        ):
            raise ValueError("resource review digest mismatch: " + role)
    return value
