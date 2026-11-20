"""Renewed-attempt limits; failed candidate evidence remains in the study ledger."""

import hashlib
from pathlib import Path

ATTEMPT_CAPACITY = 768 * 1024**2
STUDY_BYTES = 4 * 1024**3
FREEZE_BYTES = 16 * 1024**2
NEXT_OUTPUT_MAX = ATTEMPT_CAPACITY + 17 * 1024**2 + 1024
COLLECTION_CPU = 33000
COLLECTION_WALL = 48000
WORKERS_CPU = 32100
FIRST_CASE_CPU = 840
FIRST_CASE_WALL = 1140
EXPECTED_PRIOR = {
    "evidence.bin": {
        "bytes": 2147483648,
        "sha256": "1cfc845682e75ded5ad3c67ac69baf77116fc0df9a46617207272c63dc709bcd",
    },
    "run-receipt.json": {
        "bytes": 2730,
        "sha256": "bea44abe6ca394de46e2fb6dc0306c6c48827a08635c4437a74f1a0ee30e82a8",
    },
    "start.json": {
        "bytes": 275,
        "sha256": "70f0e7fb9f7159dc8711b9aca89693c7512e2deb7759ded2668ee57df99558e1",
    },
}
SETTINGS = {
    "attempt_capacity": ATTEMPT_CAPACITY,
    "cumulative_limit": STUDY_BYTES,
    "cpu_debit_seconds": 60,
    "wall_debit_seconds": 60,
    "collection_cpu_seconds": COLLECTION_CPU,
    "collection_wall_seconds": COLLECTION_WALL,
    "worker_cpu_seconds": WORKERS_CPU,
    "first_case_cpu_seconds": FIRST_CASE_CPU,
    "first_case_wall_seconds": FIRST_CASE_WALL,
    "reanalysis_cpu_seconds": 600,
    "reanalysis_wall_seconds": 900,
    "next_output_max_bytes": NEXT_OUTPUT_MAX,
}


def validate_ledger(ledger, *, verify_files=True, release_sha256=None):
    if not isinstance(ledger, dict) or ledger.get("settings") != SETTINGS:
        raise ValueError("renewed cumulative ledger settings mismatch")
    prior_cpu = ledger.get("prior_cpu_debit_seconds")
    prior_wall = ledger.get("prior_wall_debit_seconds")
    if (
        type(prior_cpu) not in (int, float)
        or type(prior_wall) not in (int, float)
        or not 8088.507136677 <= prior_cpu <= 43200 - COLLECTION_CPU - 600
        or not 8082.244731169005 <= prior_wall <= 57600 - COLLECTION_WALL - 900
    ):
        raise ValueError("explicit cumulative CPU/wall ledger exceeds study budget")
    records = ledger.get("retained_records")
    if not isinstance(records, list) or not 2 <= len(records) <= 512:
        raise ValueError("complete retained-record inventory required")
    attempts = ledger.get("prior_attempts")
    if not isinstance(attempts, list) or not {"candidate-attempt-1", "candidate-attempt-2"} <= set(
        attempts
    ):
        raise ValueError("both prior attempts must remain charged")
    required_retained = [
        {
            "bytes": 2147483648,
            "sha256": "1cfc845682e75ded5ad3c67ac69baf77116fc0df9a46617207272c63dc709bcd",
        },
        {
            "bytes": 2730,
            "sha256": "bea44abe6ca394de46e2fb6dc0306c6c48827a08635c4437a74f1a0ee30e82a8",
        },
        {
            "bytes": 275,
            "sha256": "70f0e7fb9f7159dc8711b9aca89693c7512e2deb7759ded2668ee57df99558e1",
        },
        {
            "bytes": 1073741824,
            "sha256": "e184d9e7636ae221060a2afe131f99bad1850c2173cf86613eeee21a876efec1",
        },
        {
            "bytes": 17002,
            "sha256": "50a8bfeded7d86cf757b58ff5d8490df51d69f10b4355f17e87e82d88b2923a9",
        },
        {
            "bytes": 275,
            "sha256": "733bc025c699f200129f4819b72cc871d34ae2888ef2f47b8bcd4f751d5b07ab",
        },
    ]
    for required in required_retained:
        if not any(all(row.get(k) == v for k, v in required.items()) for row in records):
            raise ValueError("original attempt container/receipt/start missing from ledger")
    paths, total = set(), 0
    for record in records:
        if (
            set(record) != {"path", "bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
        ):
            raise ValueError("invalid ledger record")
        path = Path(record["path"])
        if (
            not path.is_absolute()
            or str(path) in paths
            or any(p.is_symlink() for p in [path, *path.parents])
        ):
            raise ValueError("duplicate or unsafe ledger path")
        paths.add(str(path))
        total += record["bytes"]
        if total + NEXT_OUTPUT_MAX > STUDY_BYTES:
            raise ValueError("cumulative storage budget exhausted")
        if verify_files:
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("retained ledger byte mismatch")
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024**2), b""):
                    h.update(chunk)
            if h.hexdigest() != record["sha256"]:
                raise ValueError("retained ledger digest mismatch")
    freeze = ledger.get("freeze_directory", {})
    root = Path(freeze.get("path", ""))
    files = freeze.get("files")
    if (
        not root.is_absolute()
        or not isinstance(files, dict)
        or len(files) > 256
        or "release.json" in files
    ):
        raise ValueError("invalid frozen input directory ledger")
    if verify_files:
        if any(p.is_symlink() for p in [root, *root.parents]):
            raise ValueError("unsafe frozen directory path")
        actual = list(root.iterdir())
        if {p.name for p in actual} != set(files) | {"release.json"}:
            raise ValueError("frozen input inventory mismatch")
        frozen_bytes = 0
        for path in actual:
            if path.is_symlink() or not path.is_file():
                raise ValueError("nonregular frozen input")
            frozen_bytes += path.stat().st_size
            if frozen_bytes > FREEZE_BYTES:
                raise ValueError("frozen inputs exceed reserve")
            if path.name == "release.json" and release_sha256 is not None:
                if hashlib.sha256(path.read_bytes()).hexdigest() != release_sha256:
                    raise ValueError("frozen release digest mismatch")
            if path.name != "release.json":
                data = path.read_bytes()
                if files[path.name] != {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }:
                    raise ValueError("frozen input digest mismatch")
        if frozen_bytes > FREEZE_BYTES:
            raise ValueError("frozen inputs exceed reserve")
    if total + NEXT_OUTPUT_MAX + FREEZE_BYTES > STUDY_BYTES:
        raise ValueError("cumulative storage budget exhausted")
    return {
        "frozen_input_reserve_bytes": FREEZE_BYTES,
        "retained_bytes": total,
        "next_output_max_bytes": NEXT_OUTPUT_MAX,
        "cumulative_max_bytes": total + NEXT_OUTPUT_MAX + FREEZE_BYTES,
        "cumulative_limit": STUDY_BYTES,
    }


def remaining_chunk_limits(
    case_number, case_cpu_used, case_wall_used, all_child_cpu, collection_wall_used
):
    case_cpu_cap = FIRST_CASE_CPU if case_number == 0 else 900
    case_wall_cap = FIRST_CASE_WALL if case_number == 0 else 1200
    cpu = int(min(120, case_cpu_cap - case_cpu_used - 10, WORKERS_CPU - 30 - all_child_cpu))
    wall = min(
        180, case_wall_cap - 30 - case_wall_used, COLLECTION_WALL - 60 - collection_wall_used
    )
    if cpu < 1 or wall <= 0:
        raise ValueError("cumulative resource ceiling")
    return cpu, wall
