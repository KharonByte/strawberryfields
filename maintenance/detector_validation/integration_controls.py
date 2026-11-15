"""Full-pipeline synthetic fixtures; no real review, execution or collection authority.

The positive fixture checks reanalysis mechanics over all 68 cases. Its markers
are rejected by production analysis. Negative variants are in-memory overlays;
no candidate engine, sampler or external service is imported or invoked.
"""

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct
import time

from .reanalyse import (
    analyse_run,
    analyse_synthetic_fixture,
    exclusive_output,
    parse_npy,
    read_json,
    set_analysis_limits,
)
from .storage import StoreReader, StoreWriter, encoded


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    payload = encoded(value)
    if len(payload) > 8 * 1024**2:
        raise ValueError("fixture report byte bound")
    with Path(path).open("xb") as stream:
        stream.write(payload)


def npy_bytes(rows):
    columns = len(rows[0])
    header = repr({"descr": "<i8", "fortran_order": False, "shape": (len(rows), columns)}).encode(
        "ascii"
    )
    header += b" " * (128 - 10 - len(header) - 1) + b"\n"
    out = io.BytesIO(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    out.seek(0, 2)
    for row in rows:
        out.write(struct.pack("<" + "q" * columns, *row))
    return out.getvalue()


class Overlay:
    """Bounded test-only virtual edits without duplicating the fixture container."""

    def __init__(self, base, edits=None, removed=(), stale_digest=()):
        self.base, self.edits = base, edits or {}
        self.removed, self.stale_digest = set(removed), set(stale_digest)
        self.state = deepcopy(base.state)

    def inventory(self):
        result = self.base.inventory()
        for name in self.removed:
            result.pop(name, None)
        for name, data in self.edits.items():
            if name not in self.stale_digest:
                result[name] = {"bytes": len(data), "sha256": sha(data)}
        return result

    def read_bytes(self, name, max_size):
        if name in self.removed:
            raise ValueError("removed synthetic member")
        data = self.edits.get(name)
        if data is None:
            return self.base.read_bytes(name, max_size)
        if len(data) > max_size:
            raise ValueError("overlay member byte bound")
        return data


def build_fixture(manifest_path, controls_root, output):
    manifest = read_json(manifest_path)
    previous = read_json(Path(controls_root) / "input-manifest.json")
    if manifest["cases"] != previous["cases"] or manifest["statistics"] != previous["statistics"]:
        raise ValueError("synthetic source controls do not bind the current frozen mathematics")
    control_receipt = read_json(Path(controls_root) / "controls-receipt.json")
    if control_receipt.get("evidence_type") != "synthetic_checker_control":
        raise ValueError("fixture source must be explicitly synthetic controls")
    original_files = {row["path"]: row for row in control_receipt["files"]}
    manifest = deepcopy(manifest)
    manifest.update(operational_acceptance_frozen=True, synthetic_fixture_only=True)
    source = {"simulator_commit": manifest["candidate"], "synthetic_fixture_only": True}
    thread_names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    threads = {name: "1" for name in thread_names}
    runtime = {
        "synthetic_fixture_only": True,
        "actual_execution": False,
        "threads": threads,
        "executable": "synthetic-resolved-python",
        "invocation_executable": "synthetic-venv-python",
    }
    writer = StoreWriter(output, capacity=256 * 1024**2, reserve=16 * 1024**2)
    try:
        writer.put("manifest.json", encoded(manifest))
        writer.put(
            "environment.json",
            encoded(
                dict(
                    threads,
                    CUDA_VISIBLE_DEVICES="-1",
                    NUMBA_ENABLE_PROFILING="0",
                    NUMBA_DEBUGINFO="0",
                )
            ),
        )
        reviewers = {}
        for role in ("qa", "scientific", "security"):
            path = "fixture-reviews/" + role + ".json"
            review = encoded(
                {
                    "synthetic_fixture_only": True,
                    "role": role,
                    "verdict": "test_data_not_an_actual_independent_review",
                }
            )
            writer.put(path, review)
            reviewers[role] = {
                "task": "synthetic-fixture/" + role,
                "path": path,
                "sha256": sha(review),
            }
        release = {
            "synthetic_fixture_only": True,
            "collection_authorized": False,
            "manifest_sha256": sha(encoded(manifest)),
            "source": source,
            "runtime": runtime,
            "reviewers": reviewers,
            "aggregate_disk_enforcement": "synthetic256MiB_singlewriter_fixture_only",
        }
        writer.put("release.json", encoded(release))
        receipt = {
            "schema": "detector-collection-v1",
            "synthetic_fixture_only": True,
            "evidence_type": "synthetic_reanalysis_fixture",
            "status": "complete",
            "manifest_sha256": release["manifest_sha256"],
            "release_sha256": sha(encoded(release)),
            "source_pre": source,
            "source_post": source,
            "runtime_pre": runtime,
            "runtime_post": runtime,
            "cases": [],
        }
        for case_index, case in enumerate(manifest["cases"]):
            chunk_receipts = []
            for index, seed in enumerate(case["chunk_seeds"]):
                member = f"correct_{case_index:02d}/{index:02d}.npy"
                data = (Path(controls_root) / member).read_bytes()
                recorded = original_files[member]
                if len(data) != recorded["bytes"] or sha(data) != recorded["sha256"]:
                    raise ValueError("original synthetic raw digest mismatch")
                prefix = f"raw/{case['id']}/{index:02d}/"
                writer.put(prefix + "samples.npy", data)
                if case["path"] == "local_engine_gaussian":
                    rows = parse_npy(data, 10000, len(case["output_mode_order"]))
                    mapping = [
                        case["output_mode_order"].index(mode) for mode in case["measured_modes"]
                    ]
                    secondary = npy_bytes([tuple(row[j] for j in mapping) for row in rows])
                    writer.put(prefix + "secondary.npy", secondary)
                metadata = {
                    "synthetic_fixture_only": True,
                    "actual_execution": False,
                    "case_id": case["id"],
                    "chunk_index": index,
                    "seed": seed,
                    "rows": 10000,
                    "output_mode_order": case["output_mode_order"],
                    "requested_mode_order": case["measured_modes"],
                    "evidence_type": "synthetic_reanalysis_fixture",
                    "status": "captured_not_accepted",
                    "case": case,
                    "imports": [{"synthetic": True}],
                    "imports_after": [{"synthetic": True}],
                    "verified_resource_limits": {
                        "address_space": {"soft": 14 * 1024**3, "hard": 14 * 1024**3},
                        "cpu_seconds": {"soft": 120, "hard": 120},
                        "file_size": {"soft": 0, "hard": 0},
                        "core_size": {"soft": 0, "hard": 0},
                        "open_files": {"soft": 64, "hard": 64},
                    },
                    "rng": {"synthetic": True},
                    "guard_policy_validated": True,
                    "python_write_or_spawn_access_granted": False,
                    "cache_policy": {
                        "jit_disabled": False,
                        "compiler_target_options_changed": False,
                        "all_intercepted_caches_null": True,
                    },
                    "direct_call_sequence": [{"synthetic": True}],
                    "actual_pre_detector_reduced_moments": {"synthetic": True},
                    "original_circuit": [{"synthetic": True}],
                    "compiled_circuit": [{"synthetic": True}],
                }
                writer.put(prefix + "metadata.json", encoded(metadata))
                command_prefix = f"commands/{case['id']}/{index:02d}/"
                writer.put(
                    command_prefix + "command.json",
                    encoded(
                        [
                            runtime["invocation_executable"],
                            "-B",
                            "-m",
                            "maintenance.detector_validation.worker",
                            "--case-sha256",
                            sha(encoded(case)),
                            "--chunk",
                            str(index),
                        ]
                    ),
                )
                writer.put(
                    command_prefix + "exit.json",
                    encoded({"exit_code": 0, "limit_failure": None, "requested_cpu_limit": 120}),
                )
                writer.put(
                    command_prefix + "started.json", encoded({"synthetic_fixture_only": True})
                )
                writer.put(command_prefix + "stdout.txt", b"")
                writer.put(command_prefix + "stderr.txt", b"")
                chunk_receipts.append(
                    {
                        "index": index,
                        "seed": seed,
                        "rows": 10000,
                        "exit_code": 0,
                        "requested_cpu_limit": 120,
                        "limit_failure": None,
                        "synthetic_fixture_only": True,
                    }
                )
            receipt["cases"].append(
                {"id": case["id"], "status": "complete", "chunks": chunk_receipts}
            )
        receipt["files"] = [dict(path=name, **meta) for name, meta in writer.inventory().items()]
        writer.put("collection.json", encoded(receipt))
        writer.checkpoint({"status": "complete", "synthetic_fixture_only": True})
    finally:
        writer.close()
    return receipt["release_sha256"]


def root_overlay(reader, mutate):
    manifest = json.loads(reader.read_bytes("manifest.json", 4 * 1024**2))
    release = json.loads(reader.read_bytes("release.json", 4 * 1024**2))
    receipt = json.loads(reader.read_bytes("collection.json", 4 * 1024**2))
    mutate(manifest, release, receipt)
    release["manifest_sha256"] = sha(encoded(manifest))
    receipt["manifest_sha256"] = release["manifest_sha256"]
    receipt["release_sha256"] = sha(encoded(release))
    edits = {"manifest.json": encoded(manifest), "release.json": encoded(release)}
    for row in receipt["files"]:
        if row["path"] in edits:
            row.update(bytes=len(edits[row["path"]]), sha256=sha(edits[row["path"]]))
    edits["collection.json"] = encoded(receipt)
    return Overlay(reader, edits), receipt["release_sha256"]


def run(manifest_path, controls_root, output):
    root = exclusive_output(output)
    started = time.monotonic()
    store = root / "synthetic-evidence.dtv"
    release_digest = build_fixture(manifest_path, controls_root, store)
    reader = StoreReader(store)
    results = {}
    errors = []
    try:
        positive = analyse_synthetic_fixture(reader, release_digest)
        write_json(root / "full68-positive.json", positive)
        results["full68_synthetic"] = positive["probability_status"]
        if (
            positive["status"] != "synthetic_fixture_checked"
            or positive["probability_status"] != "accepted"
        ):
            errors.append("full68 synthetic fixture did not satisfy probability rules")

        def reject(name, view, digest, expected, production=False):
            try:
                if production:
                    analyse_run(
                        "manifest.json",
                        "collection.json",
                        trusted_release_sha256=digest,
                        evidence_reader=view,
                    )
                else:
                    analyse_synthetic_fixture(view, digest)
            except (ValueError, KeyError, OSError) as exc:
                reason = str(exc)
                results[name] = {"rejected": True, "reason": reason}
                if expected not in reason:
                    errors.append(name + ": rejected at unexpected gate")
            else:
                results[name] = {"rejected": False}
                errors.append(name + ": unexpectedly accepted")

        reject(
            "production_rejects_synthetic_release",
            reader,
            release_digest,
            "synthetic review fixture is not production",
            production=True,
        )
        scenarios = [
            ("missing_chunk", lambda m, a, r: r["cases"][0]["chunks"].pop(), "incomplete case"),
            (
                "unfrozen",
                lambda m, a, r: m.update(operational_acceptance_frozen=False),
                "not complete or frozen",
            ),
            (
                "wrong_revision",
                lambda m, a, r: m.update(candidate="0" * 40),
                "candidate identity mismatch",
            ),
            (
                "limit_failure",
                lambda m, a, r: r["cases"][0]["chunks"][0].update(limit_failure="wall"),
                "chunk execution mismatch",
            ),
            (
                "incomplete_collection",
                lambda m, a, r: r.update(status="incomplete"),
                "not complete or frozen",
            ),
            (
                "missing_review",
                lambda m, a, r: a["reviewers"].pop("qa"),
                "missing independent reviewer",
            ),
            (
                "false_fixture_authorization",
                lambda m, a, r: a.update(collection_authorized=True),
                "cannot authorize collection",
            ),
            (
                "runtime_drift",
                lambda m, a, r: r.update(runtime_post={"changed": True}),
                "identity changed",
            ),
        ]
        for name, mutation, expected in scenarios:
            view, trusted = root_overlay(reader, mutation)
            reject(name, view, trusted, expected)
        raw_name = next(n for n in reader.inventory() if n.endswith("samples.npy"))
        original = reader.read_bytes(raw_name, 200000)
        corrupt = original[:-1] + bytes([original[-1] ^ 1])
        reject(
            "tampered_payload",
            Overlay(reader, {raw_name: corrupt}, stale_digest=[raw_name]),
            release_digest,
            "reader member digest mismatch",
        )
        reject(
            "missing_raw",
            Overlay(reader, removed=[raw_name]),
            release_digest,
            "evidence inventory digest mismatch",
        )
        bad_state = Overlay(reader)
        bad_state.state = {"status": "incomplete"}
        reject("incomplete_checkpoint", bad_state, release_digest, "checkpoint is not complete")
        reject("untrusted_release_digest", reader, "0" * 64, "release byte identity mismatch")
        meta_name = next(n for n in reader.inventory() if n.endswith("metadata.json"))
        metadata = json.loads(reader.read_bytes(meta_name, 4 * 1024**2))
        metadata["status"] = "error"
        changed_meta = encoded(metadata)
        receipt = json.loads(reader.read_bytes("collection.json", 4 * 1024**2))
        for row in receipt["files"]:
            if row["path"] == meta_name:
                row.update(bytes=len(changed_meta), sha256=sha(changed_meta))
        view = Overlay(reader, {meta_name: changed_meta, "collection.json": encoded(receipt)})
        reject("metadata_failure", view, release_digest, "chunk metadata mismatch")
        original_metadata = json.loads(reader.read_bytes(meta_name, 4 * 1024**2))
        resource_mutants = [
            (
                "resource_missing",
                lambda m: m["verified_resource_limits"].pop("cpu_seconds"),
                "resource cap fields",
            ),
            (
                "resource_oversize_memory",
                lambda m: m["verified_resource_limits"]["address_space"].update(hard=15 * 1024**3),
                "excessive resource cap: address_space",
            ),
            (
                "resource_cpu_above_requested",
                lambda m: m["verified_resource_limits"]["cpu_seconds"].update(soft=121, hard=121),
                "excessive resource cap: cpu_seconds",
            ),
            (
                "resource_file_write_enabled",
                lambda m: m["verified_resource_limits"]["file_size"].update(hard=1),
                "excessive resource cap: file_size",
            ),
        ]
        for label, mutation, expected in resource_mutants:
            altered = deepcopy(original_metadata)
            mutation(altered)
            payload = encoded(altered)
            changed_receipt = json.loads(reader.read_bytes("collection.json", 4 * 1024**2))
            for item in changed_receipt["files"]:
                if item["path"] == meta_name:
                    item.update(bytes=len(payload), sha256=sha(payload))
            overlay = Overlay(
                reader, {meta_name: payload, "collection.json": encoded(changed_receipt)}
            )
            reject(label, overlay, release_digest, expected)
        view, trusted = root_overlay(
            reader, lambda m, a, r: r["cases"][0]["chunks"][0].update(requested_cpu_limit=60)
        )
        reject("resource_exit_cap_mismatch", view, trusted, "chunk exit resource binding mismatch")
    finally:
        reader.close()
    report = {
        "status": "supported" if not errors else "needs_changes",
        "results": results,
        "errors": errors,
        "seconds": time.monotonic() - started,
        "synthetic_fixture_only": True,
        "collection_authorized": False,
        "actual_independent_approvals": False,
        "candidate_imports_or_sampling": False,
        "fixed_fixture_bytes": store.stat().st_size,
        "source_sha256": {
            name: sha(Path(__file__).with_name(name).read_bytes())
            for name in ("reanalyse.py", "integration_controls.py", "storage.py")
        },
    }
    write_json(root / "integration-receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_analysis_limits()
    result = run(args.manifest, args.controls, args.out)
    print(json.dumps({k: result[k] for k in ("status", "errors", "seconds")}))
    return 0 if result["status"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
