"""Single-writer bounded terminal detector collection; explicit release required.

Run with python -B -m maintenance.detector_validation.collect. This bounds stored
application evidence, not arbitrary filesystem behavior of hostile native code.
"""

import argparse
import ctypes
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import struct
import subprocess
import sys
import time
import ast

from .provenance import (
    ROOT,
    THREADS,
    validate_candidate_binding,
    digest,
    read_json,
    parse_json,
    source_identity,
    runtime_identity,
    sanitized_environment,
)
from .storage import StoreWriter, CAPACITY, RESERVE, encoded, sha, valid_name
from .stream import CAPS, HEADER
from .supervision import supervise, snapshot, delta, cleanup_reap
from .output_bounds import (
    COMMAND_BYTES,
    EXIT_BYTES,
    ENV_BYTES,
    LOG_BYTES,
    PARTIAL_BYTES,
    CollectionSnapshots,
    put_bounded,
)
from .budget import (
    ATTEMPT_CAPACITY,
    WORKERS_CPU,
    FIRST_CASE_CPU,
    FIRST_CASE_WALL,
    COLLECTION_WALL,
    validate_ledger,
    remaining_chunk_limits,
)

MEMORY = 14 * 1024**3
LOG_CAP = LOG_BYTES
CONTROLLER_CPU = 890
FRAME_NAMES = {
    1: "started.json",
    2: "samples.npy",
    3: "secondary.npy",
    4: "returned-invalid.npy",
    5: "metadata.json",
}


def child_limits(cpu):
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "parent-death signal setup")
    if os.getppid() != parent:
        os._exit(125)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # Defense in depth only; no worker is given an evidence file descriptor.
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


class FrameParser:
    """Validate before accepting payload; at most one bounded frame in memory."""

    def __init__(self, store, prefix):
        self.store, self.prefix = store, prefix
        self.buffer, self.pending, self.seen, self.final_metadata = bytearray(), None, set(), None
        self.total = 0

    def feed(self, block):
        if len(block) > 65536:
            raise ValueError("oversized pipe read")
        self.total += len(block)
        if self.total > sum(CAPS.values()) + 5 * HEADER.size:
            raise ValueError("evidence stream flood")
        self.buffer.extend(block)
        while True:
            if self.pending is None:
                if len(self.buffer) < HEADER.size:
                    return
                magic, kind, size, checksum = HEADER.unpack(self.buffer[: HEADER.size])
                del self.buffer[: HEADER.size]
                if (
                    magic != b"DTV1"
                    or kind not in CAPS
                    or kind in self.seen
                    or size > CAPS[kind]
                    or len(self.seen) >= 5
                    or (not self.seen and kind != 1)
                    or 5 in self.seen
                ):
                    raise ValueError("invalid evidence frame header/sequence/length")
                self.pending = kind, size, checksum
            kind, size, checksum = self.pending
            if len(self.buffer) < size:
                return
            payload = bytes(self.buffer[:size])
            del self.buffer[:size]
            if hashlib.sha256(payload).digest() != checksum:
                raise ValueError("evidence frame hash mismatch")
            category = "commands" if kind == 1 else "raw"
            self.store.put(f"{category}/{self.prefix}/{FRAME_NAMES[kind]}", payload)
            self.seen.add(kind)
            if kind == 5:
                self.final_metadata = parse_json(payload)
            self.pending = None

    def finish(self):
        if self.buffer or self.pending is not None or not {1, 5}.issubset(self.seen):
            raise ValueError("truncated or incomplete evidence stream")

    def preserve_partial(self):
        if self.buffer:
            put_bounded(
                self.store,
                f"commands/{self.prefix}/partial-frame.bin",
                bytes(self.buffer),
                PARTIAL_BYTES,
            )


def kill_group(proc, sig):
    if proc is not None:
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass


def run_worker(store, command, case_bytes, env, prefix, cpu_limit=120, wall_limit=180):
    """All logs/partials go to store; no subprocess-owned evidence files."""
    if type(cpu_limit) is not int or not 1 <= cpu_limit <= 120 or not 0 < wall_limit <= 240:
        raise ValueError("invalid worker resource request")
    start = time.monotonic()
    parent_cpu_start = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = FrameParser(store, prefix)
    selector, proc, read_fd, write_fd = selectors.DefaultSelector(), None, None, None
    logs = {"stdout": bytearray(), "stderr": bytearray()}
    reason, deadline, code = None, None, None
    supervision = None
    self_before = snapshot()
    try:
        read_fd, write_fd = os.pipe()
        child_env = dict(env, DETECTOR_EVIDENCE_FD=str(write_fd))
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(write_fd,),
            preexec_fn=functools.partial(child_limits, cpu_limit),
        )
        os.close(write_fd)
        write_fd = None
        if len(case_bytes) > 32768:
            raise ValueError("case stdin bound")
        input_remaining = memoryview(case_bytes)
        os.set_blocking(proc.stdin.fileno(), False)
        selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
        for fd, label in ((read_fd, "evidence"), (proc.stdout, "stdout"), (proc.stderr, "stderr")):
            selector.register(fd, selectors.EVENT_READ, label)
        # Move existing registrations to deadline-driven supervision with sole wait4 reaping.
        registrations = list(selector.get_map().values())
        selector.close()
        callbacks = {}
        io_counts = {"stdin": 0, "evidence": 0, "stdout": 0, "stderr": 0}

        def callback(fd, label):
            def consume(mask):
                nonlocal input_remaining
                if label == "stdin":
                    try:
                        if input_remaining:
                            size = os.write(fd, input_remaining[:4096])
                            io_counts[label] += size
                            input_remaining = input_remaining[size:]
                        if not input_remaining:
                            proc.stdin.close()
                            return False
                    except BrokenPipeError:
                        proc.stdin.close()
                        return "worker_stdin_closed"
                    except BlockingIOError:
                        pass
                    return True
                block = os.read(fd, 65536)
                io_counts[label] += len(block)
                if not block:
                    return False
                if label == "evidence":
                    parser.feed(block)
                else:
                    available = max(0, LOG_CAP - sum(map(len, logs.values())))
                    logs[label].extend(block[:available])
                    if len(block) > available:
                        return "worker_log_limit"
                return True

            return consume

        for key in registrations:
            fd = key.fd
            callbacks[fd] = (key.events, callback(fd, key.data))
        supervision = supervise(
            proc,
            callbacks,
            started=start,
            wall_ceiling=wall_limit,
            cleanup_grace=30,
            own_cpu_start=parent_cpu_start,
            own_cpu_tripwire=5,
        )
        supervision["io_bytes"] = io_counts
        reason, code = supervision["limit_failure"], supervision["exit_code"]
        if reason is None:
            try:
                parser.finish()
            except ValueError as exc:
                reason = "protocol:" + str(exc)
    except BaseException as exc:
        reason = "supervisor:" + type(exc).__name__ + ":" + str(exc)[:1024]
    finally:
        if proc is not None and proc.returncode is None:
            kill_group(proc, signal.SIGKILL)
        if proc is not None:
            if proc.returncode is not None:
                code = proc.returncode
            else:
                cleaned = cleanup_reap(proc, start + wall_limit)
                code = proc.returncode
                reason = reason or "supervisor_cleanup"
                if not cleaned["reaped"]:
                    reason += ":unreaped_accounting_missing"
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream and not stream.closed:
                    stream.close()
        selector.close()
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        # Store failures propagate; the last checkpoint remains an incomplete attempt.
        parser.preserve_partial()
        for label, data in logs.items():
            store.put(f"commands/{prefix}/{label}.txt", bytes(data))
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "requested_cpu_limit": cpu_limit,
        "strict_cpu_enforcement_verified": False,
        "supervision": supervision,
        "controller_post_cleanup": snapshot(),
        "controller_whole_chunk_delta": delta(self_before, snapshot()),
        "exit_code": code,
        "limit_failure": reason,
        "wall_seconds": time.monotonic() - start,
        "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "cpu_seconds_scope": "aggregate reaped children delta; do not add to tracked_child_wait4",
        "frames": sorted(parser.seen),
        "log_bytes": {k: len(v) for k, v in logs.items()},
        "log_received_bytes": {k: (supervision or {}).get("io_bytes", {}).get(k) for k in logs},
    }
    violations = []
    waited = (supervision or {}).get("tracked_child_wait4")
    if waited is None:
        violations.append("tracked_child_accounting_missing")
    elif waited["cpu_seconds"] > cpu_limit:
        violations.append("observed_worker_waited_cpu_exceeds_requested_ceiling")
    if result["wall_seconds"] > wall_limit:
        violations.append("observed_chunk_wall_exceeds_requested_ceiling")
    if result["controller_whole_chunk_delta"]["process_cpu_seconds"] > 5:
        violations.append("observed_controller_chunk_cpu_exceeds_tripwire")
    result["postflight_resource_violations"] = violations
    if violations and result["limit_failure"] is None:
        result["limit_failure"] = "postflight_resource_check"
    put_bounded(store, f"commands/{prefix}/exit.json", encoded(result), EXIT_BYTES)
    return result, parser.final_metadata


def check_metadata(meta, case, index, frames, requested_cpu):
    expected_frames = {1, 2, 5} | ({3} if case["path"] == "local_engine_gaussian" else set())
    if set(frames) != expected_frames or not isinstance(meta, dict):
        raise ValueError("required frame family incomplete or invalid")
    for field, value in {
        "status": "captured_not_accepted",
        "case": case,
        "case_id": case["id"],
        "chunk_index": index,
        "seed": case["chunk_seeds"][index],
        "rows": 10000,
        "output_mode_order": case["output_mode_order"],
        "evidence_type": "candidate_terminal_samples",
    }.items():
        if meta.get(field) != value:
            raise ValueError("worker metadata mismatch:" + field)
    caps = meta.get("verified_resource_limits", {})
    required_caps = {
        "address_space": MEMORY,
        "cpu_seconds": requested_cpu,
        "file_size": 0,
        "core_size": 0,
        "open_files": 64,
    }
    if set(caps) != set(required_caps):
        raise ValueError("missing/extra worker resource cap records")
    for name, maximum in required_caps.items():
        pair = caps[name]
        if set(pair) != {"soft", "hard"}:
            raise ValueError("invalid cap record")
        if pair["soft"] > pair["hard"]:
            raise ValueError("soft cap above hard cap")
        for value in pair.values():
            if type(value) is not int or (value != 0 if maximum == 0 else not 0 < value <= maximum):
                raise ValueError("worker cap exceeds request:" + name)
    policy = meta.get("cache_policy", {})
    if (
        policy.get("jit_disabled") is not False
        or policy.get("compiler_target_options_changed") is not False
        or policy.get("all_intercepted_caches_null") is not True
        or meta.get("python_write_or_spawn_access_granted") is not False
        or meta.get("guard_policy_validated") is not True
    ):
        raise ValueError("worker execution-policy receipt missing or invalid")


def validate_release(manifest_path, release_path, release_hash):
    release, manifest = read_json(release_path), read_json(manifest_path)
    from .envelope import validate_envelope

    validate_envelope(release)
    if digest(release_path) != release_hash or release.get("collection_authorized") is not True:
        raise ValueError("reviewed release required")
    if release.get("synthetic_fixture_only") or digest(manifest_path) != release["manifest_sha256"]:
        raise ValueError("release/manifest mismatch")
    if release.get("aggregate_disk_enforcement") != "fixed_container_application_bound_v1":
        raise ValueError("bounded application storage review required")
    if set(release["reviewers"]) != {"qa", "scientific", "security"}:
        raise ValueError("three review roles required")
    tasks = [r["task"] for r in release["reviewers"].values()]
    if len(set(tasks)) != 3:
        raise ValueError("distinct reviewers required")
    for record in release["reviewers"].values():
        if digest(record["path"]) != record["sha256"]:
            raise ValueError("review bytes changed")
    frozen = read_json(ROOT / "maintenance/detector_validation/cases.json")
    if manifest["cases"] != frozen["cases"] or manifest["statistics"] != frozen["statistics"]:
        raise ValueError("frozen case/statistics family changed")
    if not manifest.get("operational_acceptance_frozen"):
        raise ValueError("acceptance not frozen")
    validate_ledger(release.get("cumulative_ledger"), release_sha256=release_hash)
    if (
        Path(release_path).absolute()
        != Path(release["cumulative_ledger"]["freeze_directory"]["path"]) / "release.json"
    ):
        raise ValueError("release outside frozen input directory")
    if manifest.get("renewed_resource_settings") != release["cumulative_ledger"]["settings"]:
        raise ValueError("manifest renewed resource mismatch")
    return release, manifest


def all_child_cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def collect(args):
    release, manifest = validate_release(args.manifest, args.release, args.release_sha256)
    if Path(args.out).absolute().parent != Path(release["output_parent"]).absolute():
        raise ValueError("unreviewed output parent")
    source, runtime = source_identity(), runtime_identity()
    validate_candidate_binding(source, manifest)
    if source != release["source"] or runtime != release["runtime"]:
        raise ValueError("source/runtime release mismatch")
    store = StoreWriter(args.out, capacity=ATTEMPT_CAPACITY)
    controller_started_observation = snapshot()
    started, child_cpu = time.monotonic(), 0.0
    snapshots = CollectionSnapshots(store)
    receipt = {
        "schema": "detector-collection-v1",
        "evidence_type": "candidate_terminal_samples",
        "manifest_sha256": digest(args.manifest),
        "release_sha256": args.release_sha256,
        "source_pre": source,
        "runtime_pre": runtime,
        "status": "incomplete",
        "cases": [],
        "storage": {
            "kind": "fixed_container_application_bound_v1",
            "capacity": ATTEMPT_CAPACITY,
            "reserve": RESERVE,
            "native_filesystem_sandbox": False,
        },
        "cumulative_ledger": release["cumulative_ledger"],
        "resource_scope": {
            "controller_cpu_cap": CONTROLLER_CPU,
            "workers_cpu_cap": WORKERS_CPU,
            "all_controller_children_included": "worker plus local Git identity helpers",
            "case_cpu_cap": 900,
            "first_case_cpu_cap": FIRST_CASE_CPU,
            "first_case_wall_cap": FIRST_CASE_WALL,
            "wall_includes_postflight": True,
        },
    }
    env = sanitized_environment()
    try:
        store.put("manifest.json", Path(args.manifest).read_bytes())
        store.put("release.json", Path(args.release).read_bytes())
        put_bounded(store, "environment.json", encoded(env), ENV_BYTES)
        store.checkpoint({"status": "preflight_passed"})
        for case_number, case in enumerate(manifest["cases"]):
            case_cpu_limit = FIRST_CASE_CPU if case_number == 0 else 900
            case_wall_limit = FIRST_CASE_WALL if case_number == 0 else 1200
            case_start, case_cpu = time.monotonic(), 0.0
            case_parent_start = time.process_time()
            case_child_start = all_child_cpu()
            row = {"id": case["id"], "status": "incomplete", "chunks": []}
            receipt["cases"].append(row)
            case_bytes = encoded(case)
            if len(case_bytes) > 32768:
                raise ValueError("case byte bound")
            for index, seed in enumerate(case["chunk_seeds"]):
                if source_identity() != source:
                    raise ValueError("source drift before worker")
                cpu, wall = remaining_chunk_limits(
                    case_number,
                    (all_child_cpu() - case_child_start)
                    + (time.process_time() - case_parent_start),
                    time.monotonic() - case_start,
                    all_child_cpu(),
                    time.monotonic() - started,
                )
                prefix = f"{case['id']}/{index:02d}"
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "maintenance.detector_validation.worker",
                    "--case-sha256",
                    sha(case_bytes),
                    "--chunk",
                    str(index),
                ]
                put_bounded(
                    store, f"commands/{prefix}/command.json", encoded(command), COMMAND_BYTES
                )
                result, metadata = run_worker(store, command, case_bytes, env, prefix, cpu, wall)
                child_cpu += result["cpu_seconds"]
                case_cpu += result["cpu_seconds"]
                row["chunks"].append(dict(result, index=index, seed=seed, rows=10000))
                if result["exit_code"] or result["limit_failure"]:
                    raise ValueError("worker failure; no retry")
                check_metadata(metadata, case, index, result["frames"], cpu)
                store.checkpoint(
                    {
                        "status": "incomplete",
                        "completed_chunks": sum(len(c["chunks"]) for c in receipt["cases"]),
                    }
                )
            row["controller_cpu_seconds"] = time.process_time() - case_parent_start
            row["worker_cpu_seconds"] = case_cpu
            row["all_child_cpu_seconds"] = all_child_cpu() - case_child_start
            row["elapsed_wall_seconds"] = time.monotonic() - case_start
            row["cpu_cap_seconds"] = case_cpu_limit
            row["wall_cap_seconds"] = case_wall_limit
            if row["elapsed_wall_seconds"] > case_wall_limit:
                raise ValueError("case wall ceiling exceeded")
            if row["all_child_cpu_seconds"] + row["controller_cpu_seconds"] > case_cpu_limit:
                raise ValueError("case CPU ceiling exceeded")
            row["status"] = "complete"
            receipt["files"] = [
                {"path": name, **item}
                for name, item in store.inventory().items()
                if name != "collection.json"
            ]
            snapshots.put(encoded(receipt))
        receipt["source_post"], receipt["runtime_post"] = source_identity(), runtime_identity()
        if receipt["source_post"] != source or receipt["runtime_post"] != runtime:
            raise ValueError("source/runtime postflight drift")
        if all_child_cpu() > WORKERS_CPU:
            raise ValueError("postflight child CPU ceiling exceeded")
        validate_ledger(release["cumulative_ledger"], release_sha256=args.release_sha256)
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        receipt["controller_postflight_observation"] = snapshot()
        receipt["controller_scope_delta"] = delta(controller_started_observation, snapshot())
        receipt["controller_observation_excludes_final_receipt_write"] = True
        receipt["strict_cpu_enforcement_verified"] = False
        if time.process_time() > CONTROLLER_CPU or time.monotonic() - started > COLLECTION_WALL:
            receipt["status"] = "incomplete"
            receipt["failure"] = {
                "type": "ResourcePostflight",
                "message": "observed controller CPU/wall exceeded",
            }
        receipt["worker_cpu_seconds"] = child_cpu
        receipt["all_child_cpu_seconds"] = all_child_cpu()
        receipt["elapsed_wall_seconds"] = time.monotonic() - started
        receipt["files"] = [
            {"path": name, **item}
            for name, item in store.inventory().items()
            if name != "collection.json"
        ]
        try:
            snapshots.put(encoded(receipt))
            store.checkpoint({"status": receipt["status"], "failure": receipt.get("failure")})
        finally:
            store.close()
    if receipt["status"] != "complete":
        raise RuntimeError("collection incomplete; preserved bounded evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--out", required=True, help="New fixed evidence container; never reused")
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise ValueError("controller requires Python -B")
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (CONTROLLER_CPU, CONTROLLER_CPU))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # External workflow also hard-stops at16h; this earlier signal preserves failure receipts.
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("controller wall ceiling"))
    )
    signal.alarm(COLLECTION_WALL)
    collect(args)


if __name__ == "__main__":
    main()
