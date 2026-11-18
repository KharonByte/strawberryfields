"""One-command bounded collector/reanalysis workflow; never imports a candidate engine.

The launcher owns only a fresh attempt directory, an immutable start receipt and
an exclusive final receipt. Controller/worker evidence remains in one fixed store.
"""

import argparse
import base64
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time

from .storage import StoreReader, encoded
from .supervision import supervise, snapshot, delta, resource_violations, cleanup_reap
from .provenance import sanitized_environment, parse_json
from .budget import (
    ATTEMPT_CAPACITY as CAPACITY,
    COLLECTION_CPU,
    COLLECTION_WALL,
    WORKERS_CPU,
    validate_ledger,
)

ROOT = Path(__file__).resolve().parents[2]
MEMORY = 14 * 1024**3
LOG_BYTES = 512 * 1024
REPORT_BYTES = 16 * 1024**2
RECEIPT_BYTES = 1024**2
START_BYTES = 1024
TOTAL_BYTES = CAPACITY + REPORT_BYTES + RECEIPT_BYTES + START_BYTES
COLLECT_MODULE = "maintenance.detector_validation.collect"
REANALYSE_MODULE = "maintenance.detector_validation.reanalyse"
THREADS = {
    name: "1"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}


class WorkflowInterrupted(BaseException):
    """Not InterruptedError: selectors may swallow that exception as an EINTR retry."""


def safe_path(path):
    result = Path(path).absolute()
    for entry in [result, *result.parents]:
        if entry.is_symlink():
            raise ValueError("symlink path forbidden")
    return result


def exclusive_json(path, value, maximum):
    payload = encoded(value)
    if len(payload) > maximum:
        raise ValueError("receipt exceeds fixed byte bound")
    fd = os.open(safe_path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def bounded_json(path, maximum):
    fd = os.open(safe_path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("not a bounded regular JSON file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise ValueError("JSON byte bound")
        return parse_json(data), hashlib.sha256(data).hexdigest()
    finally:
        os.close(fd)


def parent_death(signum, *, expected_parent=None):
    """Install a process-local Linux death signal and check the parent identity.

    Spawned children receive the parent's pre-fork PID. At launcher entry no
    pre-launch identity is available, so that caller can only detect PID 1 or
    a parent change during setup, not earlier adoption by a non-PID-1 subreaper.
    """
    before = os.getppid()
    if before == 1 or (expected_parent is not None and before != expected_parent):
        raise RuntimeError("parent vanished before death-signal setup")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signum, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != before or before == 1:
        raise RuntimeError("parent vanished before death-signal setup")


def child_limits(cpu, file_size, *, expected_parent):
    parent_death(signal.SIGKILL, expected_parent=expected_parent)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def launcher_limits():
    # Best effort at entry; unlike spawn(), no pre-fork parent identity is available.
    parent_death(signal.SIGTERM)
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    # Children cannot raise a hard limit inherited as10. During bootstrap only,
    # default SIGXCPU at soft10 terminates this process; hard890 permits spawning
    # the two separately bounded children. Harden to10 immediately after both forks.
    signal.signal(signal.SIGXCPU, signal.SIG_DFL)
    resource.setrlimit(resource.RLIMIT_CPU, (10, 890))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAPACITY, CAPACITY))


def terminate_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def spawn(command, env, cpu, file_size, gated=False):
    if gated:
        gate_code = (
            "import os,sys; token=os.read(0,1); "
            "sys.exit(125) if token!=b'G' else None; "
            "os.execv(sys.executable,[sys.executable]+sys.argv[1:])"
        )
        executed = [sys.executable, "-B", "-c", gate_code, *command[1:]]
    else:
        executed = command
    expected_parent = os.getpid()
    return subprocess.Popen(
        executed,
        cwd=ROOT,
        env=env,
        stdin=subprocess.PIPE if gated else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
        start_new_session=True,
        preexec_fn=lambda: child_limits(cpu, file_size, expected_parent=expected_parent),
    )


def phase(process, command, cpu, soft_wall, grace, log, label, started):
    """Bounded pipe draining and external whole-controller lifetime, including postflight."""

    def consume(mask):
        data = os.read(process.stdout.fileno(), 16384)
        if not data:
            return False
        remaining = LOG_BYTES - len(log)
        log.extend(data[: max(0, remaining)])
        return "combined_output_limit" if len(data) > remaining else True

    observed = supervise(
        process,
        {process.stdout.fileno(): (selectors.EVENT_READ, consume)},
        started=started,
        wall_ceiling=soft_wall + grace,
        cleanup_grace=grace,
        own_cpu_start=0,
        own_cpu_tripwire=8,
        tripwire_reason="launcher_cpu_cleanup_reserve",
    )
    process.stdout.close()
    usage = observed["tracked_child_wait4"]
    violations = []
    own = (usage or {}).get("own_process_stat_before_reap", {})
    if "own_cpu_seconds_tick_resolution" not in own:
        violations.append("child_own_cpu_observation_missing")
    elif own["own_cpu_seconds_tick_resolution"] > cpu:
        violations.append("observed_child_own_cpu_exceeds_requested_ceiling")
    if observed["observed_since_launch_wall_seconds"] > soft_wall + grace:
        violations.append("observed_phase_wall_exceeds_ceiling")
    if violations and observed["limit_failure"] is None:
        observed["limit_failure"] = "postflight_resource_check"
    return {
        "phase": label,
        "exit_code": observed["exit_code"],
        "stop_reason": observed["limit_failure"],
        "wall_seconds": observed["observed_since_launch_wall_seconds"],
        "child_cpu_seconds": usage["cpu_seconds"] if usage else None,
        "requested_cpu_rlimit_seconds": cpu,
        "strict_cpu_enforcement_verified": False,
        "soft_wall_seconds": soft_wall,
        "kill_grace_seconds": grace,
        "command": command,
        "supervision": observed,
        "postflight_resource_violations": violations,
    }


def collect_complete(path, expected_capacity, synthetic):
    if safe_path(path).stat().st_size != expected_capacity:
        raise ValueError("evidence store size differs from fixed capacity")
    reader = StoreReader(path)
    try:
        if reader.state.get("status") != "complete":
            raise ValueError("collection checkpoint is not complete")
        receipt = json.loads(reader.read_bytes("collection.json", 4 * 1024**2))
        expected = "synthetic_workflow_fixture" if synthetic else "candidate_terminal_samples"
        if receipt.get("status") != "complete" or receipt.get("evidence_type") != expected:
            raise ValueError("collection receipt is not complete")
        if not synthetic and receipt.get("synthetic_fixture_only"):
            raise ValueError("synthetic collection cannot start production reanalysis")
        return {
            "status": receipt["status"],
            "capacity": reader.capacity,
            "members": len(reader.inventory()),
            "synthetic_fixture_only": synthetic,
        }
    finally:
        reader.close()


def record_final_children(receipt):
    """RUSAGE_CHILDREN accounts only reaped children; never imply missing cleanup."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt["final_all_reaped_children_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    complete = "unreaped_child_pid" not in receipt
    receipt["final_children_cpu_includes_gated_stub_cleanup"] = complete
    if not complete:
        receipt["status"] = "incomplete"
        receipt["cleanup_accounting_incomplete"] = True


def workflow(args, *, _test_modules=None, _test_limits=None):
    """Internal synthetic module injection is never exposed by the production CLI."""
    synthetic = _test_modules is not None
    if not synthetic:
        from .envelope import validate_envelope

        release_input, _ = bounded_json(args.release, 4 * 1024**2)
        validate_envelope(release_input, args.resource_envelope, args.resource_envelope_sha256)
    launcher_limits()
    launcher_start_snapshot = snapshot()
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(
            signum, lambda *_: (_ for _ in ()).throw(WorkflowInterrupted("launcher interrupted"))
        )
    attempt = safe_path(args.out)
    attempt.mkdir(mode=0o700, parents=False, exist_ok=False)
    start_wall = time.monotonic()
    start = {
        "schema": "detector-workflow-start-v1",
        "status": "started_completion_unknown",
        "synthetic_test_only": synthetic,
        "parent_pid": os.getppid(),
        "pid": os.getpid(),
        "collection_total_cpu_cap": COLLECTION_CPU,
        "reanalysis_cpu_cap_separate": 600,
        "output_bytes_max": TOTAL_BYTES,
        "actual_collection_approval_claim": False,
    }
    exclusive_json(attempt / "start.json", start, START_BYTES)
    receipt = {
        "schema": "detector-workflow-receipt-v1",
        "status": "incomplete",
        "synthetic_test_only": synthetic,
        "phases": [],
        "start": "start.json",
        "resource_scope": {
            "launcher_cpu": 10,
            "controller_cpu": 890,
            "workers_cpu_total": WORKERS_CPU,
            "collection_cpu_total": COLLECTION_CPU,
            "collection_wall_including_postflight": COLLECTION_WALL,
            "reanalysis_cpu_separate": 600,
            "reanalysis_wall_separate": 900,
            "combined_pipe_bytes": LOG_BYTES,
            "aggregate_output_bytes_max": TOTAL_BYTES,
            "existing_qualification_evidence_included": False,
        },
    }
    log = bytearray()
    children = []
    try:
        modules = _test_modules or (COLLECT_MODULE, REANALYSE_MODULE)
        limits = _test_limits or {
            "capacity": CAPACITY,
            "collect_soft": COLLECTION_WALL - 30,
            "analysis_soft": 870,
            "grace": 30,
        }
        env = sanitized_environment()
        if synthetic:
            env["PYTHONPATH"] = str(args.fixture_modules) + os.pathsep + str(ROOT)
            env["DETECTOR_SYNTHETIC_WORKFLOW_TEST"] = "1"
        if not synthetic:
            release, release_sha = bounded_json(args.release, 4 * 1024**2)
            if release_sha != args.release_sha256:
                raise ValueError("release digest mismatch")
            if (
                safe_path(args.release)
                != safe_path(release["cumulative_ledger"]["freeze_directory"]["path"])
                / "release.json"
            ):
                raise ValueError("release outside frozen input directory")
            receipt["cumulative_ledger"] = release.get("cumulative_ledger")
            receipt["cumulative_budget"] = validate_ledger(
                receipt["cumulative_ledger"], release_sha256=args.release_sha256
            )
        evidence = attempt / "evidence.bin"
        collect_command = [
            sys.executable,
            "-B",
            "-m",
            modules[0],
            "--manifest",
            str(safe_path(args.manifest)),
            "--release",
            str(safe_path(args.release)),
            "--release-sha256",
            args.release_sha256,
            "--out",
            str(evidence),
        ]
        analysis_command = [
            sys.executable,
            "-B",
            "-m",
            modules[1],
            "--container",
            str(evidence),
            "--release-sha256",
            args.release_sha256,
            "--out",
            str(attempt / "report"),
        ]
        receipt["bootstrap_cpu_limits"] = list(resource.getrlimit(resource.RLIMIT_CPU))
        analyser = spawn(analysis_command, env, 600, REPORT_BYTES, gated=True)
        children.append(analyser)
        controller = spawn(collect_command, env, 890, limits["capacity"])
        children.append(controller)
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        receipt["launcher_hardened_cpu_limits"] = list(resource.getrlimit(resource.RLIMIT_CPU))
        receipt["bootstrap_cpu_seconds_whole_process"] = time.process_time()
        receipt["bootstrap_scope"] = (
            "soft10 with default SIGXCPU, hard890 only until both children spawned; hard10 thereafter"
        )
        result = phase(
            controller,
            collect_command,
            890,
            limits["collect_soft"],
            limits["grace"],
            log,
            "collect",
            start_wall,
        )
        receipt["phases"].append(result)
        if result["exit_code"] != 0 or result["stop_reason"]:
            raise RuntimeError("collection did not exit successfully; reanalysis not started")
        receipt["collection_summary"] = collect_complete(evidence, limits["capacity"], synthetic)
        if time.monotonic() - start_wall > limits["collect_soft"] + limits["grace"]:
            raise TimeoutError("collection completion check exceeded whole-phase wall ceiling")
        analysis_started = time.monotonic()
        analyser.stdin.write(b"G")
        analyser.stdin.flush()
        analyser.stdin.close()
        result = phase(
            analyser,
            analysis_command,
            600,
            limits["analysis_soft"],
            limits["grace"],
            log,
            "reanalyse",
            analysis_started,
        )
        receipt["phases"].append(result)
        if result["exit_code"] != 0 or result["stop_reason"]:
            raise RuntimeError("reanalysis failed or did not accept; preserved report/evidence")
        report, report_sha = bounded_json(attempt / "report/report.json", REPORT_BYTES)
        expected_status = "synthetic_test_pass" if synthetic else "accepted"
        if report.get("status") != expected_status:
            raise ValueError("successful process returned a nonaccepting report")
        if not synthetic and report.get("synthetic_fixture_only"):
            raise ValueError("synthetic report cannot establish acceptance")
        if not synthetic:
            validate_ledger(receipt["cumulative_ledger"], release_sha256=args.release_sha256)
        receipt["report_sha256"] = report_sha
        receipt["status"] = "synthetic_workflow_completed" if synthetic else "completed"
    except BaseException as exc:
        receipt["failure"] = {"type": type(exc).__name__, "reason": str(exc)[:2048]}
    finally:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        for child in children:
            if child.returncode is None:
                cleanup = cleanup_reap(child, time.monotonic() + 1)
                receipt.setdefault("exceptional_child_cleanup", []).append(
                    dict(cleanup, pid=child.pid)
                )
                if not cleanup["reaped"]:
                    receipt["unreaped_child_pid"] = child.pid
            if child.stdin is not None and not child.stdin.closed:
                child.stdin.close()
            if child.stdout is not None and not child.stdout.closed:
                child.stdout.close()
        record_final_children(receipt)
        receipt["wall_seconds"] = time.monotonic() - start_wall
        receipt["launcher_cpu_seconds"] = time.process_time()
        receipt["launcher_observed_post_cleanup"] = snapshot()
        receipt["launcher_scope_delta"] = delta(launcher_start_snapshot, snapshot())
        receipt["requested_launcher_cpu_ceiling"] = 10
        receipt["strict_cpu_enforcement_verified"] = False
        if receipt["launcher_cpu_seconds"] > 10:
            receipt["status"] = "incomplete"
            receipt["observed_resource_violations"] = ["launcher_process_cpu_exceeded"]
        receipt["launcher_cpu_includes_imports_and_both_phases"] = True
        receipt["launcher_cpu_observation_excludes_final_receipt_write"] = True
        receipt["combined_pipe_log_base64"] = base64.b64encode(log).decode("ascii")
        receipt["combined_pipe_bytes_retained"] = len(log)
        receipt["approval_scope"] = (
            "synthetic test only"
            if synthetic
            else "collector validates actual release/source/runtime; launcher does not grant approval"
        )
        exclusive_json(attempt / "run-receipt.json", receipt, RECEIPT_BYTES)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--resource-envelope", required=True)
    parser.add_argument("--resource-envelope-sha256", required=True)
    parser.add_argument("--out", required=True, help="New attempt directory, never reused")
    args = parser.parse_args()
    if not sys.dont_write_bytecode:
        raise ValueError("workflow requires Python -B")
    receipt = workflow(args)
    print(json.dumps({"status": receipt["status"], "failure": receipt.get("failure")}))
    return 0 if receipt["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
