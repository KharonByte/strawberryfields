"""Test-only detector workflow: synthetic smoke, retained reanalysis, or explicitly frozen full run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time


def output(path, report):
    target = Path(path).absolute()
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("symlink output forbidden")
    data = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(data) > 1024**2:
        raise ValueError("report exceeds 1MiB")
    with target.open("xb") as stream:
        stream.write(data)


def analytic_controls():
    """Small exact-support analytic controls, no simulated or candidate outcomes."""
    from fractions import Fraction as F
    from .oracle import reference
    from .reanalyse import reference_b, evaluate_counts
    from .binomial import clopper_pearson

    cases = json.loads((Path(__file__).parent / "cases.json").read_text())["cases"]
    controls = []
    for family in (
        "vacuum",
        "coherent_product",
        "thermal",
        "squeezed_vacuum",
        "two_mode_squeezed",
        "coherent_interference",
    ):
        case = next(
            c
            for c in cases
            if c["family"] == family and c["detector"] == "pnr" and all(e > 0 for e in c["eta"])
        )
        primary = reference(case).comparisons
        secondary = reference_b(case)
        # Independently derived references must overlap for every original indicator.
        for key, first in primary.items():
            other = secondary[key]
            if first.hi < other.lo or other.hi < first.lo:
                raise AssertionError("independent analytic interval disagreement")
        controls.append(
            {"family": family, "case_id": case["id"], "indicators": len(primary), "status": "pass"}
        )
    bound = clopper_pearson(100000, 0)
    lo, hi = bound["lower_outward"], bound["upper_outward"]
    if lo != 0 or not 0 < hi < F(1, 100):
        raise AssertionError("zero-event confidence bound")
    manifest = json.loads((Path(__file__).parent / "cases.json").read_text())
    case = next(
        c
        for c in cases
        if c["family"] == "coherent_product"
        and c["detector"] == "threshold"
        and all(e > 0 for e in c["eta"])
    )
    primary, secondary = reference(case).comparisons, reference_b(case)
    assert all(not (x.hi < secondary[k].lo or secondary[k].hi < x.lo) for k, x in primary.items())
    wrong = {k: 0 for k in primary}
    wrong[max(primary, key=lambda k: primary[k].midpoint)] = 0
    wrong[min(primary, key=lambda k: primary[k].midpoint)] = case["shots"]
    assert (
        evaluate_counts(
            case,
            wrong,
            case["shots"],
            dict(manifest["statistics"], comparison_count=manifest["comparison_count"]),
        )["status"]
        == "defect"
    )
    try:
        evaluate_counts(
            case,
            wrong,
            1,
            dict(manifest["statistics"], comparison_count=manifest["comparison_count"]),
        )
    except ValueError as error:
        assert str(error) == "incomplete sample budget"
    else:
        raise AssertionError("incomplete outcomes accepted")
    controls.append(
        {
            "family": "coherent_threshold",
            "case_id": case["id"],
            "status": "pass",
            "wrong_distribution_rejected": True,
            "insufficient_rows_rejected": True,
        }
    )
    return controls


def smoke():
    """Tiny in-memory synthetic framing/checkpoint exercise, never an accuracy test."""
    import struct
    from .collect import FrameParser
    from .output_bounds import storage_plan

    class Memory:
        def __init__(self):
            self.members = {}

        def put(self, name, data):
            if sum(map(len, self.members.values())) + len(data) > 65536:
                raise ValueError("smoke output bound")
            self.members[name] = data

    memory = Memory()
    from .collect import run_worker
    from .provenance import sanitized_environment

    child = r"""import os,sys,struct
from maintenance.detector_validation.stream import EvidenceStream
s=EvidenceStream(int(os.environ['DETECTOR_EVIDENCE_FD']))
sys.stdin.buffer.read()
s.emit(1,b'{}')
header=b"{'descr': '<i8', 'fortran_order': False, 'shape': (2, 1), }"
header+=b' '*((64-(10+len(header)+1)%64)%64)+b'\n'
raw=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<qq',0,1)
s.emit(2,raw)
s.emit(5,b'{}')
"""
    normal, _ = run_worker(
        memory,
        [sys.executable, "-B", "-c", child],
        b"{}",
        sanitized_environment(),
        "synthetic/00",
        2,
        3,
    )
    assert normal["exit_code"] == 0 and normal["limit_failure"] is None
    failure, _ = run_worker(
        memory,
        [sys.executable, "-B", "-c", "raise RuntimeError('planted smoke failure')"],
        b"{}",
        sanitized_environment(),
        "synthetic-failure/00",
        2,
        3,
    )
    assert failure["exit_code"] != 0 and failure["limit_failure"] is not None
    from .retained import npy as parse_npy

    raw = memory.members["raw/synthetic/00/samples.npy"]
    rows = parse_npy(raw, 2, 1)
    assert rows == [(0,), (1,)]
    retained_synthetic = {
        name: {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "text": (
                data.decode("utf-8", errors="replace") if name.endswith((".log", ".txt")) else None
            ),
        }
        for name, data in memory.members.items()
    }
    return {
        "schema": "detector-test-report-v6",
        "mode": "smoke",
        "execution_status": "completed",
        "scientific_verdict": "NOT_EVALUATED",
        "coverage_status": "SYNTHETIC_ONLY",
        "synthetic": True,
        "acceptance_claim": False,
        "analytic_controls": analytic_controls(),
        "synthetic_members": retained_synthetic,
        "synthetic_rows": rows,
        "collector_controls": {"normal": normal, "planted_failure": failure},
        "storage_plan": storage_plan(),
        "strict_cpu_guarantee": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    commands.add_parser("smoke").add_argument("--out", required=True)
    retained = commands.add_parser("retained")
    for name in ("container", "manifest", "release", "release-sha256", "references", "out"):
        retained.add_argument("--" + name, required=True)
    full = commands.add_parser("full")
    for name in (
        "manifest",
        "release",
        "release-sha256",
        "resource-envelope",
        "resource-envelope-sha256",
        "out",
    ):
        full.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if sys.flags.optimize:
        raise ValueError("optimized Python disables integrity assertions")
    if not sys.dont_write_bytecode:
        raise ValueError("use Python -B")
    if args.mode == "full":
        from .run import bounded_json, workflow
        from .envelope import validate_envelope

        release, digest = bounded_json(args.release, 4 * 1024**2)
        if digest != args.release_sha256:
            raise ValueError("explicit release digest mismatch")
        validate_envelope(release, args.resource_envelope, args.resource_envelope_sha256)
        receipt = workflow(args)
        return 0 if receipt["status"] == "completed" else 2
    before, started = time.process_time(), time.monotonic()
    cpu, wall = (20, 30) if args.mode == "smoke" else (60, 120)
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("workflow wall deadline"))
    )
    signal.alarm(wall)
    try:
        if args.mode == "smoke":
            report = smoke()
        else:
            from .retained import retained_report

            report = retained_report(args)
        report["resources"] = {
            "requested_cpu_seconds": cpu,
            "requested_wall_seconds": wall,
            "observed_process_cpu_seconds": time.process_time() - before,
            "observed_wall_seconds": time.monotonic() - started,
            "strict_cpu_guarantee": False,
            "observation_excludes_final_receipt_write": True,
        }
        output(args.out, report)
    except Exception as error:
        report = {
            "schema": "detector-test-report-v6",
            "mode": args.mode,
            "execution_status": "failed",
            "scientific_verdict": "NOT_EVALUATED",
            "coverage_status": "INCOMPLETE",
            "failure": type(error).__name__ + ": " + str(error)[:2000],
            "resources": {
                "requested_cpu_seconds": cpu,
                "requested_wall_seconds": wall,
                "observed_process_cpu_seconds": time.process_time() - before,
                "observed_wall_seconds": time.monotonic() - started,
                "strict_cpu_guarantee": False,
            },
        }
        output(args.out, report)
        return 1
    finally:
        signal.alarm(0)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("mode", "execution_status", "scientific_verdict", "coverage_status")
            }
        )
    )
    return 0 if args.mode == "smoke" else 2 if report["scientific_verdict"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
