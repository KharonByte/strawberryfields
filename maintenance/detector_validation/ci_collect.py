"""Single-attempt CI collection and raw-data aggregation under a prospective release."""

import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import resource
import signal
import sys
import time

from .ci import LIMITS, SHARDS, cases_for, context, git, require, validate_release
from .collect import run_worker
from .fresh import load, save, validate_metadata
from .fresh_profile import profile
from .provenance import (
    ROOT,
    digest,
    parse_json,
    runtime_identity,
    sanitized_environment,
    source_identity,
)
from .reanalyse import evaluate_counts, parse_npy, raw_indicators, replay_period, structurally_valid
from .storage import StoreReader, StoreWriter, encoded, sha


def runtime_content(runtime):
    return {k: v for k, v in runtime.items() if k not in ("executable", "invocation_executable")}


def binding(release_sha, role, shard, source, runtime):
    return {
        **context(release_sha, role, shard),
        "source": source,
        "runtime": runtime,
        "source_root": str(ROOT),
        "case_ids": [c["id"] for c in cases_for(shard)],
        "limits": LIMITS,
    }


def cpu():
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time() + children.ru_utime + children.ru_stime


def collect(args, release):
    root = Path(args.out).absolute()
    root.mkdir(parents=True, exist_ok=False)
    source, runtime = source_identity(), runtime_identity()
    require(source["simulator_commit"] == release[args.role], "wrong simulator checkout")
    bound = binding(args.sha256, args.role, args.shard, source, runtime)
    save(root / "binding.json", bound)
    start, start_cpu = time.monotonic(), cpu()
    report = {"status": "incomplete", "failure": None, "completed_chunks": 0}
    store = None
    try:
        store = StoreWriter(
            root / "evidence.bin", LIMITS["capacity_bytes"], LIMITS["reserve_bytes"]
        )
        store.put("binding.json", encoded(bound))
        store.checkpoint({"status": "incomplete"})
        for case in cases_for(args.shard):
            for chunk in range(case["chunk_count"]):
                require(
                    cpu() - start_cpu + 125 <= LIMITS["cpu_seconds"], "CPU allocation exhausted"
                )
                require(
                    time.monotonic() - start + 185 <= LIMITS["wall_seconds"],
                    "wall allocation exhausted",
                )
                prefix = f"{case['id']}/{chunk:02d}"
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "maintenance.detector_validation.fresh_worker",
                    "--case-sha256",
                    sha(encoded(case)),
                    "--chunk",
                    str(chunk),
                ]
                store.put(f"commands/{prefix}/command.json", encoded(command))
                result, meta = run_worker(
                    store, command, encoded(case), sanitized_environment(), prefix, 120, 180
                )
                store.checkpoint({"status": "incomplete"})
                require(
                    result["exit_code"] == 0 and result["limit_failure"] is None,
                    "worker failure; no retry",
                )
                validate_metadata(meta, case, chunk, result, runtime, source)
                report["completed_chunks"] += 1
        require(source_identity() == source, "source changed during collection")
        require(runtime_identity() == runtime, "runtime changed during collection")
        report["status"] = "collected_not_accepted"
    except BaseException as error:
        report["failure"] = type(error).__name__ + ": " + str(error)[:2000]
    finally:
        report["cpu_seconds"] = cpu() - start_cpu
        report["wall_seconds"] = time.monotonic() - start
        report["strict_cpu_guarantee"] = False
        if (
            report["cpu_seconds"] > LIMITS["cpu_seconds"]
            or report["wall_seconds"] > LIMITS["wall_seconds"]
        ):
            report["status"] = "incomplete"
            report["failure"] = report["failure"] or "postflight resource overrun"
        if store is not None:
            try:
                store.put("collection.json", encoded(report))
                store.checkpoint({"status": report["status"]})
            finally:
                store.close()
        save(root / "collection.json", report)
        # The raw file includes unused bounded capacity. Preserve the exact original bytes.
        files = {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()}
        save(root / "checksums.json", files)
    return 0 if report["status"] == "collected_not_accepted" else 1


def validate_binding(bound, expected, source, runtime):
    for key, value in expected.items():
        require(bound.get(key) == value, "shard binding mismatch: " + key)
    require(bound.get("source") == source, "shard source mismatch")
    require(
        runtime_content(bound.get("runtime", {})) == runtime_content(runtime),
        "shard runtime mismatch",
    )
    require(
        bound.get("case_ids") == [c["id"] for c in cases_for(expected["shard"])],
        "shard case ownership mismatch",
    )
    require(bound.get("limits") == LIMITS, "shard limits mismatch")
    require(
        isinstance(bound.get("source_root"), str) and Path(bound["source_root"]).is_absolute(),
        "invalid source location",
    )


def portable_metadata(meta, bound, runtime):
    """Translate location fields only; validate original exact paths before translation.

    Content hashes, versions, cases, seeds and every execution-policy field remain intact.
    The ordinary metadata validator then hashes actual local source/distribution files.
    """
    result = copy.deepcopy(meta)
    old_root = Path(bound["source_root"])
    old_env = Path(bound["runtime"]["invocation_executable"]).absolute().parent.parent
    new_env = Path(runtime["invocation_executable"]).absolute().parent.parent
    for field in ("imports", "imports_after"):
        for item in result.get(field, []):
            path = Path(item["path"])
            require(path.is_absolute() and ".." not in path.parts, "unsafe import location")
            if path.is_relative_to(old_root):
                item["path"] = str(ROOT / path.relative_to(old_root))
            elif path.is_relative_to(old_env):
                item["path"] = str(new_env / path.relative_to(old_env))
            else:
                raise ValueError("import outside recorded source/runtime")
    return result


def inspect_shard(directory, args, source, runtime):
    """Validate every raw member and recompute scientific criteria, never trust a verdict file."""
    files = list(directory.iterdir())
    require(all(p.is_file() and not p.is_symlink() for p in files), "unexpected artifact member")
    require(
        {p.name for p in files}
        == {"binding.json", "collection.json", "evidence.bin", "checksums.json"},
        "missing/extra artifact files",
    )
    checksums = load(directory / "checksums.json")
    require(
        (directory / "evidence.bin").stat().st_size == LIMITS["capacity_bytes"],
        "container size mismatch",
    )
    require(
        all(p.stat().st_size <= 4 * 1024**2 for p in files if p.name != "evidence.bin"),
        "oversized artifact metadata",
    )
    require(
        checksums == {p.name: digest(p) for p in files if p.name != "checksums.json"},
        "artifact checksum mismatch",
    )
    bound = load(directory / "binding.json")
    shard = bound.get("shard")
    cases = cases_for(shard)
    validate_binding(bound, context(args.sha256, args.role, shard), source, runtime)
    reader = StoreReader(directory / "evidence.bin")
    try:
        require(
            reader.capacity == LIMITS["capacity_bytes"]
            and reader.reserve == LIMITS["reserve_bytes"],
            "storage allocation mismatch",
        )
        require(
            parse_json(reader.read_bytes("binding.json")) == bound, "container binding mismatch"
        )
        collection = parse_json(reader.read_bytes("collection.json"))
        require(collection == load(directory / "collection.json"), "collection receipt mismatch")
        require(reader.state == {"status": "collected_not_accepted"}, "incomplete checkpoint")
        require(
            collection.get("status") == "collected_not_accepted"
            and collection.get("failure") is None,
            "incomplete collection",
        )
        require(collection.get("completed_chunks") == 40, "incomplete chunk count")
        for key in ("cpu_seconds", "wall_seconds"):
            value = collection.get(key)
            require(
                type(value) in (float, int) and 0 <= value <= LIMITS[key],
                "resource evidence invalid",
            )
        inventory = reader.inventory()
        allowed = {"binding.json", "collection.json"}
        for case in cases:
            for chunk in range(10):
                prefix = f"{case['id']}/{chunk:02d}"
                allowed.update(
                    f"commands/{prefix}/{name}"
                    for name in (
                        "command.json",
                        "started.json",
                        "exit.json",
                        "stdout.txt",
                        "stderr.txt",
                    )
                )
                allowed.update(f"raw/{prefix}/{name}" for name in ("samples.npy", "metadata.json"))
                if case["path"] == "local_engine_gaussian":
                    allowed.add(f"raw/{prefix}/secondary.npy")
        require(set(inventory) == allowed, "missing, unexpected or duplicate chunk members")
        for name in inventory:
            reader.read_bytes(name)
        reports = []
        for case in cases:
            rows = []
            for chunk in range(10):
                prefix = f"{case['id']}/{chunk:02d}"
                meta = parse_json(reader.read_bytes(f"raw/{prefix}/metadata.json"))
                exit_record = parse_json(reader.read_bytes(f"commands/{prefix}/exit.json"))
                require(
                    exit_record.get("exit_code") == 0 and exit_record.get("limit_failure") is None,
                    "worker failed",
                )
                require(
                    exit_record.get("requested_cpu_limit") == 120
                    and exit_record.get("supervision", {}).get("requested_wall_ceiling_seconds")
                    == 180,
                    "worker limits differ",
                )
                expected_command = [
                    bound["runtime"]["invocation_executable"],
                    "-B",
                    "-m",
                    "maintenance.detector_validation.fresh_worker",
                    "--case-sha256",
                    sha(encoded(case)),
                    "--chunk",
                    str(chunk),
                ]
                require(
                    parse_json(reader.read_bytes(f"commands/{prefix}/command.json"))
                    == expected_command,
                    "worker command mismatch",
                )
                validate_metadata(
                    portable_metadata(meta, bound, runtime),
                    case,
                    chunk,
                    exit_record,
                    runtime,
                    source,
                )
                block = parse_npy(
                    reader.read_bytes(f"raw/{prefix}/samples.npy"),
                    10000,
                    len(case["output_mode_order"]),
                )
                require(
                    all(
                        all(v >= 0 and (case["detector"] != "threshold" or v <= 1) for v in r)
                        for r in block
                    ),
                    "invalid detector sample domain",
                )
                if case["path"] == "local_engine_gaussian":
                    secondary = parse_npy(
                        reader.read_bytes(f"raw/{prefix}/secondary.npy"),
                        10000,
                        len(case["measured_modes"]),
                    )
                    mapping = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
                    require(
                        all(
                            tuple(row[j] for j in mapping) == primary
                            for row, primary in zip(secondary, block)
                        ),
                        "Engine samples mapping mismatch",
                    )
                rows.extend(block)
            require(len(rows) == 100000, "full-N case required")
            evaluation = evaluate_counts(
                case,
                Counter(k for row in rows for k in raw_indicators(case, row)),
                len(rows),
                profile("scientific")["statistics"],
            )
            state = {"accepted": "PASS", "defect": "FAIL", "inconclusive": "INCOMPLETE"}[
                evaluation["status"]
            ]
            if (
                any(not structurally_valid(case, r) for r in rows)
                or replay_period(rows, case) is not None
            ):
                state = "FAIL"
            reports.append(
                {
                    "id": case["id"],
                    "rows": len(rows),
                    "scientific_status": state,
                    "evaluation": evaluation,
                }
            )
        return shard, reports, collection
    finally:
        reader.close()


def summarize(case_reports, errors):
    expected = [c["id"] for c in profile("scientific")["cases"]]
    ids = [c["id"] for c in case_reports]
    complete = (
        not errors
        and len(ids) == 68
        and len(set(ids)) == 68
        and set(ids) == set(expected)
        and all(c["rows"] == 100000 for c in case_reports)
    )
    states = [c["scientific_status"] for c in case_reports]
    verdict = (
        "FAIL"
        if "FAIL" in states
        else "PASS" if complete and set(states) == {"PASS"} else "INCOMPLETE"
    )
    return {
        "integrity_status": "PASS" if complete else "INCOMPLETE",
        "coverage_status": "COMPLETE" if complete else "INCOMPLETE",
        "scientific_verdict": verdict,
        "required_cases": 68,
        "required_chunks": 680,
        "required_rows": 6800000,
        "required_criteria": 1022,
        "missing_cases": [c for c in expected if c not in ids],
        "errors": errors,
        "cases": case_reports,
    }


def aggregate(args, release):
    source, runtime = source_identity(), runtime_identity()
    require(source["simulator_commit"] == release[args.role], "wrong analyser revision")
    reports, errors, shards, resources, checksums = [], [], set(), [], {}
    try:
        directories = list(Path(args.input).iterdir())
    except OSError as error:
        directories = []
        errors.append({"input": "missing artifact directory: " + str(error)[:500]})
    if len(directories) > SHARDS:
        errors.append({"input": "too many shard artifacts; only first 17 inspected"})
        directories = sorted(directories)[:SHARDS]
    for directory in sorted(directories):
        try:
            require(directory.is_dir() and not directory.is_symlink(), "unexpected download member")
            shard, result, observed = inspect_shard(directory, args, source, runtime)
            require(shard not in shards, "duplicate shard artifact")
            shards.add(shard)
            reports.extend(result)
            resources.append(observed)
            checksums[directory.name] = load(directory / "checksums.json")
        except Exception as error:
            errors.append(
                {
                    "artifact": directory.name,
                    "error": type(error).__name__ + ": " + str(error)[:1000],
                }
            )
    if shards != set(range(SHARDS)):
        errors.append({"missing_shards": sorted(set(range(SHARDS)) - shards)})
    report = {
        **context(args.sha256, args.role),
        **summarize(reports, errors),
        "resources": resources,
        "runtime_content": runtime_content(runtime),
        "artifact_checksums": checksums,
        "confidence_scope": "simultaneous 99% within this revision; not a 99% two-revision claim",
    }
    save(Path(args.out), report)
    # Integrity permits candidate collection even when the complete base scientifically fails.
    return 0 if report["integrity_status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("collect", "aggregate"))
    parser.add_argument("--release", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--role", choices=("base", "candidate"), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    require(sys.dont_write_bytecode and not sys.flags.optimize, "use python -B without -O")
    context(args.sha256, args.role, args.shard)
    release = validate_release(args.release, args.sha256)
    require(
        set(git("diff", "--name-only", "HEAD").splitlines()) <= set(release["suite_files"]),
        "simulator source differs from frozen revision",
    )
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    # Controller CPU bound is distinct from child reservations and job wall timeout.
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CI wall deadline"))
    )
    signal.alarm(3360 if args.command == "collect" else 1200)
    return globals()[args.command](args, release)


if __name__ == "__main__":
    raise SystemExit(main())
