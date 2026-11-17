"""Portable fresh detector experiment: init, external review, freeze, run, report.

Fresh IDs never continue, overwrite or rehabilitate historical experiments.
Public review JSON binds bytes; it is not cryptographic human authentication.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import uuid

from .fresh_profile import profile
from .provenance import (
    source_identity,
    runtime_identity,
    sanitized_environment,
    parse_json,
)
from .run import bounded_json, exclusive_json, safe_path
from .storage import StoreWriter, StoreReader, encoded

MAX_JSON = 4 * 1024**2
ROLES = ("qa", "scientific", "security")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw_bytes(path, maximum=MAX_JSON):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("bounded regular input required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(maximum + 1)
        if len(data) != info.st_size:
            raise ValueError("input changed during read")
        return data
    finally:
        os.close(fd)


def validate_accounting(value, resources):
    if value.get("schema") != "detector-public-accounting-v7" or value.get("scope") not in (
        "new_independent_budget",
        "existing_budget_new_activity",
    ):
        raise ValueError("explicit portable accounting scope required")
    if not isinstance(value.get("scope_id"), str) or not value["scope_id"].strip():
        raise ValueError("public enclosing budget identity required")
    prior = value.get("prior_cpu_seconds")
    wall = value.get("prior_wall_seconds")
    if value["scope"] == "new_independent_budget" and (
        prior != 0 or wall != 0 or value.get("historical_status") != "not_applicable"
    ):
        raise ValueError("new budget must be explicitly attested, not continuation")
    if value["scope"] == "existing_budget_new_activity" and value.get("historical_status") not in (
        "FAIL_INCOMPLETE_preserved",
        "prior_results_preserved",
    ):
        raise ValueError("existing evidence disposition required")
    for item in (prior, wall):
        if item is not None and (type(item) not in (int, float) or item < 0):
            raise ValueError("invalid prior debit; use null for unresolved")
    if (prior is None or wall is None) and value.get(
        "prior_accounting_status"
    ) != "unreconciled_explicit_new_activity_allowance":
        raise ValueError("unknown historical costs must be explicit")
    retained = value.get("retained_workspace_bytes")
    if (
        type(retained) is not int
        or retained < 0
        or retained + resources["capacity_bytes"] + 16 * 1024**2 > 4 * 1024**3
    ):
        raise ValueError("declared cumulative workspace preservation cap")
    if value.get("activity_allowance") != resources:
        raise ValueError("public activity allowance must match reviewed profile")
    return value


def record(path):
    data = raw_bytes(path)
    return parse_json(data), sha(data)


def load(path):
    return record(path)[0]


def save(path, value):
    exclusive_json(path, value, MAX_JSON)


def experiment(path):
    root = safe_path(path)
    manifest = load(root / "manifest.json")
    if (
        manifest.get("schema") != "detector-fresh-manifest-v7"
        or manifest.get("lineage") != "fresh_independent_experiment"
    ):
        raise ValueError("fresh manifest required; continuation is a different workflow")
    identifier = manifest.get("experiment_id", "")
    if str(uuid.UUID(identifier)) != identifier or manifest.get("parent_experiment_id") is not None:
        raise ValueError("fresh experiment identity/lineage mismatch")
    reference = manifest.get("diagnostic_reference")
    if reference is not None and (
        str(uuid.UUID(reference)) != reference or reference == identifier
    ):
        raise ValueError("invalid separate diagnostic reference")
    if manifest.get("profile_definition") != profile(manifest.get("profile")):
        raise ValueError("canonical profile mismatch; reduced scientific N forbidden")
    ledger = load(root / "ledger.json")
    if (
        ledger.get("schema") != "detector-fresh-ledger-v7"
        or ledger.get("experiment_id") != identifier
        or ledger.get("resources") != manifest["profile_definition"]["resources"]
        or ledger.get("historical_budget_reset") is not False
    ):
        raise ValueError("fresh ledger binding mismatch")
    validate_accounting(ledger.get("enclosing_accounting", {}), ledger["resources"])
    linked = ledger["enclosing_accounting"].get("diagnostic_reference")
    if linked is not None and (
        not isinstance(linked, dict)
        or linked.get("experiment_id") != reference
        or linked.get("continuation_or_row_pooling") is not False
    ):
        raise ValueError("diagnostic manifest/accounting link mismatch")
    return root, manifest


def bindings(root):
    return {
        name: sha(raw_bytes(root / name))
        for name in ("manifest.json", "source.json", "runtime.json", "ledger.json")
    }


def init(args):
    root = safe_path(args.out)
    root.mkdir(mode=0o700, parents=False, exist_ok=False)
    spec = profile(args.profile)
    accounting = validate_accounting(load(args.accounting), spec["resources"])
    manifest = {
        "schema": "detector-fresh-manifest-v7",
        "experiment_id": str(uuid.uuid4()),
        "lineage": "fresh_independent_experiment",
        "parent_experiment_id": None,
        "diagnostic_reference": args.diagnostic_reference
        or accounting.get("diagnostic_reference", {}).get("experiment_id"),
        "diagnostic_relation": "separate authorized operational attempt; no continuation or row pooling",
        "profile": args.profile,
        "profile_definition": spec,
        "historical_result_disposition": "unchanged; fresh results cannot erase or pool with historical failures",
        "accuracy_claim": False,
    }
    save(root / "manifest.json", manifest)
    save(root / "source.json", source_identity())
    save(root / "runtime.json", runtime_identity())
    save(
        root / "ledger.json",
        {
            "schema": "detector-fresh-ledger-v7",
            "experiment_id": manifest["experiment_id"],
            "resources": spec["resources"],
            "new_namespace_only": True,
            "historical_budget_reset": False,
            "workspace_preservation_accounting": "separate enclosing workspace obligation; no existing experiment is deleted or recharged",
            "single_attempt_no_retry": True,
            "enclosing_accounting": accounting,
        },
    )
    bound = bindings(root)
    for role in ROLES:
        save(
            root / ("review-" + role + ".json"),
            {
                "schema": "detector-public-review-v7",
                "role": role,
                "reviewer": "",
                "decision": "pending",
                "experiment_id": manifest["experiment_id"],
                "bindings": bound,
                "scope": spec["purpose"],
                "notes": "Complete only after independent review; generated template is not approval.",
            },
        )
    return {
        "status": "initialized_review_required",
        "experiment_id": manifest["experiment_id"],
        "bindings": bound,
    }


def freeze(args):
    root, manifest = experiment(args.experiment)
    if (root / "attempt-start.json").exists():
        raise ValueError("attempt already started")
    if source_identity() != load(root / "source.json"):
        raise ValueError("source changed since initialization")
    bound = bindings(root)
    reviews = {}
    identities = set()
    for role in ROLES:
        path = safe_path(getattr(args, role))
        review_record, digest = record(path)
        if (
            review_record.get("schema") != "detector-public-review-v7"
            or review_record.get("role") != role
            or review_record.get("decision") != "approved"
            or not isinstance(review_record.get("reviewer"), str)
            or not review_record["reviewer"].strip()
        ):
            raise ValueError("completed external review required: " + role)
        if (
            review_record.get("experiment_id") != manifest["experiment_id"]
            or review_record.get("bindings") != bound
            or review_record.get("scope") != manifest["profile_definition"]["purpose"]
        ):
            raise ValueError("review does not bind this experiment/profile")
        identities.add(review_record["reviewer"])
        reviews[role] = {"record": review_record, "sha256": sha(encoded(review_record))}
    if len(identities) != 3:
        raise ValueError("three distinct reviewer identities required")
    release = {
        "schema": "detector-fresh-release-v7",
        "experiment_id": manifest["experiment_id"],
        "bindings": bound,
        "reviews": reviews,
        "strict_cpu_guarantee": False,
        "review_identity_authentication": "external responsibility; JSON identity is not authentication",
    }
    save(root / "release.json", release)
    return {"status": "frozen", "release_sha256": sha(raw_bytes(root / "release.json"))}


def check_release(root, manifest, expected):
    release, digest = record(root / "release.json")
    if (
        digest != expected
        or release.get("schema") != "detector-fresh-release-v7"
        or release.get("experiment_id") != manifest["experiment_id"]
        or release.get("bindings") != bindings(root)
    ):
        raise ValueError("fresh release identity mismatch")
    reviews = release.get("reviews", {})
    if (
        set(reviews) != set(ROLES)
        or len({r["record"].get("reviewer") for r in reviews.values()}) != 3
    ):
        raise ValueError("release review identities invalid")
    for role, item in reviews.items():
        r = item["record"]
        if (
            item.get("sha256") != sha(encoded(r))
            or r.get("schema") != "detector-public-review-v7"
            or not isinstance(r.get("reviewer"), str)
            or not r["reviewer"].strip()
            or r.get("role") != role
            or r.get("decision") != "approved"
            or r.get("bindings") != release["bindings"]
            or r.get("experiment_id") != manifest["experiment_id"]
            or r.get("scope") != manifest["profile_definition"]["purpose"]
        ):
            raise ValueError("release review binding invalid")
    return release


def job_limits(limits, cpu_used, wall_used, controller_cpu_used=0):
    """Reserve a complete cold worker and finalization before starting a chunk."""
    if "worker_cpu_seconds" in limits:
        cpu = limits["worker_cpu_seconds"]
        wall = limits["worker_wall_seconds"]
        if (
            type(cpu) is not int
            or not 1 <= cpu <= 180
            or type(wall) is not int
            or not 0 < wall <= 240
        ):
            raise ValueError("invalid frozen worker reservation")
        controller = limits.get("controller_chunk_cpu_seconds", 5)
        final_cpu = limits.get("finalization_cpu_seconds", 0)
        final_wall = limits.get("finalization_wall_seconds", 5)
        if (
            limits["cpu_seconds"] - cpu_used < cpu + controller + final_cpu
            or limits["wall_seconds"] - wall_used < wall + final_wall
            or limits.get("controller_cpu_seconds", float("inf")) - controller_cpu_used
            < controller + final_cpu
        ):
            raise ValueError(
                "remaining budget cannot fit frozen cold worker plus finalization reserve"
            )
        return cpu, wall
    # Unchanged original scientific profile allocation behavior.
    cpu = int(min(120, limits["cpu_seconds"] - cpu_used - 5))
    wall = min(180, limits["wall_seconds"] - wall_used - 5)
    if cpu < 1 or wall <= 0:
        raise ValueError("fresh experiment resource allocation exhausted; no retry")
    return cpu, wall


def run(args):
    from .collect import run_worker

    started = time.monotonic()
    cpu_start = 0.0
    root, manifest = experiment(args.experiment)
    release = check_release(root, manifest, args.release_sha256)
    spec = manifest["profile_definition"]
    limits = spec["resources"]
    save(
        root / "attempt-start.json",
        {
            "schema": "detector-fresh-start-v7",
            "experiment_id": manifest["experiment_id"],
            "release_sha256": args.release_sha256,
            "status": "started_completion_unknown",
            "single_attempt": True,
        },
    )
    store = None
    report = {
        "schema": "detector-fresh-collection-v7",
        "experiment_id": manifest["experiment_id"],
        "profile": manifest["profile"],
        "diagnostic_reference": manifest.get("diagnostic_reference"),
        "release_sha256": args.release_sha256,
        "status": "incomplete",
        "cases": [],
        "accuracy": "UNTESTED",
        "strict_cpu_guarantee": False,
    }

    def used_cpu():
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time() - cpu_start + child.ru_utime + child.ru_stime

    try:
        if source_identity() != load(root / "source.json") or runtime_identity() != load(
            root / "runtime.json"
        ):
            raise ValueError("source/runtime drift before collection")
        store = StoreWriter(
            root / "evidence.bin",
            capacity=limits["capacity_bytes"]
            - (1024**2 if manifest["profile"] == "operational" else 0),
            reserve=1024**2 if manifest["profile"] == "operational" else 16 * 1024**2,
        )
        for name in (
            "manifest.json",
            "release.json",
            "source.json",
            "runtime.json",
            "ledger.json",
            "attempt-start.json",
        ):
            store.put(name, raw_bytes(root / name))
        store.checkpoint({"status": "started", "experiment_id": manifest["experiment_id"]})
        for case in spec["cases"]:
            row = {"id": case["id"], "chunks": []}
            report["cases"].append(row)
            for chunk in range(case["chunk_count"]):
                cpu, wall = job_limits(
                    limits, used_cpu(), time.monotonic() - started, time.process_time()
                )
                data = encoded(case)
                prefix = f"{case['id']}/{chunk:02d}"
                command = [
                    sys.executable,
                    "-B",
                    "-m",
                    "maintenance.detector_validation.fresh_worker",
                    "--case-sha256",
                    sha(data),
                    "--chunk",
                    str(chunk),
                ]
                store.put(f"commands/{prefix}/command.json", encoded(command))
                result, metadata = run_worker(
                    store,
                    command,
                    data,
                    sanitized_environment(),
                    prefix,
                    cpu,
                    wall,
                    monitored=manifest["profile"] == "scientific-v2",
                )
                row["chunks"].append({"index": chunk, **result})
                store.checkpoint(
                    {
                        "status": "incomplete",
                        "experiment_id": manifest["experiment_id"],
                        "completed_chunks": sum(len(c["chunks"]) for c in report["cases"]),
                    }
                )
                if (
                    result["exit_code"] != 0
                    or result["limit_failure"] is not None
                    or not metadata
                    or metadata.get("status") != "captured_not_accepted"
                ):
                    raise ValueError("worker failed; evidence retained; no retry")
                if manifest["profile"] == "scientific-v2":
                    validate_metadata(
                        metadata,
                        case,
                        chunk,
                        result,
                        load(root / "runtime.json"),
                        load(root / "source.json"),
                        limits,
                    )
        report["status"] = "collected_not_accepted"
    except BaseException as error:
        report["failure"] = type(error).__name__ + ": " + str(error)[:2000]
    finally:
        # Failed attempts need provenance postflight as well as successful ones.
        try:
            report["source_post_matches"] = source_identity() == load(root / "source.json")
            report["runtime_post_matches"] = runtime_identity() == load(root / "runtime.json")
        except Exception as error:
            report["postflight_identity_failure"] = type(error).__name__ + ": " + str(error)[:512]
            report["source_post_matches"] = report["runtime_post_matches"] = False
        if not report["source_post_matches"] or not report["runtime_post_matches"]:
            report["status"] = "incomplete"
        report["resources"] = {
            "observed_family_cpu_seconds": used_cpu(),
            "observed_wall_seconds": time.monotonic() - started,
            "requested": limits,
            "observation_excludes_final_receipt_write": True,
        }
        if (
            used_cpu() > limits["cpu_seconds"]
            or time.monotonic() - started > limits["wall_seconds"]
        ):
            report["status"] = "incomplete"
            report["resource_overrun"] = True
        receipt_max = 512 * 1024 if manifest["profile"] == "operational" else MAX_JSON
        if len(encoded(report)) > receipt_max:
            report = {
                "schema": "detector-fresh-collection-v7",
                "experiment_id": manifest["experiment_id"],
                "status": "incomplete",
                "failure": "terminal receipt exceeds bounded reserve",
                "resources": report["resources"],
            }
        if store is not None:
            try:
                store.put("collection.json", encoded(report))
                store.checkpoint(
                    {
                        "status": report["status"],
                        "experiment_id": manifest["experiment_id"],
                    }
                )
            finally:
                store.close()
        exclusive_json(root / "collection-receipt.json", report, receipt_max)
    return report


def validate_monitored_record(exit_record, limits):
    """Recheck observed costs and monitor evidence, not just a producer verdict."""
    supervision = exit_record.get("supervision") or {}
    monitor = supervision.get("child_cpu_monitor") or {}
    last = monitor.get("last") or {}

    def bounded(value, maximum):
        return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum

    if (
        exit_record.get("requested_controller_cpu_ceiling")
        != limits["controller_chunk_cpu_seconds"]
        or supervision.get("requested_own_cpu_tripwire_seconds")
        != limits["controller_cpu_tripwire_seconds"]
        or monitor.get("tripwire_cpu_seconds") != limits["child_cpu_tripwire_seconds"]
        or monitor.get("poll_seconds") != limits["child_poll_seconds"]
        or type(monitor.get("polls")) is not int
        or monitor["polls"] <= 0
        or not bounded(monitor.get("max_poll_gap_seconds"), 1.0)
        or type(monitor.get("affinity_count")) is not int
        or not 1 <= monitor["affinity_count"] <= 5
        or "unavailable" in last
        or not bounded(last.get("own_cpu_seconds_tick_resolution"), limits["worker_cpu_seconds"])
        or not bounded(
            (supervision.get("tracked_child_wait4") or {}).get("cpu_seconds"),
            limits["worker_cpu_seconds"],
        )
        or not bounded(
            (exit_record.get("controller_whole_chunk_delta") or {}).get("process_cpu_seconds"),
            limits["controller_chunk_cpu_seconds"],
        )
        or not bounded(exit_record.get("wall_seconds"), limits["worker_wall_seconds"])
        or supervision.get("cleanup_reaped") is not True
        or exit_record.get("postflight_resource_violations") != []
    ):
        raise ValueError("monitored worker policy evidence missing or mismatched")


def validate_metadata(meta, case, chunk, exit_record, runtime, source, frozen_limits=None):
    from .collect import check_metadata

    if (
        meta.get("rows") != case["chunk_rows"]
        or meta.get("measurement_execution", {}).get("status") != "returned"
        or meta.get("requested_mode_order") != case["measured_modes"]
    ):
        raise ValueError("measurement return/row/mode binding")
    if frozen_limits and "child_cpu_tripwire_seconds" in frozen_limits:
        validate_monitored_record(exit_record, frozen_limits)
    if frozen_limits and "worker_cpu_seconds" in frozen_limits:
        expected_cpu = frozen_limits["worker_cpu_seconds"]
        expected_wall = frozen_limits["worker_wall_seconds"]
        if (
            exit_record.get("requested_cpu_limit") != expected_cpu
            or exit_record.get("supervision", {}).get("requested_wall_ceiling_seconds")
            != expected_wall
        ):
            raise ValueError("worker reservation differs from frozen diagnostic profile")
        if meta.get("verified_resource_limits", {}).get("cpu_seconds") != {
            "soft": expected_cpu,
            "hard": expected_cpu,
        }:
            raise ValueError("actual worker CPU limit differs from frozen reservation")
    policy_view = dict(meta, rows=10000)
    check_metadata(
        policy_view,
        case,
        chunk,
        exit_record["frames"],
        exit_record["requested_cpu_limit"],
    )
    if meta.get("imports") != meta.get("imports_after") or not meta.get("imports"):
        raise ValueError("import provenance missing or changed")
    rng = meta.get("rng", {})
    if not rng.get("before_sha256") or not rng.get("after_sha256"):
        raise ValueError("RNG provenance missing")
    if case["path"] == "gaussian_direct":
        if (
            not meta.get("direct_call_sequence")
            or not meta.get("actual_pre_detector_reduced_moments")
            or not meta.get("pre_loss_physical_moments")
        ):
            raise ValueError("actual direct preparation provenance missing")
    elif not meta.get("original_circuit") or not meta.get("compiled_circuit"):
        raise ValueError("original/compiled circuit provenance missing")
    worker_record = meta.get("worker", {})
    if worker_record.get("sha256") != source["files"]["maintenance/detector_validation/worker.py"]:
        raise ValueError("worker source identity mismatch")
    allowed = Path(__file__).resolve().parents[2]
    environment = Path(runtime["invocation_executable"]).absolute().parent.parent
    expected_imports = [
        str(importlib.metadata.distribution("numpy").locate_file("numpy/__init__.py")),
        str(allowed / "strawberryfields/__init__.py"),
        str(importlib.metadata.distribution("thewalrus").locate_file("thewalrus/__init__.py")),
        str(importlib.metadata.distribution("thewalrus").locate_file("thewalrus/samples.py")),
    ]
    expected_imports += [
        str(allowed / name)
        for name in (
            "strawberryfields/engine.py",
            "strawberryfields/ops.py",
            "strawberryfields/apps/sample.py",
            "strawberryfields/decompositions.py",
            "strawberryfields/backends/gaussianbackend/backend.py",
            "strawberryfields/backends/gaussianbackend/gaussiancircuit.py",
            "strawberryfields/compilers/gaussian.py",
        )
    ]
    from .provenance import SINGLE_MODE_SOURCE, validate_sampler_configuration

    single_mode_available = SINGLE_MODE_SOURCE in source["files"]
    if single_mode_available:
        expected_imports.append(str(allowed / SINGLE_MODE_SOURCE))
    validate_sampler_configuration(meta, case, source)
    if [str(Path(r["path"]).resolve()) for r in meta["imports"]] != [
        str(Path(p).resolve()) for p in expected_imports
    ]:
        raise ValueError("exact worker import inventory mismatch")
    for imported in meta["imports"]:
        path = safe_path(imported["path"])
        if not (path.is_relative_to(allowed) or path.is_relative_to(environment)):
            raise ValueError("import outside frozen source/runtime")
        data = raw_bytes(path)
        if len(data) != imported["bytes"] or sha(data) != imported["sha256"]:
            raise ValueError("import hash mismatch")
    for name, version in meta.get("versions", {}).items():
        if runtime["pins"][name]["version"] != version:
            raise ValueError("runtime version mismatch")


def report(args):
    from .reanalyse import (
        parse_npy,
        raw_indicators,
        evaluate_counts,
        structurally_valid,
        replay_period,
    )
    from .retained import npy as tiny_npy

    root, manifest = experiment(args.experiment)
    release_sha = args.release_sha256
    check_release(root, manifest, release_sha)
    reader = StoreReader(root / "evidence.bin")
    try:
        if reader.read_bytes("manifest.json") != raw_bytes(
            root / "manifest.json"
        ) or reader.read_bytes("release.json") != raw_bytes(root / "release.json"):
            raise ValueError("container identity differs from frozen experiment")
        for name in (
            "source.json",
            "runtime.json",
            "ledger.json",
            "attempt-start.json",
        ):
            if reader.read_bytes(name) != raw_bytes(root / name):
                raise ValueError("stored provenance mismatch")
        if source_identity() != load(root / "source.json"):
            raise ValueError("analyser source differs from freeze")
        if manifest["profile"] == "scientific-v2" and runtime_identity() != load(
            root / "runtime.json"
        ):
            raise ValueError("analyser runtime differs from freeze")
        collection = parse_json(reader.read_bytes("collection.json"))
        if collection != load(root / "collection-receipt.json"):
            raise ValueError("terminal collection receipt mismatch")
        collection_ok = (
            collection.get("status") == "collected_not_accepted"
            and collection.get("source_post_matches") is True
            and collection.get("runtime_post_matches") is True
            and not collection.get("resource_overrun")
            and not collection.get("failure")
        )
        observed = collection.get("resources", {})
        requested = manifest["profile_definition"]["resources"]
        if (
            observed.get("requested") != requested
            or type(observed.get("observed_family_cpu_seconds")) not in (int, float)
            or type(observed.get("observed_wall_seconds")) not in (int, float)
        ):
            raise ValueError("collection resource evidence missing")
        if observed["observed_family_cpu_seconds"] < 0 or observed["observed_wall_seconds"] < 0:
            raise ValueError("negative observed resource use")
        collection_ok = (
            collection_ok
            and observed["observed_family_cpu_seconds"] <= requested["cpu_seconds"]
            and observed["observed_wall_seconds"] <= requested["wall_seconds"]
        )
        inventory = reader.inventory()
        for name in inventory:
            reader.read_bytes(name)
        result = {
            "schema": "detector-fresh-report-v7",
            "experiment_id": manifest["experiment_id"],
            "lineage": manifest["lineage"],
            "diagnostic_reference": manifest.get("diagnostic_reference"),
            "profile": manifest["profile"],
            "execution_status": "completed",
            "accuracy": "UNTESTED",
            "scientific_verdict": "UNTESTED",
            "historical_results": "unchanged; no pooling",
            "cases": [],
        }
        operational = manifest["profile"] == "operational"
        for case in manifest["profile_definition"]["cases"]:
            rows = []
            phase_timings = []
            missing = []
            for chunk in range(case["chunk_count"]):
                prefix = f"{case['id']}/{chunk:02d}"
                name = f"raw/{prefix}/samples.npy"
                if name not in inventory:
                    missing.append(chunk)
                    continue
                raw = reader.read_bytes(name)
                block = (
                    tiny_npy(raw, case["chunk_rows"], len(case["output_mode_order"]))
                    if operational
                    else parse_npy(raw, case["chunk_rows"], len(case["output_mode_order"]))
                )
                meta = parse_json(reader.read_bytes(f"raw/{prefix}/metadata.json"))
                exit_record = parse_json(reader.read_bytes(f"commands/{prefix}/exit.json"))
                if (
                    meta.get("case") != case
                    or meta.get("seed") != case["chunk_seeds"][chunk]
                    or meta.get("rows") != case["chunk_rows"]
                    or meta.get("output_mode_order") != case["output_mode_order"]
                    or meta.get("status") != "captured_not_accepted"
                    or exit_record.get("exit_code") != 0
                    or exit_record.get("limit_failure") is not None
                ):
                    raise ValueError("chunk provenance or exit mismatch")
                if any(
                    any(v < 0 or (case["detector"] == "threshold" and v > 1) for v in r)
                    for r in block
                ):
                    raise ValueError("raw detector domain mismatch")
                expected_command = [
                    load(root / "runtime.json")["invocation_executable"],
                    "-B",
                    "-m",
                    "maintenance.detector_validation.fresh_worker",
                    "--case-sha256",
                    sha(encoded(case)),
                    "--chunk",
                    str(chunk),
                ]
                if (
                    parse_json(reader.read_bytes(f"commands/{prefix}/command.json"))
                    != expected_command
                ):
                    raise ValueError("worker command/case binding mismatch")
                if case["path"] == "local_engine_gaussian":
                    secondary = parse_npy(
                        reader.read_bytes(f"raw/{prefix}/secondary.npy"),
                        case["chunk_rows"],
                        len(case["measured_modes"]),
                    )
                    mapping = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
                    if any(
                        tuple(row[j] for j in mapping) != primary
                        for row, primary in zip(secondary, block)
                    ):
                        raise ValueError("Engine samples_dict mapping mismatch")
                validate_metadata(
                    meta,
                    case,
                    chunk,
                    exit_record,
                    load(root / "runtime.json"),
                    load(root / "source.json"),
                    manifest["profile_definition"]["resources"],
                )
                phase_timings.append(
                    {
                        "chunk": chunk,
                        "events": meta.get("operational_phase_timings", []),
                        "capture_scope": "imports, possible JIT, preparation and measurement combined",
                    }
                )
                rows.extend(block)
            item = {
                "id": case["id"],
                "rows": len(rows),
                "missing_chunks": missing,
                "phase_timings": phase_timings,
                "scientific_status": ("UNTESTED" if operational or not rows else "INCOMPLETE"),
                "operational_status": "complete" if not missing else "incomplete",
            }
            if not operational and not missing:
                if (
                    any(not structurally_valid(case, r) for r in rows)
                    or replay_period(rows, case) is not None
                ):
                    item["scientific_status"] = "FAIL"
                    item["failure_kind"] = "support_or_replay"
                else:
                    counts = Counter(k for r in rows for k in raw_indicators(case, r))
                    scored = evaluate_counts(
                        case,
                        counts,
                        len(rows),
                        manifest["profile_definition"]["statistics"],
                    )
                    item["evaluation"] = scored
                    item["scientific_status"] = {
                        "accepted": "PASS",
                        "defect": "FAIL",
                        "inconclusive": "INCOMPLETE",
                    }[scored["status"]]
            result["cases"].append(item)
        result["coverage_status"] = (
            "OPERATIONAL_ONLY"
            if operational
            else (
                "COMPLETE"
                if all(not c["missing_chunks"] for c in result["cases"])
                else "INCOMPLETE"
            )
        )
        if not operational:
            states = [c["scientific_status"] for c in result["cases"]]
            result["scientific_verdict"] = (
                "FAIL" if "FAIL" in states else "PASS" if set(states) == {"PASS"} else "INCOMPLETE"
            )
        result["collection_status"] = collection["status"]
        result["collection_resources"] = collection.get("resources")
        result["operational_verdict"] = (
            "PASS"
            if collection_ok and all(not c["missing_chunks"] for c in result["cases"])
            else "INCOMPLETE"
        )
        if not collection_ok:
            result["execution_status"] = "incomplete_collection"
            if not operational and result["scientific_verdict"] != "FAIL":
                result["scientific_verdict"] = "INCOMPLETE"
            if not operational:
                result["coverage_status"] = "INCOMPLETE"
        result["required_scientific_cases"] = 68
        if operational:
            result["scientific_case_statuses"] = {
                c["id"]: "UNTESTED" for c in profile("scientific")["cases"]
            }
        result["accuracy_cases_tested"] = (
            0 if operational else sum(not c["missing_chunks"] for c in result["cases"])
        )
        save(Path(args.out), result)
        return result
    finally:
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("init")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--profile",
        choices=("operational", "scientific", "scientific-v2"),
        required=True,
    )
    p.add_argument("--accounting", required=True)
    p.add_argument(
        "--diagnostic-reference",
        help="Prior experiment UUID for a separate diagnostic; never continuation",
    )
    p = commands.add_parser("freeze")
    p.add_argument("--experiment", required=True)
    for role in ROLES:
        p.add_argument("--" + role, required=True)
    p = commands.add_parser("run")
    p.add_argument("--experiment", required=True)
    p.add_argument("--release-sha256", required=True)
    p = commands.add_parser("report")
    p.add_argument("--experiment", required=True)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if not sys.dont_write_bytecode or sys.flags.optimize:
        raise ValueError("use python -B without -O")
    if args.command == "run":
        _, m = experiment(args.experiment)
        lim = m["profile_definition"]["resources"]
        cpu, wall = (
            min(lim.get("controller_cpu_seconds", 890), lim["cpu_seconds"]),
            lim["wall_seconds"],
        )
    elif args.command == "report":
        _, m = experiment(args.experiment)
        lim = m["profile_definition"]["resources"]
        cpu, wall = lim.get("report_cpu_seconds", 20), lim.get("report_wall_seconds", 30)
    else:
        cpu, wall = 20, 30
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    final_wall = lim.get("finalization_wall_seconds", 0) if args.command == "run" else 0

    def deadline(*_):
        if final_wall:
            # A second, default-disposition alarm bounds exception-path finalization.
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(final_wall)
        raise TimeoutError("whole command wall deadline; finalization reserve")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(wall - final_wall)
    command_started = time.monotonic()
    try:
        result = globals()[args.command](args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "detector-fresh-command-v7",
                    "command": args.command,
                    "status": "failed",
                    "error": type(error).__name__ + ": " + str(error)[:2000],
                    "observed_process_cpu_seconds": time.process_time(),
                    "observed_command_wall_seconds": time.monotonic() - command_started,
                    "strict_cpu_guarantee": False,
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "release_sha256",
                    "experiment_id",
                    "scientific_verdict",
                )
                if k in result
            }
        )
    )
    return (
        2
        if result.get("operational_verdict") == "INCOMPLETE"
        or result.get("status") == "incomplete"
        or result.get("scientific_verdict") in ("FAIL", "INCOMPLETE")
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
