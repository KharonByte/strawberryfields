"""Shared validation of successful worker receipts; never authorizes collection."""

import math


def bounded_number(value, maximum):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum


def validate_execution_record(
    record, *, cpu_limit=None, wall_limit=None, controller_ceiling=5, controller_tripwire=5
):
    """Check observed costs and cleanup, independently of producer success strings.

    Explicit limits bind fixed profiles. Without them, the original scientific
    profile may reserve a smaller worker, but never above its 120 CPU/180 wall caps.
    Failed or partial receipts are not eligible for acceptance through this API.
    """
    if not isinstance(record, dict):
        raise ValueError("worker execution record missing")
    supervision = record.get("supervision")
    if not isinstance(supervision, dict):
        raise ValueError("worker execution supervision missing")
    requested_cpu = record.get("requested_cpu_limit")
    requested_wall = supervision.get("requested_wall_ceiling_seconds")
    if (
        type(requested_cpu) is not int
        or not 1 <= requested_cpu <= (120 if cpu_limit is None else cpu_limit)
        or (cpu_limit is not None and requested_cpu != cpu_limit)
        or not bounded_number(requested_wall, 180 if wall_limit is None else wall_limit)
        or requested_wall == 0
        or (wall_limit is not None and requested_wall != wall_limit)
        or record.get("requested_controller_cpu_ceiling") != controller_ceiling
        or supervision.get("requested_own_cpu_tripwire_seconds") != controller_tripwire
    ):
        raise ValueError("worker execution reservation mismatch")
    if (
        type(record.get("exit_code")) is not int
        or record["exit_code"] != 0
        or record.get("limit_failure", "missing") is not None
        or type(supervision.get("exit_code")) is not int
        or supervision["exit_code"] != 0
        or supervision.get("limit_failure", "missing") is not None
        or supervision.get("cleanup_reaped") is not True
        or record.get("postflight_resource_violations") != []
    ):
        raise ValueError("worker execution did not finish cleanly")
    child_usage = supervision.get("tracked_child_wait4") or {}
    whole_chunk = record.get("controller_whole_chunk_delta") or {}
    supervised = supervision.get("controller_delta") or {}
    if any(not isinstance(item, dict) for item in (child_usage, whole_chunk, supervised)):
        raise ValueError("worker execution cost records malformed")
    if (
        not bounded_number(child_usage.get("cpu_seconds"), requested_cpu)
        or not bounded_number(record.get("cpu_seconds"), requested_cpu)
        or not bounded_number(record.get("wall_seconds"), requested_wall)
        or not bounded_number(supervision.get("observed_since_launch_wall_seconds"), requested_wall)
        or not bounded_number(whole_chunk.get("process_cpu_seconds"), controller_ceiling)
        or not bounded_number(supervised.get("process_cpu_seconds"), controller_ceiling)
    ):
        raise ValueError("worker execution observed costs missing or exceed reservation")
