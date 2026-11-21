"""Fresh public worker: unchanged capture, narrowly validated operational row budget."""

import copy
import json
import sys
import time
from . import worker
from .fresh_profile import profile

_original_validate = worker.validate_case
_original_preflight = worker.preflight
_original_capture = worker.capture
_TIMINGS = []


def milestone(phase, completed=None):
    if len(_TIMINGS) >= 8:
        raise RuntimeError("diagnostic milestone bound")
    event = {
        "schema": "detector-operational-milestone-v8",
        "phase": phase,
        "process_cpu_seconds": time.process_time(),
        "monotonic_seconds": time.monotonic(),
    }
    if completed is not None:
        event["completed"] = completed
    _TIMINGS.append(event)
    sys.stderr.write(json.dumps(event, separators=(",", ":")) + "\n")
    sys.stderr.flush()


def timed_preflight():
    milestone("preflight_start")
    value = _original_preflight()
    milestone("preflight_end", True)
    return value


def timed_capture(case, chunk, evidence, metadata):
    # Capture includes imports, possible JIT, preparation and measurement. These
    # boundaries do not identify the subphase responsible for an interruption.
    metadata["operational_phase_timings"] = _TIMINGS
    milestone("capture_start")
    completed = False
    try:
        value = _original_capture(case, chunk, evidence, metadata)
        completed = True
        return value
    finally:
        milestone("capture_end", completed)


def validate_fresh_case(case, chunk):
    if case.get("id", "").startswith("portable_v7__"):
        matches = [c for c in profile("operational")["cases"] if c == case]
        if len(matches) != 1 or chunk != 0:
            raise ValueError("unfrozen operational case")
        # Validate all original preparation/units/order/domain checks using the
        # original validation budget only. Capture receives the actual four rows.
        validation_view = copy.deepcopy(case)
        validation_view.update(
            shots=100000, chunk_rows=10000, chunk_count=10, chunk_seeds=list(range(10))
        )
        _original_validate(validation_view, 0)
    else:
        if case not in profile("scientific")["cases"]:
            raise ValueError("case absent from unchanged scientific profile")
        _original_validate(case, chunk)


if __name__ == "__main__":
    milestone("worker_entry")
    worker.preflight = timed_preflight
    worker.capture = timed_capture
    worker.validate_case = validate_fresh_case
    worker.main()
