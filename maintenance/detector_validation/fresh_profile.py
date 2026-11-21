"""Public fresh experiment profiles; operational rows are never accuracy evidence."""

import copy
import json
from pathlib import Path


def profile(name):
    original = json.loads((Path(__file__).parent / "cases.json").read_text())
    if name == "scientific-v5":
        result = profile("scientific-v4")
        result["purpose"] = "full_frozen_scientific_matrix_preservation_v5"
        result["resources"]["preservation_ceiling_bytes"] = 6 * 1024**3
        return result
    if name == "scientific-v4":
        result = profile("scientific-v3")
        result["purpose"] = "full_frozen_scientific_matrix_resource_v4"
        result["resources"].update(
            worker_cpu_seconds=300,
            worker_wall_seconds=360,
            worker_cpu_timer_seconds=290,
            child_cpu_tripwire_seconds=290,
            cpu_seconds=72000,
            wall_seconds=86400,
            preservation_ceiling_bytes=5 * 1024**3,
        )
        return result
    if name == "scientific-v3":
        result = profile("scientific-v2")
        result["purpose"] = "full_frozen_scientific_matrix_resource_v3"
        result["resources"].update(
            worker_cpu_timer_seconds=170,
            timer_observation_tolerance_seconds=0.01,
            monitor_gap_policy="telemetry_with_worker_cpu_timer",
        )
        return result
    if name == "scientific-v2":
        result = profile("scientific")
        result["purpose"] = "full_frozen_scientific_matrix_resource_v2"
        result["resources"].update(
            worker_cpu_seconds=180,
            worker_wall_seconds=240,
            controller_chunk_cpu_seconds=20,
            controller_cpu_tripwire_seconds=18,
            child_cpu_tripwire_seconds=170,
            child_poll_seconds=0.02,
            finalization_cpu_seconds=60,
            finalization_wall_seconds=60,
            controller_cpu_seconds=13680,
            report_cpu_seconds=120,
            report_wall_seconds=180,
        )
        return result
    if name == "scientific":
        return {
            "purpose": "full_frozen_scientific_matrix",
            "accuracy": "requires_complete_original_criteria",
            "cases": original["cases"],
            "statistics": dict(
                original["statistics"], comparison_count=original["comparison_count"]
            ),
            "resources": {
                "cpu_seconds": 43200,
                "wall_seconds": 57600,
                "capacity_bytes": 768 * 1024**2,
            },
        }
    if name != "operational":
        raise ValueError("unsupported profile")
    vacuum = copy.deepcopy(original["cases"][0])
    coherent = copy.deepcopy(
        next(
            c
            for c in original["cases"]
            if c["path"] == "gaussian_direct"
            and c["family"] == "coherent_product"
            and c["detector"] == "threshold"
        )
    )
    vacuum.update(
        id="portable_v7__vacuum_pnr",
        comparison_count=0,
        shots=4,
        chunk_rows=4,
        chunk_count=1,
        chunk_seeds=[270007001],
    )
    coherent.update(
        id="portable_v7__asymmetric_coherent_threshold",
        modes=2,
        measured_modes=[1, 0],
        output_mode_order=[1, 0],
        eta=[1, 0.4],
        parameters={"alpha_re_im": [[0.1, 0.05], [-0.2, 0.1]]},
        comparison_count=0,
        shots=4,
        chunk_rows=4,
        chunk_count=1,
        chunk_seeds=[270007002],
    )
    return {
        "purpose": "operational_cold_start_diagnostic_v8",
        "diagnostic_revision": "v8_one_new_attempt_no_accuracy_pooling",
        "accuracy": "UNTESTED",
        "cases": [vacuum, coherent],
        "statistics": None,
        "resources": {
            "cpu_seconds": 300,
            "wall_seconds": 600,
            "worker_cpu_seconds": 120,
            "worker_wall_seconds": 240,
            "capacity_bytes": 8 * 1024**2,
        },
    }
