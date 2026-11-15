"""Public fresh experiment profiles; operational rows are never accuracy evidence."""

import copy
import json
from pathlib import Path


def profile(name):
    original = json.loads((Path(__file__).parent / "cases.json").read_text())
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
