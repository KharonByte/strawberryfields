"""Deterministic checker controls. These files are never candidate samples or iid evidence."""

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct
import time

from .oracle import reference, indicators, interval
from .reanalyse import (
    read_json,
    reference_b,
    evaluate_counts,
    raw_indicators,
    structurally_valid,
    replay_period,
    read_npy,
    exclusive_output,
    set_analysis_limits,
)
from .binomial import clopper_pearson, binomial_cdf

_WRITTEN = 0
_CONTROL_BYTES = 256 * 1024**2


def reserve(size):
    global _WRITTEN
    if size < 0 or _WRITTEN + size > _CONTROL_BYTES:
        raise ValueError("deterministic-control output exceeds 256MiB cap")
    _WRITTEN += size


def allocate(case, table=None):
    """One joint semantic allocation with a certificate for every scored indicator."""
    table = reference(case).semantic if table is None else table
    N = case["shots"]
    keys = sorted(table)
    counts = {row: int(table[row].midpoint * N) for row in keys}
    remainder = N - sum(counts.values())
    ranked = sorted(keys, key=lambda row: (-(table[row].midpoint * N - counts[row]), row))
    if not 0 <= remainder <= len(keys):
        raise ValueError("cannot allocate certified semantic probabilities")
    for row in ranked[:remainder]:
        counts[row] += 1
    for row in keys:
        mass, frequency = table[row], F(counts[row], N)
        if max(abs(frequency - mass.lo), abs(frequency - mass.hi)) > F(1, N):
            raise ValueError("semantic allocation fails 1/N certificate")
        if mass.hi == 0 and counts[row]:
            raise ValueError("allocated an exact-zero category")
        if counts[row] and not structurally_valid(case, row):
            raise ValueError("semantic representative violates physical support")
    scored = Counter()
    bounds = {}
    for row, count in counts.items():
        for name in indicators(case, row):
            scored[name] += count
    for name in reference(case).comparisons:
        s = sum(name in indicators(case, row) for row in keys)
        bounds[name] = float(F(min(s, len(keys) - s), N))
    return (
        counts,
        scored,
        {
            "categories": len(keys),
            "rows": N,
            "all_category_errors_le_1_over_N": True,
            "indicator_rounding_bounds": bounds,
        },
    )


def emit_rows(allocation):
    left = dict(allocation)
    keys = sorted(left)
    while any(left.values()):
        for row in keys:
            if left[row]:
                left[row] -= 1
                yield row


def write_npy(path, rows, columns):
    """Fixed numeric raw format; no NumPy or pickle dependency."""
    header = repr({"descr": "<i8", "fortran_order": False, "shape": (len(rows), columns)})
    encoded = header.encode("ascii")
    padding = 128 - 10 - len(encoded) - 1
    if padding < 0:
        raise ValueError("fixed header budget")
    encoded += b" " * padding + b"\n"
    reserve(128 + len(rows) * columns * 8)
    with Path(path).open("xb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(encoded)) + encoded)
        for row in rows:
            stream.write(struct.pack("<" + "q" * columns, *row))


def _json(path, data):
    payload = (json.dumps(data, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > 4 * 1024**2:
        raise ValueError("control JSON exceeds 4MiB")
    reserve(len(payload))
    with Path(path).open("xb") as stream:
        stream.write(payload)


def _stats(manifest):
    return dict(manifest["statistics"], comparison_count=manifest["comparison_count"])


def _find(manifest, family, eta, detector="pnr", path="gaussian_direct"):
    return next(
        c
        for c in manifest["cases"]
        if c["family"] == family
        and c["eta"] == eta
        and c["detector"] == detector
        and c["path"] == path
    )


def _save_table(root, name, case, table, statistics):
    directory = root / name
    directory.mkdir(mode=0o700)
    allocation, scored, certificate = allocate(case, table)
    rows = list(emit_rows(allocation))
    # Retain all original synthetic rows; never interpret their deterministic order as iid.
    for chunk in range(10):
        write_npy(
            directory / f"{chunk:02d}.npy",
            rows[chunk * 10000 : (chunk + 1) * 10000],
            len(case["output_mode_order"]),
        )
    reread = Counter()
    reread_count = 0
    for chunk in range(10):
        saved = read_npy(directory / f"{chunk:02d}.npy", 10000, len(case["output_mode_order"]))
        for raw_row in saved:
            reread.update(raw_indicators(case, raw_row))
            if not structurally_valid(case, raw_row):
                raise ValueError("saved raw row violates support")
        reread_count += len(saved)
    if reread != scored or reread_count != len(rows):
        raise ValueError("saved raw rows disagree with joint allocation")
    report = evaluate_counts(case, reread, reread_count, statistics)
    report["counts_recomputed_from_persisted_raw"] = True
    _json(
        directory / "recipe.json",
        {
            "evidence_type": "synthetic_checker_control",
            "case_id": case["id"],
            "arrangement": "lexicographic round-robin semantic categories",
            "allocation": [
                {"representative": list(row), "count": count} for row, count in allocation.items()
            ],
            "certificate": certificate,
            "iid_claim": False,
            "candidate_evidence_eligible": False,
        },
    )
    _json(directory / "report.json", report)
    return report


def run_controls(manifest_path, output):
    manifest = read_json(manifest_path)
    global _WRITTEN
    _WRITTEN = 0
    root = exclusive_output(output)
    started = time.monotonic()
    _json(root / "input-manifest.json", manifest)
    code_identity = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("oracle.py", "reanalyse.py", "binomial.py", "controls.py")
    }
    _json(root / "code-identity.json", code_identity)
    reports, errors = {}, []
    statistics = _stats(manifest)
    try:
        for index, case in enumerate(manifest["cases"]):
            primary = reference(case)
            certificate = primary.certificate(statistics["reference_abs_error_max"])
            secondary = reference_b(case)
            if set(primary.comparisons) != set(secondary):
                raise ValueError("oracle comparison coverage mismatch")
            for name, a in primary.comparisons.items():
                b = secondary[name]
                if max(a.lo, b.lo) > min(a.hi, b.hi):
                    raise ValueError("independent reference intervals do not overlap")
            report = _save_table(root, f"correct_{index:02d}", case, primary.semantic, statistics)
            _json(root / f"correct_{index:02d}" / "oracle-certificate.json", certificate)
            reports[f"correct_{index:02d}"] = report["status"]
            if report["status"] != "accepted":
                errors.append(f"correct_{index:02d} did not accept")
        coherent = _find(manifest, "coherent_product", [0.4, 0.4, 0.4])
        mutant = deepcopy(coherent)
        mutant["eta"] = [0.8, 0.8, 0.8]
        result = _save_table(
            root, "wrong_efficiency", coherent, reference(mutant).semantic, statistics
        )
        reports["wrong_efficiency"] = result["status"]
        if result["status"] != "defect":
            errors.append("wrong efficiency not flagged")
        for tested_path in ("gaussian_direct", "local_engine_gaussian"):
            for tested_detector in ("pnr", "threshold"):
                if (tested_path, tested_detector) == ("gaussian_direct", "pnr"):
                    continue
                control = _find(
                    manifest,
                    "coherent_product",
                    [0.4, 0.4, 0.4],
                    detector=tested_detector,
                    path=tested_path,
                )
                wrong = deepcopy(control)
                wrong["eta"] = [0.8, 0.8, 0.8]
                label = "wrong_efficiency_" + tested_path + "_" + tested_detector
                report = _save_table(root, label, control, reference(wrong).semantic, statistics)
                reports[label] = report["status"]
                if report["status"] != "defect":
                    errors.append(label + " not flagged")
                control = _find(
                    manifest,
                    "coherent_product",
                    [1, 1, 1],
                    detector=tested_detector,
                    path=tested_path,
                )
                swapped_table = {
                    tuple(reversed(row)): mass for row, mass in reference(control).semantic.items()
                }
                label = "mode_swap_" + tested_path + "_" + tested_detector
                report = _save_table(root, label, control, swapped_table, statistics)
                reports[label] = report["status"]
                if report["status"] != "defect":
                    errors.append(label + " not flagged")
        original = _find(manifest, "coherent_product", [1, 1, 1])
        swapped = {tuple(reversed(row)): mass for row, mass in reference(original).semantic.items()}
        result = _save_table(root, "mode_swap", original, swapped, statistics)
        reports["mode_swap"] = result["status"]
        if result["status"] != "defect":
            errors.append("mode swap not flagged")
        moved = dict(reference(original).semantic)
        moved[(0, 0)] -= F(6, 1000)
        moved[(5, 0)] += F(6, 1000)
        result = _save_table(root, "tail_mass", original, moved, statistics)
        reports["tail_mass"] = result["status"]
        if result.get("aggregate_tail", {}).get("status") != "defect":
            errors.append("tail mutant not flagged by tail criterion")
        weak = _find(manifest, "two_mode_squeezed", [0.4, 0.4], path="apps_sample")
        altered = deepcopy(weak)
        altered["eta"] = [0.6, 0.6]
        result = _save_table(root, "weak_graph_loss", weak, reference(altered).semantic, statistics)
        reports["weak_graph_loss"] = result["status"]
        # Frozen v3 expects probability nonacceptance without a guaranteed defect flag.
        if result["status"] != "inconclusive":
            errors.append("weak graph control differs from preregistered inconclusive status")
        threshold = _find(manifest, "coherent_product", [1, 1, 1], detector="threshold")
        # The wrong expectation is separately rejected by the independently derived law.
        law = reference_b(threshold)
        mode0_click = law["cell:0,1"] + law["cell:1,1"]
        poisson_single = interval(F(2, 5)) * (-interval(F(2, 5))).exp()
        mean_instead = interval(F(2, 5))
        wrong_reference = {
            "P_N1_instead_of_click": max(poisson_single.lo, mode0_click.lo)
            > min(poisson_single.hi, mode0_click.hi),
            "mean_instead_of_click": max(mean_instead.lo, mode0_click.lo)
            > min(mean_instead.hi, mode0_click.hi),
        }
        if not all(wrong_reference.values()):
            errors.append("wrong threshold reference not rejected")
        rejected = []
        for label, row in [
            ("count_as_threshold", (2, 0)),
            ("negative", (-1, 0)),
            ("fractional", (0.5, 0)),
            ("wrong_columns", (1,)),
        ]:
            try:
                raw_indicators(threshold, row)
            except ValueError:
                rejected.append(label)
        if len(rejected) != 4:
            errors.append("raw domain invalid control accepted")
        # Persist the nonbinary raw bytes before invoking the reader/domain checker.
        invalid_dir = root / "invalid_raw"
        invalid_dir.mkdir(mode=0o700)
        write_npy(invalid_dir / "count_as_threshold.npy", [(2, 0)] * 10000, 2)
        read_back = read_npy(invalid_dir / "count_as_threshold.npy", 10000, 2)
        try:
            raw_indicators(threshold, read_back[0])
            errors.append("invalid NPY domain accepted")
        except ValueError:
            pass
        nonconstant = [(0, 0)] * 999 + [(0, 1)]
        replay = nonconstant * 100
        replay_dir = root / "repeated_batch"
        replay_dir.mkdir(mode=0o700)
        for j in range(10):
            write_npy(replay_dir / f"{j:02d}.npy", replay[j * 10000 : (j + 1) * 10000], 2)
        period = replay_period(replay, original)
        if period != 1000:
            errors.append("specified replay not detected")
        vacuum = _find(manifest, "vacuum", [1])
        if replay_period([(0,)] * 100000, vacuum) is not None:
            errors.append("valid deterministic vacuum incorrectly rejected")
        cp_controls = {}
        for x in (0, 1, 600, 100000):
            bounds = clopper_pearson(100000, x)
            cp_controls[str(x)] = {
                k: [str(t) for t in v] if isinstance(v, tuple) else str(v)
                for k, v in bounds.items()
            }
            if bounds["endpoint_width_max"] > F(1, 10**10):
                errors.append("endpoint certificate width exceeded")
        if binomial_cdf(4, 1, F(1, 2)) != (F(5, 16), F(5, 16)):
            errors.append("small exact binomial identity failed")
        invalid_checks = {}
        for label, action in [
            ("reference_budget_breach", lambda: reference(original).certificate(0)),
            ("incomplete_rows", lambda: evaluate_counts(original, {}, 99999, statistics)),
            ("endpoint_width_invalid", lambda: clopper_pearson(100000, 10, width=F(1, 100))),
            ("endpoint_invalid_count", lambda: clopper_pearson(100000, -1)),
        ]:
            try:
                action()
                invalid_checks[label] = "unexpected_acceptance"
                errors.append(label)
            except ValueError:
                invalid_checks[label] = "rejected"
        _json(
            root / "edge-controls.json",
            {
                "wrong_threshold_reference_rejected": wrong_reference,
                "invalid_domain_rejected": rejected,
                "replay_period": period,
                "iid_proof": False,
                "exact_binomial": cp_controls,
                "invalid_checks": invalid_checks,
            },
        )
    except (
        Exception
    ) as exc:  # Preserve the failed deterministic attempt; never erase earlier files.
        errors.append(type(exc).__name__ + ": " + str(exc))
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            inventory.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    report = {
        "status": "supported" if not errors else "needs_changes",
        "errors": errors,
        "case_statuses": reports,
        "seconds": time.monotonic() - started,
        "files": inventory,
        "evidence_type": "synthetic_checker_control",
        "candidate_sampling": False,
        "full_candidate_reanalysis_positive_not_run": True,
        "remaining_coverage": [
            "independent actual collection release unavailable",
            "whole-analyser post-release malformed/missing metadata paths require reviewed fixture boundary",
        ],
        "iid_or_empirical_power_claim": False,
    }
    _json(root / "controls-receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    set_analysis_limits()
    result = run_controls(args.manifest, args.out)
    print(json.dumps({k: result[k] for k in ("status", "errors", "seconds")}))
    return 0 if result["status"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
