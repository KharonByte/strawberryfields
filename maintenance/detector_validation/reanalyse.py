"""Simulator-free raw reanalysis and a separately derived PGF reference.

The secondary oracle shares rational interval arithmetic, not primary probability
formulas or latent-sum helpers. Collection is never invoked by this module.
"""

import argparse
import ast
from collections import Counter
from fractions import Fraction as F
import hashlib
import io
from itertools import product
import json
import os
from pathlib import Path
import re
import math
import resource
import signal
import struct
from decimal import Decimal, localcontext

from .binomial import clopper_pearson

from .oracle import Interval, interval, parameter, probability, trig, confidence_radius

MAX_JSON = 4 * 1024 * 1024
MAX_RAW = 200000
MAX_FILES = 8192
MAX_TOTAL = 4 * 1024**3


def _means_b(case, binary):
    a = [tuple(parameter(v, binary) for v in z) for z in case["parameters"]["alpha_re_im"]]
    energies = [x**2 + y**2 for x, y in a]
    if case["family"] == "coherent_interference":
        p = case["parameters"]
        c = trig(parameter(p["beamsplitter_theta"], binary), True)
        s = trig(parameter(p["beamsplitter_theta"], binary))
        cp = trig(parameter(p["beamsplitter_phi"], binary), True)
        sp = trig(parameter(p["beamsplitter_phi"], binary))
        (ar, ai), (br, bi) = a
        cross = (ar * br + ai * bi) * cp + (ar * bi - ai * br) * sp
        first = c**2 * energies[0] + s**2 * energies[1] - 2 * c * s * cross
        second = s**2 * energies[0] + c**2 * energies[1] + 2 * c * s * cross
        energies = [first, second]
    return [parameter(e, binary) * v for e, v in zip(case["eta"], energies)]


def _single_pgf(case, binary):
    family, p = case["family"], case["parameters"]
    e = parameter(case["eta"][0], binary)
    if family == "vacuum":
        return [interval(1)] + [interval(0)] * 5, interval(0)
    if family == "thermal":
        m = parameter(p["mean_photons"], binary) * e
        low = [1 / (1 + m)]
        for _ in range(4):
            low.append(low[-1] * m / (1 + m))
        return low + [probability(1 - sum(low, interval(0)))], None
    t = parameter(p["tanh_r_squared"], binary)
    A, B, C = 1 - t * (1 - e) ** 2, -2 * t * e * (1 - e), -t * e**2
    low = [((1 - t) / A).sqrt()]
    for n in range(1, 5):
        earlier = low[n - 2] if n >= 2 else interval(0)
        low.append(
            probability(-(B * (2 * n - 1) * low[n - 1] + 2 * C * (n - 1) * earlier) / (2 * A * n))
        )
    odd = probability((1 - ((1 - t) / (1 - t * (1 - 2 * e) ** 2)).sqrt()) / 2)
    if e.lo == e.hi == 1:
        low[1] = low[3] = odd = interval(0)
    if e.hi == 0:
        low = [interval(1)] + [interval(0)] * 4
        odd = interval(0)
    return low + [probability(1 - sum(low, interval(0)))], odd


def reference_b(case, binary=False):
    """Independently compute cells using PGF recurrences and marginal complements."""
    family = case["family"]
    detector = case["detector"]
    order = case["output_mode_order"]
    events = {}
    if family in ("vacuum", "thermal", "squeezed_vacuum"):
        values, odd = _single_pgf(case, binary)
        table = {(i,): v for i, v in enumerate(values)}
        if "odd_total_count" in case["additional_events"]:
            events["event:odd_total_count"] = odd
    elif family.startswith("coherent"):
        means = _means_b(case, binary)
        margins = []
        for mode in order:
            m = means[mode]
            low = [(-m).exp()]
            for n in range(1, 5):
                low.append(low[-1] * m / n)
            margins.append(low + [probability(1 - sum(low, interval(0)))])
        table = {
            key: margins[0][key[0]] * margins[1][key[1]] for key in product(range(6), repeat=2)
        }
    elif family == "two_mode_squeezed":
        q = parameter(case["parameters"]["q"], binary)
        e, f = [parameter(case["eta"][j], binary) for j in order]
        A, B, C, D = 1 - q * (1 - e) * (1 - f), -q * e * (1 - f), -q * f * (1 - e), -q * e * f
        table = {}
        for i in range(5):
            for j in range(5):
                if i == j == 0:
                    table[i, j] = (1 - q) / A
                else:
                    table[i, j] = probability(
                        -(
                            B * table.get((i - 1, j), interval(0))
                            + C * table.get((i, j - 1), interval(0))
                            + D * table.get((i - 1, j - 1), interval(0))
                        )
                        / A
                    )
        margins = []
        for loss in (e, f):
            m = loss * q / (1 - q)
            vals = [1 / (1 + m)]
            for _ in range(4):
                vals.append(vals[-1] * m / (1 + m))
            margins.append(vals)
        for i in range(5):
            table[i, 5] = probability(
                margins[0][i] - sum((table[i, j] for j in range(5)), interval(0))
            )
        for j in range(5):
            table[5, j] = probability(
                margins[1][j] - sum((table[i, j] for i in range(5)), interval(0))
            )
        table[5, 5] = probability(1 - sum(table.values(), interval(0)))
        if "unequal_counts" in case["additional_events"]:
            if case["eta"] != [1, 1]:
                raise ValueError("unsupported equality event outside loss-free pair")
            events["event:unequal_counts"] = interval(0)
    else:
        raise ValueError("unsupported reference family")
    if detector == "threshold":
        # Zero/nonzero conversion is of reference probabilities only, never raw output data.
        collapsed = {k: interval(0) for k in product(range(2), repeat=len(order))}
        for row, value in table.items():
            collapsed[tuple(int(v > 0) for v in row)] += value
        table = collapsed
    result = {"cell:" + ",".join(map(str, row)): probability(v) for row, v in table.items()}
    result.update(events)
    return result


def raw_indicators(case, row):
    if len(row) != len(case["output_mode_order"]):
        raise ValueError("raw column count mismatch")
    if any(type(v) is not int or v < 0 or v > 2**63 - 1 for v in row):
        raise ValueError("invalid raw photon count")
    if case["detector"] == "threshold" and any(v not in (0, 1) for v in row):
        raise ValueError("nonbinary raw threshold output")
    cell = [min(v, 5) for v in row] if case["detector"] == "pnr" else list(row)
    names = ["cell:" + ",".join(map(str, cell))]
    if "odd_total_count" in case["additional_events"] and sum(row) % 2:
        names.append("event:odd_total_count")
    if "unequal_counts" in case["additional_events"] and len(set(row)) != 1:
        names.append("event:unequal_counts")
    return names


def structurally_valid(case, row):
    if case["family"] == "vacuum" and any(row):
        return False
    if any(case["eta"][m] == 0 and row[j] != 0 for j, m in enumerate(case["output_mode_order"])):
        return False
    if case["detector"] == "pnr":
        if case["family"] == "squeezed_vacuum" and case["eta"] == [1] and row[0] % 2:
            return False
        if (
            case["family"] == "two_mode_squeezed"
            and all(e == 1 for e in case["eta"])
            and row[0] != row[1]
        ):
            return False
    return True


def replay_period(rows, case):
    """A finite replay diagnostic, not a proof of iid or of an implementation defect."""
    if case["family"] == "vacuum" or all(case["eta"][m] == 0 for m in case["output_mode_order"]):
        return None
    for period in (1, 10, 100, 1000, 10000):
        if len(rows) >= 2 * period and len(rows) % period == 0:
            if all(rows[j] == rows[j % period] for j in range(period, len(rows))):
                return period
    return None


def evaluate_counts(case, counts, N, statistics):
    """Probability rules only; caller separately enforces raw/provenance/operational gates."""
    if N != case["shots"]:
        raise ValueError("incomplete sample budget")
    nominal, binary = reference_b(case), reference_b(case, True)
    K = statistics["comparison_count"]
    e = confidence_radius(K, N, statistics.get("alpha_cell", statistics["alpha_familywise"]))
    b = F(str(statistics["reference_abs_error_max"]))
    entries = {}
    status = "accepted"
    for name, value in nominal.items():
        if (
            name.startswith("cell:")
            and sum(v for k, v in counts.items() if k.startswith("cell:")) != N
        ):
            raise ValueError("cell counts do not total N")
        error = max(abs(binary[name].lo - value.midpoint), abs(binary[name].hi - value.midpoint))
        if error > b:
            raise ValueError("secondary oracle error certificate exceeds b")
        # Optional independently frozen per-comparison targets; no data-chosen fallback.
        delta = F(
            str(
                statistics.get("comparison_targets", {}).get(
                    case["id"] + "/" + name, statistics["delta_absolute_per_cell"]
                )
            )
        )
        phat = F(counts.get(name, 0), N)
        discrepancy = abs(phat - value.midpoint)
        upper, lower = discrepancy + e + b, max(F(0), discrepancy - e - b)
        verdict = "accepted" if upper <= delta else "defect" if lower > delta else "inconclusive"
        if verdict == "defect" or (verdict == "inconclusive" and status == "accepted"):
            status = verdict
        entries[name] = {
            "count": counts.get(name, 0),
            "N": N,
            "reference": value.outward_strings(),
            "binary_parameter_envelope": binary[name].outward_strings(),
            "error_certificate_upper": float(error),
            "delta": float(delta),
            "phat": float(phat),
            "epsilon_upper": float(e),
            "error_upper": float(upper),
            "error_lower": float(lower),
            "status": verdict,
        }
    if len(entries) != case["comparison_count"]:
        raise ValueError("comparison coverage mismatch")
    result = {"status": status, "scope": "probability_rules_only", "comparisons": entries}
    if statistics.get("acceptance_version") == 3:
        joint, tail = evaluate_joint_tail(case, counts, N, statistics, nominal, binary)
        result["joint_tv"] = joint
        if tail is not None:
            result["aggregate_tail"] = tail
        verdicts = [result["status"], joint["status"]] + ([tail["status"]] if tail else [])
        result["status"] = (
            "defect"
            if "defect" in verdicts
            else "accepted" if set(verdicts) == {"accepted"} else "inconclusive"
        )
    return result


def evaluate_joint_tail(case, counts, N, statistics, nominal, binary):
    """v3 joint-TV and aggregate overflow criteria; thresholds remain frozen inputs."""
    names = [k for k in nominal if k.startswith("cell:")]
    m = len(names)
    b = F(str(statistics["reference_abs_error_max"]))
    with localcontext() as ctx:
        ctx.prec = 60
        logarg = Decimal(statistics["joint_case_count"]) / Decimal(str(statistics["alpha_joint"]))
        rad = ((Decimal(m) * Decimal(2).ln() + logarg.ln()) / Decimal(2 * N)).sqrt()
        r = F(rad) + F(1, 10**50)
    distance = sum((abs(F(counts.get(k, 0), N) - nominal[k].midpoint) for k in names), F(0)) / 2
    reference_allowance = m * b / 2
    upper = distance + r + reference_allowance
    lower = max(F(0), distance - r - reference_allowance)
    target = F(str(statistics["delta_joint_tv"]))
    status = "accepted" if upper <= target else "defect" if lower > target else "inconclusive"
    joint = {
        "status": status,
        "cells": m,
        "distance": float(distance),
        "radius_upper": float(r),
        "error_upper": float(upper),
        "error_lower": float(lower),
        "target": float(target),
        "full_binary_law": case["detector"] == "threshold",
    }
    if case["detector"] != "pnr":
        return joint, None
    tail_names = [k for k in names if 5 in [int(v) for v in k[5:].split(",")]]
    ref = sum((nominal[k] for k in tail_names), interval(0))
    envelope = sum((binary[k] for k in tail_names), interval(0))
    error = max(abs(envelope.lo - ref.midpoint), abs(envelope.hi - ref.midpoint))
    if error > b:
        raise ValueError("aggregate-tail reference error exceeds its separate b budget")
    if envelope.hi > F(str(statistics["ideal_tail_upper_max"])):
        raise ValueError("ideal tail bound exceeds frozen full-law lifting budget")
    x = sum(counts.get(k, 0) for k in tail_names)
    a = F(str(statistics["alpha_tail"])) / (2 * statistics["tail_case_count"])
    cp = clopper_pearson(N, x, a)
    L, U = cp["lower_outward"], cp["upper_outward"]
    maximum = max(abs(L - ref.midpoint), abs(U - ref.midpoint)) + b
    minimum = max(L - ref.midpoint - b, ref.midpoint - U - b, F(0))
    delta = F(str(statistics["delta_tail_absolute"]))
    tail_status = (
        "accepted" if maximum <= delta else "defect" if minimum > delta else "inconclusive"
    )
    tail = {
        "status": tail_status,
        "count": x,
        "N": N,
        "reference": ref.outward_strings(),
        "reference_error_upper": float(error),
        "lower_outward": Interval(L, L).outward_strings()["lower"],
        "upper_outward": Interval(U, U).outward_strings()["upper"],
        "lower_endpoint_bracket": [str(v) for v in cp["lower_bracket"]],
        "upper_endpoint_bracket": [str(v) for v in cp["upper_bracket"]],
        "endpoint_width_max": float(cp["endpoint_width_max"]),
        "method": cp["method"],
        "error_upper": float(maximum),
        "error_lower": float(minimum),
        "target": float(delta),
    }
    joint["ideal_aggregate_tail_upper"] = envelope.outward_strings()["upper"]
    joint["full_pnr_tv_upper_if_accepted"] = float(target + envelope.hi)
    return joint, tail


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def read_json(path):
    with secure_open(path) as stream:
        return parse_json(stream.read(MAX_JSON + 1))


def parse_json(payload):
    if len(payload) > MAX_JSON:
        raise ValueError("JSON size limit")
    value = json.loads(
        payload,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )

    def check(obj, depth=0):
        if depth > 30:
            raise ValueError("JSON depth limit")
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ValueError("nonfinite JSON number")
        if isinstance(obj, dict):
            for item in obj.values():
                check(item, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                check(item, depth + 1)

    check(value)
    return value


def secure_open(path):
    path = Path(path)
    for parent in [path, *path.parents]:
        if parent.is_symlink():
            raise ValueError("symlink path forbidden")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    import stat

    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("not a regular file")
    return os.fdopen(fd, "rb")


def digest(path):
    hasher, size = hashlib.sha256(), 0
    with secure_open(path) as stream:
        while data := stream.read(1024 * 1024):
            size += len(data)
            if size > MAX_TOTAL:
                raise ValueError("file exceeds budget")
            hasher.update(data)
    return size, hasher.hexdigest()


def parse_npy(payload, expected_rows, expected_columns):
    """Bounded numeric NPY parser; never extracts or loads pickle/object data."""
    if expected_rows != 10000 or expected_columns not in (1, 2):
        raise ValueError("shape outside frozen chunk bounds")
    if len(payload) > MAX_RAW + 1034:
        raise ValueError("NPY member exceeds bounded size")
    stream = io.BytesIO(payload)
    if stream.read(8) != b"\x93NUMPY\x01\x00":
        raise ValueError("requires NPY version1")
    length_bytes = stream.read(2)
    if len(length_bytes) != 2:
        raise ValueError("truncated NPY header")
    length = struct.unpack("<H", length_bytes)[0]
    if length > 1024:
        raise ValueError("NPY header size limit")
    header = ast.literal_eval(stream.read(length).decode("ascii").strip())
    if header != {
        "descr": "<i8",
        "fortran_order": False,
        "shape": (expected_rows, expected_columns),
    }:
        raise ValueError("NPY header/dtype/shape mismatch")
    body = stream.read()
    if len(body) != expected_rows * expected_columns * 8:
        raise ValueError("NPY payload length mismatch")
    return list(struct.iter_unpack("<" + "q" * expected_columns, body))


def read_npy(path, expected_rows, expected_columns):
    with secure_open(path) as stream:
        return parse_npy(stream.read(MAX_RAW + 1035), expected_rows, expected_columns)


def member_name(name):
    if not isinstance(name, str) or not name or len(name) > 512:
        raise ValueError("invalid evidence member name")
    if (
        name.startswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError("unsafe evidence member")
    return name


class DirectoryEvidence:
    """Bounded read-only adapter retained for explicitly local synthetic fixtures."""

    def __init__(self, root):
        self.root = Path(root)

    def read_bytes(self, name, max_size):
        with secure_open(self.root / member_name(name)) as stream:
            data = stream.read(max_size + 1)
        if len(data) > max_size:
            raise ValueError("evidence member size bound")
        return data

    def inventory(self):
        result = {}
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in dirs + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("symlink evidence member")
            for name in files:
                path = Path(directory) / name
                member = path.relative_to(self.root).as_posix()
                size, sha = digest(path)
                result[member_name(member)] = {"bytes": size, "sha256": sha}
                if len(result) > MAX_FILES:
                    raise ValueError("evidence member count bound")
        return result


class Evidence:
    """Reader protocol used for container-backed and directory-backed evidence."""

    def __init__(self, reader):
        self.reader = reader
        self.members = reader.inventory()
        if not isinstance(self.members, dict) or not 1 <= len(self.members) <= MAX_FILES:
            raise ValueError("invalid reader inventory")
        total = 0
        for name, item in self.members.items():
            member_name(name)
            if type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= MAX_TOTAL:
                raise ValueError("invalid reader member size")
            if not isinstance(item.get("sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", item["sha256"]
            ):
                raise ValueError("invalid reader member digest")
            total += item["bytes"]
        if total > MAX_TOTAL:
            raise ValueError("reader aggregate byte budget")

    def data(self, name, limit=MAX_JSON):
        member_name(name)
        if name not in self.members:
            raise ValueError("missing evidence member: " + name)
        meta = self.members[name]
        if meta["bytes"] > limit:
            raise ValueError("evidence member exceeds type budget")
        data = self.reader.read_bytes(name, limit)
        if not isinstance(data, bytes) or len(data) != meta["bytes"]:
            raise ValueError("reader member byte count mismatch")
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError("reader member digest mismatch")
        return data

    def json(self, name):
        return parse_json(self.data(name))

    def hash(self, name):
        data = self.data(name)
        return hashlib.sha256(data).hexdigest()

    def npy(self, name, rows, columns):
        return parse_npy(self.data(name, MAX_RAW + 1034), rows, columns)


def analyse_run(
    manifest_path, receipt_path, raw_root=None, trusted_release_sha256=None, evidence_reader=None
):
    """Production entry: rejects synthetic fixtures even with a matching release digest."""
    return _analyse_run(
        manifest_path,
        receipt_path,
        raw_root,
        trusted_release_sha256,
        evidence_reader,
        synthetic=False,
    )


def analyse_synthetic_fixture(reader, trusted_fixture_digest):
    """Exercise the full pipeline without representing its review fixture as authorization."""
    return _analyse_run(
        "manifest.json", "collection.json", None, trusted_fixture_digest, reader, synthetic=True
    )


def check_case_budget(observed, first):
    cpu, wall = (840, 1140) if first else (900, 1200)
    if observed.get("cpu_cap_seconds") != cpu or observed.get("wall_cap_seconds") != wall:
        raise ValueError("case resource scope mismatch")
    for key in ("all_child_cpu_seconds", "controller_cpu_seconds", "elapsed_wall_seconds"):
        value = observed.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("missing or invalid case resource observation")
    if (
        observed["all_child_cpu_seconds"] + observed["controller_cpu_seconds"] > cpu
        or observed["elapsed_wall_seconds"] > wall
    ):
        raise ValueError("observed case resource ceiling exceeded")


def check_resource_evidence(caps, requested_cpu):
    """Validate actual reported numeric limits; their enforcement remains a reviewed claim."""
    if type(requested_cpu) is not int or not 1 <= requested_cpu <= 120:
        raise ValueError("invalid requested CPU limit")
    maxima = {
        "address_space": 14 * 1024**3,
        "cpu_seconds": requested_cpu,
        "file_size": 0,
        "core_size": 0,
        "open_files": 64,
    }
    if not isinstance(caps, dict) or set(caps) != set(maxima):
        raise ValueError("missing or unexpected resource cap fields")
    for name, maximum in maxima.items():
        pair = caps[name]
        if not isinstance(pair, dict) or set(pair) != {"soft", "hard"}:
            raise ValueError("malformed resource cap pair")
        soft, hard = pair["soft"], pair["hard"]
        if type(soft) is not int or type(hard) is not int:
            raise ValueError("resource cap must be an integer")
        if maximum == 0:
            valid = soft == hard == 0
        else:
            valid = 0 < soft <= hard <= maximum
        if not valid:
            raise ValueError("missing or excessive resource cap: " + name)


def _analyse_run(
    manifest_path, receipt_path, raw_root, trusted_release_sha256, evidence_reader, synthetic
):
    if evidence_reader is None:
        root = Path(receipt_path).parent
        if Path(manifest_path).absolute() != (root / "manifest.json").absolute():
            raise ValueError("directory evidence requires its preserved manifest.json")
        if Path(raw_root).resolve() != (root / "raw").resolve():
            raise ValueError("raw root must be collection receipt sibling")
        evidence_reader = DirectoryEvidence(root)
    else:
        if manifest_path != "manifest.json" or receipt_path != "collection.json":
            raise ValueError("container evidence uses fixed virtual root names")
    evidence = Evidence(evidence_reader)
    if hasattr(evidence_reader, "state") and evidence_reader.state.get("status") != "complete":
        raise ValueError("container checkpoint is not complete")
    manifest, receipt = evidence.json("manifest.json"), evidence.json("collection.json")
    expected_type = "synthetic_reanalysis_fixture" if synthetic else "candidate_terminal_samples"
    frozen_design = read_json(Path(__file__).with_name("cases.json"))
    if (
        manifest.get("cases") != frozen_design["cases"]
        or manifest.get("statistics") != frozen_design["statistics"]
    ):
        raise ValueError(
            "case matrix/statistics differ from the independently reviewed packaged design"
        )
    if not trusted_release_sha256 or not re.fullmatch(r"[0-9a-f]{64}", trusted_release_sha256):
        raise ValueError(
            "an independently authenticated release digest must be supplied explicitly"
        )
    if (
        evidence.hash("release.json") != trusted_release_sha256
        or receipt.get("release_sha256") != trusted_release_sha256
    ):
        raise ValueError("release byte identity mismatch")
    release = evidence.json("release.json")
    if synthetic:
        if (
            release.get("synthetic_fixture_only") is not True
            or release.get("collection_authorized") is not False
        ):
            raise ValueError("synthetic review fixture cannot authorize collection")
    elif release.get("synthetic_fixture_only") or receipt.get("synthetic_fixture_only"):
        raise ValueError("synthetic review fixture is not production evidence")
    if (not synthetic and release.get("collection_authorized") is not True) or release.get(
        "manifest_sha256"
    ) != evidence.hash("manifest.json"):
        raise ValueError("collection release does not bind this manifest")
    if release.get("source") != receipt.get("source_pre") or release.get("runtime") != receipt.get(
        "runtime_pre"
    ):
        raise ValueError("release source/runtime identity mismatch")
    reviewers = release.get("reviewers", {})
    if set(reviewers) != {"qa", "scientific", "security"}:
        raise ValueError("missing independent reviewer bindings")
    tasks = [r.get("task") for r in reviewers.values()]
    if len(set(tasks)) != 3 or not all(tasks) or "/root/detector_validator" in tasks:
        raise ValueError("reviewers not independently identified")
    for record in reviewers.values():
        if synthetic:
            if not record["task"].startswith("synthetic-fixture/") or not record["path"].startswith(
                "fixture-reviews/"
            ):
                raise ValueError("fixture review must be explicitly synthetic")
            review_hash = evidence.hash(record["path"])
        else:
            if record["task"].startswith("synthetic-fixture/"):
                raise ValueError("synthetic reviewer cannot authorize production")
            review_hash = digest(record["path"])[1]
        if review_hash != record["sha256"]:
            raise ValueError("review bytes changed")
    if not synthetic:
        from .budget import ATTEMPT_CAPACITY, validate_ledger, SETTINGS

        ledger = release.get("cumulative_ledger")
        validate_ledger(ledger, release_sha256=trusted_release_sha256)
        if (
            receipt.get("cumulative_ledger") != ledger
            or manifest.get("renewed_resource_settings") != SETTINGS
        ):
            raise ValueError("renewed cumulative ledger binding mismatch")
        if receipt.get("storage", {}).get("capacity") != ATTEMPT_CAPACITY:
            raise ValueError("renewed attempt storage capacity mismatch")
        if getattr(evidence.reader, "capacity", None) != ATTEMPT_CAPACITY:
            raise ValueError("renewed production requires fixed1GiB container")
    if not release.get("aggregate_disk_enforcement"):
        raise ValueError("missing reviewed aggregate storage enforcement")
    if (
        receipt.get("schema") != "detector-collection-v1"
        or receipt.get("evidence_type") != expected_type
    ):
        raise ValueError("not candidate evidence")
    if receipt.get("status") != "complete" or not manifest.get("operational_acceptance_frozen"):
        raise ValueError("collection/acceptance not complete or frozen")
    if evidence.hash("manifest.json") != receipt.get("manifest_sha256"):
        raise ValueError("manifest digest mismatch")
    for key in ("source", "runtime"):
        if not isinstance(receipt.get(key + "_pre"), dict) or not receipt[key + "_pre"]:
            raise ValueError("missing identity provenance")
        if receipt[key + "_pre"] != receipt.get(key + "_post"):
            raise ValueError("identity changed during collection")
    if receipt["source_pre"].get("simulator_commit") != manifest["candidate"]:
        raise ValueError("candidate identity mismatch")
    listed = receipt.get("files", [])
    if not 1 <= len(listed) <= MAX_FILES:
        raise ValueError("invalid evidence inventory size")
    inventory, total = {}, 0
    for item in listed:
        name = item["path"]
        parts = Path(name).parts
        if (
            not parts
            or Path(name).is_absolute()
            or any(p in ("..", ".") for p in parts)
            or "\\" in name
        ):
            raise ValueError("unsafe evidence member")
        if name in inventory:
            raise ValueError("duplicate evidence member")
        if type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_TOTAL:
            raise ValueError("invalid evidence size")
        total += item["bytes"]
        if total > MAX_TOTAL:
            raise ValueError("total evidence budget")
        if name not in evidence.members or evidence.members[name] != {
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }:
            raise ValueError("evidence inventory digest mismatch")
        # Independently verify every stored payload; the container index is not an oracle.
        limit = MAX_RAW + 1034 if name.endswith(".npy") else MAX_JSON
        evidence.data(name, limit)
        inventory[name] = item
    if set(evidence.members) != set(inventory) | {"collection.json"}:
        raise ValueError("unlisted evidence member or incomplete inventory")
    thread_names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    environment = evidence.json("environment.json")
    expected_threads = {name: "1" for name in thread_names}
    if (
        any(environment.get(name) != "1" for name in thread_names)
        or environment.get("CUDA_VISIBLE_DEVICES") != "-1"
    ):
        raise ValueError("thread/GPU environment evidence mismatch")
    if any(environment.get(name) != "0" for name in ("NUMBA_ENABLE_PROFILING", "NUMBA_DEBUGINFO")):
        raise ValueError("Numba profiler/debug environment evidence mismatch")
    if (
        not isinstance(receipt["runtime_pre"].get("invocation_executable"), str)
        or not receipt["runtime_pre"]["invocation_executable"]
    ):
        raise ValueError("missing frozen invocation executable")
    if receipt["runtime_pre"].get("threads") != expected_threads:
        raise ValueError("runtime thread identity mismatch")
    cases = manifest["cases"]
    if len(cases) != 68 or manifest["case_count"] != 68 or len({c["id"] for c in cases}) != 68:
        raise ValueError("case coverage count")
    if sum(c["comparison_count"] for c in cases) != manifest["comparison_count"]:
        raise ValueError("comparison family mismatch")
    observed_cases = {c["id"]: c for c in receipt["cases"]}
    if len(observed_cases) != len(receipt["cases"]) or set(observed_cases) != {
        c["id"] for c in cases
    }:
        raise ValueError("missing/duplicate/extra case")
    statistics = dict(manifest["statistics"], comparison_count=manifest["comparison_count"])
    reports, expected_raw, all_seeds = {}, set(), set()
    for case in cases:
        cid = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*", cid):
            raise ValueError("unsafe case id")
        if case["chunk_count"] != 10 or case["chunk_rows"] != 10000 or case["shots"] != 100000:
            raise ValueError("chunk budget changed")
        if (
            len(case["chunk_seeds"]) != 10
            or len(set(case["chunk_seeds"])) != 10
            or any(
                type(s) is not int or not 0 <= s < 2**32 or s in all_seeds
                for s in case["chunk_seeds"]
            )
        ):
            raise ValueError("duplicate/missing seed")
        all_seeds.update(case["chunk_seeds"])
        observed = observed_cases[cid]
        if not synthetic:
            check_case_budget(observed, cid == cases[0]["id"])
        if observed.get("status") != "complete" or len(observed["chunks"]) != 10:
            raise ValueError("incomplete case")
        chunks = {c["index"]: c for c in observed["chunks"]}
        if set(chunks) != set(range(10)):
            raise ValueError("missing/duplicate chunk")
        rows, counts = [], Counter()
        for index, seed in enumerate(case["chunk_seeds"]):
            chunk = chunks[index]
            if (
                "limit_failure" not in chunk
                or chunk["limit_failure"] is not None
                or any(
                    chunk.get(k) != v
                    for k, v in {"seed": seed, "rows": 10000, "exit_code": 0}.items()
                )
            ):
                raise ValueError("chunk execution mismatch")
            requested_cpu = chunk.get("requested_cpu_limit")
            if type(requested_cpu) is not int or not 1 <= requested_cpu <= 120:
                raise ValueError("invalid requested CPU limit")
            command_prefix = f"commands/{cid}/{index:02d}/"
            execution = evidence.json(command_prefix + "exit.json")
            for key in ("requested_cpu_limit", "exit_code", "limit_failure"):
                if execution.get(key) != chunk.get(key):
                    raise ValueError("chunk exit resource binding mismatch")
            case_bytes = (
                json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            expected_command = [
                receipt["runtime_pre"].get("invocation_executable"),
                "-B",
                "-m",
                "maintenance.detector_validation.worker",
                "--case-sha256",
                hashlib.sha256(case_bytes).hexdigest(),
                "--chunk",
                str(index),
            ]
            if evidence.json(command_prefix + "command.json") != expected_command:
                raise ValueError("worker command/case binding mismatch")
            prefix = f"raw/{cid}/{index:02d}/"
            names = ["samples.npy", "metadata.json"]
            if case["path"] == "local_engine_gaussian":
                names.append("secondary.npy")
            for name in names:
                member = prefix + name
                expected_raw.add(member)
                if member not in inventory:
                    raise ValueError("missing raw member")
            metadata = evidence.json(prefix + "metadata.json")
            required = {
                "case_id": cid,
                "chunk_index": index,
                "seed": seed,
                "rows": 10000,
                "output_mode_order": case["output_mode_order"],
                "evidence_type": expected_type,
                "status": "captured_not_accepted",
                "case": case,
                "requested_mode_order": case["measured_modes"],
            }
            if any(metadata.get(k) != v for k, v in required.items()):
                raise ValueError("chunk metadata mismatch")
            if not metadata.get("imports") or metadata.get("imports") != metadata.get(
                "imports_after"
            ):
                raise ValueError("worker import identity missing or changed")
            check_resource_evidence(metadata.get("verified_resource_limits"), requested_cpu)
            policy = metadata.get("cache_policy", {})
            if (
                metadata.get("guard_policy_validated") is not True
                or metadata.get("python_write_or_spawn_access_granted") is not False
                or policy.get("jit_disabled") is not False
                or policy.get("compiler_target_options_changed") is not False
                or policy.get("all_intercepted_caches_null") is not True
            ):
                raise ValueError("guard/JIT/cache policy evidence mismatch")
            if not metadata.get("rng"):
                raise ValueError("worker RNG provenance missing")
            if case["path"] == "gaussian_direct":
                if not metadata.get("direct_call_sequence") or not metadata.get(
                    "actual_pre_detector_reduced_moments"
                ):
                    raise ValueError("direct detector execution provenance missing")
            elif not metadata.get("original_circuit") or not metadata.get("compiled_circuit"):
                raise ValueError("actual circuit provenance missing")
            block = evidence.npy(prefix + "samples.npy", 10000, len(case["output_mode_order"]))
            if case["path"] == "local_engine_gaussian":
                secondary = evidence.npy(
                    prefix + "secondary.npy", 10000, len(case["measured_modes"])
                )
                mapping = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
                if any(tuple(s[j] for j in mapping) != r for s, r in zip(secondary, block)):
                    raise ValueError("Engine samples_dict mapping mismatch")
            for row in block:
                counts.update(raw_indicators(case, row))
                if not structurally_valid(case, row):
                    raise ValueError("exact physical support violated")
            rows.extend(block)
        replay = replay_period(rows, case)
        result = evaluate_counts(case, counts, len(rows), statistics)
        if replay is not None:
            result["status"] = "insufficient_independence"
            result["replay_period"] = replay
        reports[cid] = result
    actual_raw = {name for name in evidence.members if name.startswith("raw/")}
    if actual_raw != expected_raw or {n for n in inventory if n.startswith("raw/")} != expected_raw:
        raise ValueError("extra or missing raw members")
    status = (
        "accepted" if all(r["status"] == "accepted" for r in reports.values()) else "not_accepted"
    )
    return {
        "schema": "detector-reanalysis-v1",
        "status": "synthetic_fixture_checked" if synthetic else status,
        "probability_status": status,
        "synthetic_fixture_only": synthetic,
        "collection_authorized": False,
        "cases": reports,
        "claim": "bounded cell/event probability rules conditional on iid innovations",
        "operational_review_required": True,
        "display_numbers_are_rounded_not_certificates": True,
        "limits": [
            "identity equality does not authenticate approval or enforce resource limits",
            "no physical hardware, conditional-state or general detector certification",
        ],
    }


def exclusive_output(path):
    target = Path(path).absolute()
    for part in [target, *target.parents]:
        if part.is_symlink():
            raise ValueError("symlink output path")
    target.mkdir(mode=0o700, parents=False, exist_ok=False)
    return target


def set_analysis_limits():
    """No candidate subprocesses; cap this deterministic process before analysis."""
    resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    _, inherited_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    file_cap = 256 * 1024**2
    if inherited_hard != resource.RLIM_INFINITY:
        file_cap = min(file_cap, inherited_hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_cap, file_cap))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("analysis wall cap"))
    )
    signal.alarm(900)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container")
    parser.add_argument("--manifest")
    parser.add_argument("--receipt")
    parser.add_argument("--raw")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--release-sha256",
        required=True,
        help="Independently authenticated digest; do not derive from untrusted receipt",
    )
    args = parser.parse_args()
    set_analysis_limits()
    output = exclusive_output(args.out)
    reader = None
    try:
        if args.container:
            if any((args.manifest, args.receipt, args.raw)):
                raise ValueError("container mode does not accept directory paths")
            from .storage import StoreReader

            reader = StoreReader(args.container)
            report = analyse_run(
                "manifest.json",
                "collection.json",
                None,
                args.release_sha256,
                evidence_reader=reader,
            )
        else:
            if not all((args.manifest, args.receipt, args.raw)):
                raise ValueError("container or complete directory paths required")
            report = analyse_run(args.manifest, args.receipt, args.raw, args.release_sha256)
    except (ValueError, OSError, KeyError, TypeError, SyntaxError, struct.error) as exc:
        report = {"status": "incomplete_or_invalid", "reason": type(exc).__name__ + ": " + str(exc)}
    finally:
        if reader is not None:
            reader.close()
    payload = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if len(payload) > 16 * 1024**2:
        raise ValueError("bounded reanalysis report exceeds 16MiB")
    with (output / "report.json").open("xb") as stream:
        stream.write(payload)
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
