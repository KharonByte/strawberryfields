"""Parity controls for shared raw-case mechanics, not new scientific evidence."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import struct
import unittest

from maintenance.detector_validation.case_evidence import (
    check_worker_command,
    evaluate_rows,
    read_chunk_samples,
    worker_command,
)
from maintenance.detector_validation.fresh_profile import profile
from maintenance.detector_validation.reanalyse import evaluate_counts, parse_npy, raw_indicators


def payload(rows):
    columns = len(rows[0])
    header = repr({"descr": "<i8", "fortran_order": False, "shape": (len(rows), columns)})
    header = (header + " " * ((-len(header) - 11) % 16) + "\n").encode("ascii")
    return (
        b"\x93NUMPY\x01\x00"
        + struct.pack("<H", len(header))
        + header
        + b"".join(struct.pack("<" + "q" * columns, *row) for row in rows)
    )


class CaseEvidenceTests(unittest.TestCase):
    def test_command_bytes_match_legacy_and_fresh_construction(self):
        for case in profile("scientific")["cases"]:
            encoded = (
                json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            ).encode()
            for module in ("worker", "fresh_worker"):
                for timer in (False, True):
                    expected = [
                        "/runtime/python",
                        "-B",
                        "-m",
                        "maintenance.detector_validation." + module,
                        "--case-sha256",
                        hashlib.sha256(encoded).hexdigest(),
                        "--chunk",
                        "3",
                    ] + (["--cpu-timer"] if timer else [])
                    self.assertEqual(
                        worker_command("/runtime/python", case, 3, module=module, cpu_timer=timer),
                        expected,
                    )
                    check_worker_command(
                        expected, "/runtime/python", case, 3, module=module, cpu_timer=timer
                    )
                    with self.assertRaises(ValueError):
                        check_worker_command(
                            expected, "/runtime/python", case, 4, module=module, cpu_timer=timer
                        )

    def test_raw_engine_mapping_preserves_real_parser_and_rejections(self):
        case = next(
            c
            for c in profile("scientific")["cases"]
            if c["path"] == "local_engine_gaussian" and len(c["output_mode_order"]) == 2
        )
        secondary = [(0, 1), (2, 3)] * 5000
        mapping = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
        expected = [tuple(row[j] for j in mapping) for row in secondary]
        files = {"raw/samples.npy": payload(expected), "raw/secondary.npy": payload(secondary)}

        def read(name, count, columns):
            return parse_npy(files[name], count, columns)

        self.assertEqual(read_chunk_samples(read, "raw/", case), expected)
        files["raw/secondary.npy"] = payload([(9, 9)] * 10000)
        with self.assertRaisesRegex(ValueError, "mapping mismatch"):
            read_chunk_samples(read, "raw/", case)
        files["raw/secondary.npy"] = b"invalid NPY"
        with self.assertRaises(ValueError):
            read_chunk_samples(read, "raw/", case)

    def test_raw_domain_admission_remains_explicit(self):
        case = deepcopy(profile("scientific")["cases"][0])
        case["detector"] = "threshold"
        rows = [(2,)] * 10000
        data = payload(rows)
        read = lambda name, count, columns: parse_npy(data, count, columns)
        self.assertEqual(read_chunk_samples(read, "raw/", case), rows)
        with self.assertRaisesRegex(ValueError, "domain mismatch"):
            read_chunk_samples(read, "raw/", case, check_domain=True)

    def test_indicator_counts_match_both_previous_counting_forms(self):
        for case in profile("scientific")["cases"]:
            columns = len(case["output_mode_order"])
            rows = [tuple(0 for _ in range(columns)), tuple(1 for _ in range(columns))] * 5
            old_streamed = Counter()
            for row in rows:
                old_streamed.update(raw_indicators(case, row))
            old_comprehension = Counter(k for row in rows for k in raw_indicators(case, row))
            self.assertEqual(old_streamed, old_comprehension)

            def evaluator(actual_case, counts, n, statistics):
                self.assertIs(actual_case, case)
                self.assertEqual(n, len(rows))
                self.assertEqual(counts, old_streamed)
                self.assertEqual(statistics, {"fixture": True})
                return {"status": "fixture", "counts": counts}

            self.assertEqual(
                evaluate_rows(case, rows, {"fixture": True}, evaluator)["counts"], old_streamed
            )

    def test_real_evaluator_result_is_identical(self):
        spec = profile("scientific")
        for case in (spec["cases"][0], spec["cases"][2]):
            rows = [tuple(0 for _ in case["output_mode_order"])] * case["shots"]
            counts = Counter(k for row in rows for k in raw_indicators(case, row))
            expected = evaluate_counts(case, counts, len(rows), spec["statistics"])
            self.assertEqual(
                evaluate_rows(case, rows, spec["statistics"], evaluate_counts), expected
            )
