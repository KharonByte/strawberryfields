"""No-sampling transport tests. Synthetic evidence never authorizes collection."""

import copy
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from maintenance.detector_validation import ci, ci_collect, ci_gate
from maintenance.detector_validation.fresh import save
from maintenance.detector_validation.fresh_profile import profile
from maintenance.detector_validation.provenance import digest
from maintenance.detector_validation.storage import StoreWriter, encoded, sha


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_complete_partition_and_original_family(self):
        cases = [c for i in range(17) for c in ci.cases_for(i)]
        self.assertEqual(cases, profile("scientific")["cases"])
        self.assertEqual(sum(c["shots"] for c in cases), 6800000)
        self.assertEqual(sum(c["comparison_count"] for c in cases), 920)
        self.assertEqual(len({s for c in cases for s in c["chunk_seeds"]}), 680)
        self.assertEqual(sum(c["detector"] == "pnr" for c in cases), 34)
        for index in (-1, 17, True, "0"):
            with self.assertRaises(ValueError):
                ci.cases_for(index)

    def test_reruns_fail(self):
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            with self.assertRaisesRegex(ValueError, "reruns forbidden"):
                ci.context("a" * 64, "base")

    def test_dispatch_history_fails_closed(self):
        sha256 = "a" * 64
        current = {"id": 123, "display_title": "detector-" + sha256}
        env = {
            "GITHUB_REF": "refs/heads/master",
            "DEFAULT_BRANCH": "master",
            "GITHUB_REPOSITORY": "fixture/repo",
        }
        with patch.dict(os.environ, env), patch.object(
            ci_gate, "validate_release", return_value={"budget_scope": "fixture"}
        ):
            for runs, total, allowed in (
                ([current], 1, True),
                ([current, {"id": 122, "display_title": current["display_title"]}], 2, False),
                ([], 1, False),
                ([current], 2, False),
                ([current, current], 2, False),
            ):
                with patch.object(
                    ci_gate.subprocess,
                    "check_output",
                    return_value=encoded({"total_count": total, "workflow_runs": runs}),
                ):
                    if allowed:
                        self.assertEqual(
                            ci_gate.admit(
                                "maintenance/detector_validation/releases/fixture.json", sha256
                            )["status"],
                            "admitted",
                        )
                    else:
                        with self.assertRaises(ValueError):
                            ci_gate.admit(
                                "maintenance/detector_validation/releases/fixture.json", sha256
                            )

    def test_missing_comparison_writes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "failure.json"
            argv = [
                "ci_gate",
                "compare",
                "--sha256",
                "a" * 64,
                "--base",
                str(Path(temp) / "absent"),
                "--candidate",
                str(Path(temp) / "absent2"),
                "--out",
                str(output),
            ]
            with patch("sys.argv", argv):
                self.assertEqual(ci_gate.main(), 2)
            self.assertEqual(json.loads(output.read_text())["scientific_verdict"], "INCOMPLETE")

    def test_pending_and_stale_reviews_fail(self):
        with patch.object(ci, "git", side_effect=lambda *a: a[-1].removesuffix("^{commit}")):
            release = ci.draft("a" * 40, "b" * 40)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "release.json"
            release["budget_scope"] = "synthetic policy test only"
            release["archive_plan"] = "synthetic fixture, no production permission"
            save(path, release)
            with self.assertRaisesRegex(ValueError, "pending review"):
                ci.validate_release(path, digest(path))
            # Deliberately fabricated test records are confined to this temporary fixture.
            for role, review in release["reviews"].items():
                review.update(
                    decision="approved",
                    reviewer="synthetic/" + role,
                    evidence="fixture",
                    binding_sha256=ci.binding(release),
                )
            path.unlink()
            save(path, release)
            self.assertEqual(ci.validate_release(path, digest(path)), release)
            release["candidate"] = "c" * 40
            path.unlink()
            save(path, release)
            with self.assertRaisesRegex(ValueError, "stale review"):
                ci.validate_release(path, digest(path))

    def test_missing_duplicates_and_failure_semantics(self):
        reports = [
            {"id": c["id"], "rows": 100000, "scientific_status": "PASS"}
            for c in profile("scientific")["cases"]
        ]
        self.assertEqual(ci_collect.summarize(reports, [])["scientific_verdict"], "PASS")
        for bad in (reports[:-1], reports + [reports[0]], reports[1:] + [reports[1]]):
            self.assertEqual(ci_collect.summarize(bad, [])["scientific_verdict"], "INCOMPLETE")
        reports[0]["scientific_status"] = "FAIL"
        result = ci_collect.summarize(reports, [])
        self.assertEqual(result["integrity_status"], "PASS")
        self.assertEqual(result["scientific_verdict"], "FAIL")
        self.assertEqual(
            ci_collect.summarize(reports[:-1], ["missing"])["scientific_verdict"], "FAIL"
        )

    def test_comparison_requires_both_runtime_and_consistent_reports(self):
        cases = [
            {"id": c["id"], "rows": 100000, "scientific_status": "PASS"}
            for c in profile("scientific")["cases"]
        ]
        base = {
            **ci.context("a" * 64, "base"),
            **ci_collect.summarize(cases, []),
            "runtime_content": {"fixture": 1},
        }
        candidate = {**base, "role": "candidate"}
        self.assertEqual(
            ci_gate.comparison(base, candidate, "a" * 64)["scientific_verdict"], "PASS"
        )
        for changes in (
            {"run_id": "124"},
            {"runtime_content": {"fixture": 2}},
            {"cases": cases[:-1]},
            {"scientific_verdict": "FAIL"},
        ):
            with self.assertRaises(ValueError):
                ci_gate.comparison(base, {**candidate, **changes}, "a" * 64)


def npy(rows, modes):
    header = f"{{'descr': '<i8', 'fortran_order': False, 'shape': ({rows}, {modes}), }}".encode()
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + bytes(rows * modes * 8)


class ArtifactTests(unittest.TestCase):
    """Exercise real bounded stores, raw parsing, seeds/commands and aggregation.

    Metadata/import validation and costly scientific evaluation are explicit mocks
    here; these fixtures test orchestration and are never scientific evidence.
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.source = {"simulator_commit": "a" * 40, "files": {}}
        self.runtime = {
            "invocation_executable": "/tmp/synthetic/bin/python",
            "executable": "/usr/bin/python",
            "pins": {},
        }
        self.args = SimpleNamespace(
            sha256="a" * 64, role="base", input=str(self.root), out=str(self.root / "report.json")
        )
        self.calls = []

    def artifact(self, name="shard0", mutation=None, defect=None):
        def execution_bytes(record):
            if defect == "worker_cpu":
                record["supervision"]["tracked_child_wait4"]["cpu_seconds"] = 999
            elif defect == "worker_wall":
                record["wall_seconds"] = 999
            elif defect == "worker_unreaped":
                record["supervision"]["cleanup_reaped"] = False
            elif defect == "worker_violation":
                record["postflight_resource_violations"] = ["synthetic violation"]
            return encoded(record)

        root = self.root / name
        root.mkdir()
        bound = ci_collect.binding(self.args.sha256, "base", 0, self.source, self.runtime)
        if mutation:
            mutation(bound)
        collection = {
            "status": "collected_not_accepted",
            "failure": None,
            "completed_chunks": 40,
            "cpu_seconds": 10 if defect == "understated_cpu" else 44,
            "wall_seconds": 10 if defect == "understated_wall" else 80,
        }
        save(root / "binding.json", bound)
        save(root / "collection.json", collection)
        store = StoreWriter(
            root / "evidence.bin", ci.LIMITS["capacity_bytes"], ci.LIMITS["reserve_bytes"]
        )
        try:
            store.put("binding.json", encoded(bound))
            store.put("collection.json", encoded(collection))
            for case in ci.cases_for(0):
                for chunk in range(10):
                    prefix = f"{case['id']}/{chunk:02d}"
                    if defect == "missing_chunk" and chunk == 0:
                        continue
                    command = [
                        self.runtime["invocation_executable"],
                        "-B",
                        "-m",
                        "maintenance.detector_validation.fresh_worker",
                        "--case-sha256",
                        sha(encoded(case)),
                        "--chunk",
                        str(chunk),
                    ]
                    if defect == "command_seed":
                        command[-3] = "b" * 64
                    store.put(f"commands/{prefix}/command.json", encoded(command))
                    store.put(f"commands/{prefix}/started.json", b"{}")
                    store.put(
                        f"commands/{prefix}/exit.json",
                        execution_bytes(
                            {
                                "exit_code": 0,
                                "limit_failure": None,
                                "requested_cpu_limit": 120,
                                "requested_controller_cpu_ceiling": 5,
                                "cpu_seconds": 1,
                                "wall_seconds": 2,
                                "controller_whole_chunk_delta": {"process_cpu_seconds": 0.1},
                                "postflight_resource_violations": [],
                                "supervision": {
                                    "requested_wall_ceiling_seconds": 180,
                                    "requested_own_cpu_tripwire_seconds": 5,
                                    "exit_code": 0,
                                    "limit_failure": None,
                                    "cleanup_reaped": True,
                                    "tracked_child_wait4": {"cpu_seconds": 1},
                                    "observed_since_launch_wall_seconds": 2,
                                    "controller_delta": {"process_cpu_seconds": 0.1},
                                },
                            }
                        ),
                    )
                    for label in ("stdout", "stderr"):
                        store.put(f"commands/{prefix}/{label}.txt", b"")
                    store.put(
                        f"raw/{prefix}/samples.npy", npy(10000, len(case["output_mode_order"]))
                    )
                    store.put(
                        f"raw/{prefix}/metadata.json",
                        encoded(
                            {
                                "case": case,
                                "seed": (
                                    -1 if defect == "metadata_seed" else case["chunk_seeds"][chunk]
                                ),
                            }
                        ),
                    )
                    if case["path"] == "local_engine_gaussian":
                        store.put(
                            f"raw/{prefix}/secondary.npy", npy(10000, len(case["measured_modes"]))
                        )
            store.checkpoint({"status": "collected_not_accepted"})
        finally:
            store.close()
        self.checksums(root)
        return root

    def checksums(self, root):
        p = root / "checksums.json"
        if p.exists():
            p.unlink()
        save(p, {f.name: digest(f) for f in root.iterdir()})

    def inspect(self, root):
        def metadata(meta, case, chunk, *_):
            self.assertEqual(meta["case"], case)
            self.assertEqual(meta["seed"], case["chunk_seeds"][chunk])
            self.calls.append((case["id"], chunk))

        def evaluate(case, counts, n, statistics):
            self.assertEqual(n, 100000)
            self.assertEqual(statistics, profile("scientific")["statistics"])
            return {"status": "accepted"}

        with patch.object(ci_collect, "validate_metadata", side_effect=metadata), patch.object(
            ci_collect, "evaluate_counts", side_effect=evaluate
        ):
            return ci_collect.inspect_shard(root, self.args, self.source, self.runtime)

    def test_real_container_raw_recomputation(self):
        index, cases, resources = self.inspect(self.artifact())
        self.assertEqual(index, 0)
        self.assertEqual(len(cases), 4)
        self.assertEqual(len(self.calls), 40)
        self.assertEqual(sum(c["rows"] for c in cases), 400000)

    def test_collection_totals_cover_validated_chunk_costs(self):
        for defect in ("understated_cpu", "understated_wall"):
            with self.subTest(defect=defect), self.assertRaisesRegex(
                ValueError, "understates credited worker costs"
            ):
                self.inspect(self.artifact(defect, defect=defect))

    def test_observed_execution_rejected_despite_success_strings(self):
        for defect in ("worker_cpu", "worker_wall", "worker_unreaped", "worker_violation"):
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.inspect(self.artifact(defect, defect=defect))

    def test_wrong_run_revision_seed_family_and_runtime(self):
        for field, value in (
            ("run_id", "124"),
            ("role", "candidate"),
            ("release_sha256", "b" * 64),
            ("case_ids", []),
            ("runtime", {"pins": {"changed": 1}}),
            ("source", {}),
        ):
            with self.subTest(field=field):
                root = self.artifact(field, lambda b: b.update({field: value}))
                with self.assertRaises(ValueError):
                    self.inspect(root)

    def test_missing_mismatched_truncated_and_extra_files(self):
        root = self.artifact()
        (root / "collection.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            self.inspect(root)
        root = self.artifact("corrupt")
        with (root / "evidence.bin").open("r+b") as stream:
            stream.seek(4100)
            stream.write(b"X")
        self.checksums(root)
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            self.inspect(root)
        root = self.artifact("truncated")
        with (root / "evidence.bin").open("r+b") as stream:
            stream.truncate(100)
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.inspect(root)
        root = self.artifact("extra")
        (root / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "missing/extra"):
            self.inspect(root)

    def test_missing_raw_chunk_and_mismatched_seed_command(self):
        for defect in ("missing_chunk", "command_seed", "metadata_seed"):
            with self.subTest(defect=defect):
                with self.assertRaises((ValueError, AssertionError)):
                    self.inspect(self.artifact(defect, defect=defect))

    def test_absent_input_and_malformed_checksums_keep_receipt(self):
        self.args.input = str(self.root / "absent")
        with patch.object(ci_collect, "source_identity", return_value=self.source), patch.object(
            ci_collect, "runtime_identity", return_value=self.runtime
        ):
            self.assertEqual(ci_collect.aggregate(self.args, {"base": "a" * 40}), 1)
        self.assertEqual(
            json.loads(Path(self.args.out).read_text())["scientific_verdict"], "INCOMPLETE"
        )
        Path(self.args.out).unlink()
        self.args.input = str(self.root)
        root = self.artifact()
        (root / "checksums.json").write_text("invalid JSON")
        with patch.object(ci_collect, "source_identity", return_value=self.source), patch.object(
            ci_collect, "runtime_identity", return_value=self.runtime
        ):
            self.assertEqual(ci_collect.aggregate(self.args, {"base": "a" * 40}), 1)
        self.assertIn("JSONDecodeError", Path(self.args.out).read_text())

    def test_duplicate_shard_aggregate_fails_and_preserves_results(self):
        first, second = self.artifact(), self.artifact("renamed_duplicate")
        with patch.object(ci_collect, "source_identity", return_value=self.source), patch.object(
            ci_collect, "runtime_identity", return_value=self.runtime
        ), patch.object(
            ci_collect,
            "inspect_shard",
            side_effect=lambda d, *_: (
                0,
                [{"id": ci.cases_for(0)[0]["id"], "rows": 100000, "scientific_status": "FAIL"}],
                {},
            ),
        ):
            self.assertEqual(ci_collect.aggregate(self.args, {"base": "a" * 40}), 1)
        report = json.loads(Path(self.args.out).read_text())
        self.assertEqual(report["scientific_verdict"], "FAIL")
        self.assertIn("duplicate shard", str(report["errors"]))
        self.assertTrue((first / "evidence.bin").exists() and (second / "evidence.bin").exists())


if __name__ == "__main__":
    unittest.main()
