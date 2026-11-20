"""Synthetic bookkeeping controls: no simulator or scientific sampling."""

import copy
import unittest

from maintenance.detector_validation.fresh_inventory import reconcile, ROOT_MEMBERS, COMMAND_MEMBERS
from maintenance.detector_validation.storage import encoded


class Reader:
    def __init__(self, members, state):
        self.members, self.state = members, state

    def inventory(self):
        return dict.fromkeys(self.members)

    def read_bytes(self, name):
        return self.members[name]


class InventoryTests(unittest.TestCase):
    def fixture(self):
        manifest = {
            "experiment_id": "fixture",
            "profile": "scientific",
            "profile_definition": {
                "cases": [{"id": "a", "chunk_count": 2, "path": "gaussian_direct"}]
            },
        }
        record = {
            "exit_code": 0,
            "limit_failure": None,
            "wall_seconds": 1,
            "supervision": {"tracked_child_wait4": {"cpu_seconds": 0.5}},
            "controller_whole_chunk_delta": {"process_cpu_seconds": 0.1},
        }
        collection = {
            "schema": "detector-fresh-collection-v7",
            "experiment_id": "fixture",
            "profile": "scientific",
            "release_sha256": "sha",
            "status": "collected_not_accepted",
            "cases": [{"id": "a", "chunks": [{"index": i, **record} for i in range(2)]}],
        }
        members = dict.fromkeys(ROOT_MEMBERS, b"{}")
        members["attempt-start.json"] = encoded(
            {
                "schema": "detector-fresh-start-v7",
                "experiment_id": "fixture",
                "release_sha256": "sha",
                "status": "started_completion_unknown",
                "single_attempt": True,
            }
        )
        for i in range(2):
            prefix = f"a/{i:02d}"
            for name in COMMAND_MEMBERS:
                members[f"commands/{prefix}/{name}"] = (
                    encoded(record) if name == "exit.json" else b"{}"
                )
            for name in ("samples.npy", "metadata.json"):
                members[f"raw/{prefix}/{name}"] = b"{}"
        reader = Reader(members, {"status": collection["status"], "experiment_id": "fixture"})
        return reader, manifest, collection

    def test_complete_and_contradictory_bookkeeping(self):
        reader, manifest, collection = self.fixture()
        self.assertEqual(reconcile(reader, manifest, collection, "sha"), {("a", 0), ("a", 1)})
        mutations = [
            lambda r, m, c: r.state.update(status="incomplete"),
            lambda r, m, c: c.update(experiment_id="wrong"),
            lambda r, m, c: c["cases"].clear(),
            lambda r, m, c: c["cases"][0]["chunks"].pop(),
            lambda r, m, c: c["cases"][0]["chunks"][0].update(exit_code=1),
            lambda r, m, c: r.members.update({"raw/a/00/unexpected": b"x"}),
            lambda r, m, c: r.members.pop("commands/a/00/started.json"),
            lambda r, m, c: r.members.update({"commands/a/00/partial-frame.bin": b"x"}),
        ]
        for mutation in mutations:
            r, m, c = copy.deepcopy((reader, manifest, collection))
            mutation(r, m, c)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, KeyError)):
                reconcile(r, m, c, "sha")

    def test_partial_unledgered_terminal_output_gets_no_credit(self):
        r, m, c = self.fixture()
        c["status"] = r.state["status"] = "incomplete"
        c["cases"][0]["chunks"].pop()
        self.assertEqual(reconcile(r, m, c, "sha"), {("a", 0)})
        # A final command that failed before launch is also legitimate partial evidence.
        r.members = {
            k: v for k, v in r.members.items() if "/a/01/" not in k or k.endswith("command.json")
        }
        self.assertEqual(reconcile(r, m, c, "sha"), {("a", 0)})

    def test_failed_ledgered_worker_gets_no_credit(self):
        r, m, c = self.fixture()
        c["status"] = r.state["status"] = "incomplete"
        record = {"exit_code": 1, "limit_failure": "failure"}
        c["cases"][0]["chunks"][1] = {"index": 1, **record}
        r.members["commands/a/01/exit.json"] = encoded(record)
        self.assertEqual(reconcile(r, m, c, "sha"), {("a", 0)})

    def test_no_retry_after_failed_worker(self):
        r, m, c = self.fixture()
        c["status"] = r.state["status"] = "incomplete"
        record = {"exit_code": 1, "limit_failure": "failure"}
        c["cases"][0]["chunks"][0] = {"index": 0, **record}
        r.members["commands/a/00/exit.json"] = encoded(record)
        with self.assertRaisesRegex(ValueError, "continued"):
            reconcile(r, m, c, "sha")


class ReportTests(unittest.TestCase):
    fixture = InventoryTests.fixture

    def test_report_credits_only_reconciled_rows_and_synchronizes_verdict(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from maintenance.detector_validation import fresh, reanalyse

        for partial in (False, True):
            r, m, c = self.fixture()
            m["lineage"] = "fresh_independent_experiment"
            m["profile_definition"]["resources"] = {"cpu_seconds": 100, "wall_seconds": 100}
            m["profile_definition"]["statistics"] = {}
            case = m["profile_definition"]["cases"][0]
            case.update(
                chunk_rows=10000,
                output_mode_order=[0],
                measured_modes=[0],
                chunk_seeds=[1, 2],
                detector="pnr",
            )
            c.update(
                source_post_matches=True,
                runtime_post_matches=True,
                resources={
                    "requested": m["profile_definition"]["resources"],
                    "observed_family_cpu_seconds": 10,
                    "observed_wall_seconds": 10,
                },
            )
            if partial:
                c["status"] = r.state["status"] = "incomplete"
                c["cases"][0]["chunks"].pop()
            runtime = {"invocation_executable": "/python"}
            for i in range(2):
                prefix = f"a/{i:02d}"
                r.members[f"raw/{prefix}/metadata.json"] = encoded(
                    {
                        "case": case,
                        "seed": i + 1,
                        "rows": 10000,
                        "output_mode_order": [0],
                        "status": "captured_not_accepted",
                    }
                )
                r.members[f"commands/{prefix}/command.json"] = encoded(
                    [
                        "/python",
                        "-B",
                        "-m",
                        "maintenance.detector_validation.fresh_worker",
                        "--case-sha256",
                        fresh.sha(encoded(case)),
                        "--chunk",
                        str(i),
                    ]
                )
            r.members["collection.json"] = encoded(c)
            r.close = lambda: None

            def load(path):
                return (
                    c
                    if path.name == "collection-receipt.json"
                    else runtime if path.name == "runtime.json" else {}
                )

            with patch.object(
                fresh, "experiment", return_value=(Path("/fixture"), m)
            ), patch.object(fresh, "check_release"), patch.object(
                fresh, "StoreReader", return_value=r
            ), patch.object(
                fresh, "raw_bytes", side_effect=lambda p: r.members[p.name]
            ), patch.object(
                fresh, "load", side_effect=load
            ), patch.object(
                fresh, "source_identity", return_value={}
            ), patch.object(
                fresh, "validate_metadata"
            ), patch.object(
                fresh, "save"
            ), patch.object(
                reanalyse, "parse_npy", return_value=[(0,)]
            ), patch.object(
                reanalyse, "structurally_valid", return_value=True
            ), patch.object(
                reanalyse, "replay_period", return_value=None
            ), patch.object(
                reanalyse, "raw_indicators", return_value=[]
            ), patch.object(
                reanalyse, "evaluate_counts", return_value={"status": "accepted"}
            ) as score:
                result = fresh.report(
                    SimpleNamespace(experiment="/fixture", release_sha256="sha", out="/unused")
                )
                evaluated = score.call_count
                if not partial:
                    c["resources"]["observed_family_cpu_seconds"] = 0.01
                    r.members["collection.json"] = encoded(c)
                    with self.assertRaisesRegex(ValueError, "understates"):
                        fresh.report(
                            SimpleNamespace(
                                experiment="/fixture", release_sha256="sha", out="/unused"
                            )
                        )
            self.assertEqual(result["accuracy"], result["scientific_verdict"])
            self.assertEqual(result["scientific_verdict"], "INCOMPLETE" if partial else "PASS")
            self.assertEqual(result["cases"][0]["rows"], 1 if partial else 2)
            self.assertEqual(evaluated, 0 if partial else 1)
