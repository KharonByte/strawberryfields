"""Prospective custody boundaries; no scientific sampling or historical edits."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from maintenance.detector_validation import fresh
from maintenance.detector_validation.fresh_profile import profile


class ResourceV5Tests(unittest.TestCase):
    def account(self, limits, retained, independent=False):
        return dict(
            schema="detector-public-accounting-v7",
            scope="new_independent_budget" if independent else "existing_budget_new_activity",
            scope_id="synthetic-boundary",
            prior_cpu_seconds=0,
            prior_wall_seconds=0,
            historical_status="not_applicable" if independent else "prior_results_preserved",
            retained_workspace_bytes=retained,
            activity_allowance=limits,
        )

    def test_historical_profiles_and_only_prospective_resource_delta(self):
        names = ("operational", "scientific", "scientific-v2", "scientific-v3", "scientific-v4")
        historical = json.dumps(
            {name: profile(name) for name in names}, sort_keys=True, separators=(",", ":")
        ).encode()
        # Captured from the owner-merged baseline before adding v5.
        self.assertEqual(
            hashlib.sha256(historical).hexdigest(),
            "f4be10a388c21d3e7515428a4a841463b9373bf559ea24752c323378f2de8f2e",
        )
        old, new = profile("scientific-v4"), profile("scientific-v5")
        self.assertEqual(new["resources"]["preservation_ceiling_bytes"], 6 * 1024**3)
        new["resources"]["preservation_ceiling_bytes"] = old["resources"][
            "preservation_ceiling_bytes"
        ]
        new["purpose"] = old["purpose"]
        self.assertEqual(new, old)
        self.assertEqual(fresh.job_limits(new["resources"], 0, 0), (300, 360))

    def test_exact_cap_and_one_byte_over_in_both_accounting_scopes(self):
        for name in ("scientific-v4", "scientific-v5"):
            limits = profile(name)["resources"]
            boundary = (
                limits["preservation_ceiling_bytes"] - limits["capacity_bytes"] - 16 * 1024**2
            )
            for independent in (False, True):
                with self.subTest(profile=name, independent=independent):
                    fresh.validate_accounting(self.account(limits, boundary, independent), limits)
                    with self.assertRaisesRegex(ValueError, "cumulative workspace"):
                        fresh.validate_accounting(
                            self.account(limits, boundary + 1, independent), limits
                        )

    def test_measured_retention_requires_new_profile_and_matching_allowance(self):
        old, new = (profile(name)["resources"] for name in ("scientific-v4", "scientific-v5"))
        retained = 4773911025
        self.assertEqual(retained + new["capacity_bytes"] + 16 * 1024**2, 5595994609)
        fresh.validate_accounting(self.account(new, retained), new)
        with self.assertRaisesRegex(ValueError, "cumulative workspace"):
            fresh.validate_accounting(self.account(old, retained), old)
        with self.assertRaisesRegex(ValueError, "allowance must match"):
            fresh.validate_accounting(self.account(old, retained), new)

    def test_v5_report_rejects_runtime_drift_before_scoring(self):
        reader = MagicMock()
        reader.read_bytes.return_value = b"{}"
        with patch.object(
            fresh, "experiment", return_value=(Path("/synthetic"), {"profile": "scientific-v5"})
        ), patch.object(fresh, "check_release"), patch.object(
            fresh, "StoreReader", return_value=reader
        ), patch.object(
            fresh, "raw_bytes", return_value=b"{}"
        ), patch.object(
            fresh, "load", return_value={}
        ), patch.object(
            fresh, "source_identity", return_value={}
        ), patch.object(
            fresh, "runtime_identity", return_value={"changed": True}
        ):
            with self.assertRaisesRegex(ValueError, "analyser runtime differs"):
                fresh.report(SimpleNamespace(experiment="/synthetic", release_sha256="synthetic"))
        reader.close.assert_called_once()

    def test_v5_collection_keeps_monitor_and_metadata_admission(self):
        from maintenance.detector_validation import collect

        spec = profile("scientific-v5")
        # Synthetic bookkeeping fixture, not an altered released scientific profile.
        spec["cases"] = [dict(spec["cases"][0], chunk_count=1)]
        manifest = dict(profile="scientific-v5", profile_definition=spec, experiment_id="synthetic")
        worker_result = {"exit_code": 0, "limit_failure": None}
        with patch.object(
            fresh, "experiment", return_value=(Path("/synthetic"), manifest)
        ), patch.object(fresh, "check_release"), patch.object(fresh, "StoreWriter"), patch.object(
            fresh, "raw_bytes", return_value=b"{}"
        ), patch.object(
            fresh, "load", return_value={}
        ), patch.object(
            fresh, "source_identity", return_value={}
        ), patch.object(
            fresh, "runtime_identity", return_value={}
        ), patch.object(
            fresh, "save"
        ), patch.object(
            collect, "run_worker", return_value=(worker_result, {"status": "captured_not_accepted"})
        ) as worker, patch.object(
            fresh, "validate_metadata", side_effect=ValueError("synthetic metadata rejection")
        ) as validate:
            result = fresh.run(SimpleNamespace(experiment="/synthetic", release_sha256="synthetic"))
        self.assertTrue(worker.call_args.kwargs["monitored"])
        self.assertTrue(worker.call_args.kwargs["cpu_timer"])
        self.assertEqual(worker.call_args.kwargs["cpu_timer_budget"], 290)
        self.assertEqual(worker.call_args.args[5:7], (300, 360))
        validate.assert_called_once()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("synthetic metadata rejection", result["failure"])
