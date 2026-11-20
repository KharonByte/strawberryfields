"""New operational allocation: historical policies and science stay immutable."""

import copy
import sys
import unittest
from unittest.mock import patch

from maintenance.detector_validation.fresh_profile import profile
from maintenance.detector_validation.fresh import job_limits, validate_accounting
from maintenance.detector_validation.case_evidence import worker_command
from maintenance.detector_validation.collect import run_worker
from maintenance.detector_validation.provenance import sanitized_environment
from maintenance.detector_validation.storage import encoded, sha


class ResourceV4Tests(unittest.TestCase):
    def test_scientific_design_and_prior_profile_unchanged(self):
        old = profile("scientific-v3")
        new = profile("scientific-v4")
        self.assertEqual(old["cases"], new["cases"])
        self.assertEqual(old["statistics"], new["statistics"])
        self.assertEqual(old["resources"]["worker_cpu_seconds"], 180)
        self.assertEqual(old["resources"]["worker_cpu_timer_seconds"], 170)
        self.assertNotIn("preservation_ceiling_bytes", old["resources"])
        self.assertEqual(job_limits(new["resources"], 0, 0), (300, 360))

    def test_complete_reservations_and_retention(self):
        lim = profile("scientific-v4")["resources"]
        used = (72000 - 380, 86400 - 420, 13680 - 80)
        self.assertEqual(job_limits(lim, *used), (300, 360))
        for i in range(3):
            bad = list(used)
            bad[i] += 0.01
            with self.assertRaises(ValueError):
                job_limits(lim, *bad)
        account = dict(
            schema="detector-public-accounting-v7",
            scope="existing_budget_new_activity",
            scope_id="synthetic",
            prior_cpu_seconds=None,
            prior_wall_seconds=None,
            prior_accounting_status="unreconciled_explicit_new_activity_allowance",
            historical_status="prior_results_preserved",
            retained_workspace_bytes=3900000000,
            activity_allowance=lim,
        )
        validate_accounting(account, lim)
        legacy = profile("scientific-v3")["resources"]
        with self.assertRaises(ValueError):
            validate_accounting(dict(account, activity_allowance=legacy), legacy)
        with self.assertRaises(ValueError):
            validate_accounting(dict(account, retained_workspace_bytes=5 * 1024**3), lim)

    def test_command_policy_pairs_rejected_before_launch(self):
        command = worker_command(sys.executable, {}, 0, cpu_timer=True, cpu_timer_budget=290)
        for cpu, wall, budget, cmd in [
            (180, 240, 290, command),
            (300, 360, 170, command),
            (300, 360, 290, command[:-2]),
            (300, 360, 290, command + ["--cpu-timer-budget", "290"]),
            (180, 240, 170, command[:-2] + ["--cpu-timer-budget=290"]),
        ]:
            with self.subTest(cpu=cpu, wall=wall, budget=budget), patch(
                "maintenance.detector_validation.collect.subprocess.Popen"
            ) as spawn:
                with self.assertRaises(ValueError):
                    run_worker(
                        None,
                        cmd,
                        b"",
                        {},
                        "synthetic",
                        cpu,
                        wall,
                        monitored=True,
                        cpu_timer=True,
                        cpu_timer_budget=budget,
                    )
                spawn.assert_not_called()

    @unittest.skipUnless(sys.platform == "linux", "Linux resource/timer policy")
    def test_actual_extended_worker_preflight_without_sampling(self):
        class Store:
            def put(self, *args):
                pass

        data = encoded({})
        cmd = [
            sys.executable,
            "-B",
            "-m",
            "maintenance.detector_validation.fresh_worker",
            "--case-sha256",
            sha(data),
            "--chunk",
            "0",
            "--cpu-timer",
            "--cpu-timer-budget",
            "290",
        ]
        result, meta = run_worker(
            Store(),
            cmd,
            data,
            sanitized_environment(),
            "synthetic",
            300,
            360,
            monitored=True,
            cpu_timer=True,
            cpu_timer_budget=290,
        )
        self.assertEqual(result["exit_code"], 1)  # invalid case, before capture
        self.assertTrue(result["supervision"]["cleanup_reaped"])
        self.assertEqual(result["frames"], [1, 5])
        self.assertEqual(
            meta["verified_resource_limits"]["cpu_seconds"], {"soft": 300, "hard": 300}
        )
        self.assertEqual(meta["cpu_timer_armed"]["budget_seconds"], 290)
        self.assertEqual(meta["cpu_timer_before"]["budget_seconds"], 290)
        self.assertNotIn("measurement_execution", meta)
