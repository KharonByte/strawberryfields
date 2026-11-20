"""Prospective resource admission and bounded synthetic CPU-monitor controls."""

import copy
import os
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from maintenance.detector_validation.fresh import (
    job_limits,
    validate_metadata,
    validate_monitored_record,
)
from maintenance.detector_validation.fresh_profile import profile
from maintenance.detector_validation.supervision import supervise


class ResourcePolicyTests(unittest.TestCase):
    def test_scientific_design_and_historical_resources_preserved(self):
        old, new = profile("scientific"), profile("scientific-v2")
        self.assertEqual(old["cases"], new["cases"])
        self.assertEqual(old["statistics"], new["statistics"])
        self.assertNotIn("worker_cpu_seconds", old["resources"])
        for field in ("cpu_seconds", "wall_seconds", "capacity_bytes"):
            self.assertEqual(old["resources"][field], new["resources"][field])
        self.assertEqual(job_limits(new["resources"], 0, 0), (180, 240))
        self.assertEqual(job_limits(profile("operational")["resources"], 0, 0), (120, 240))

    def test_complete_reservation_boundary_no_shrinking(self):
        limits = profile("scientific-v2")["resources"]
        cpu = limits["cpu_seconds"] - 260
        wall = limits["wall_seconds"] - 300
        own = limits["controller_cpu_seconds"] - 80
        self.assertEqual(job_limits(limits, cpu, wall, own), (180, 240))
        for values in (
            (cpu + 0.01, wall, own),
            (cpu, wall + 0.01, own),
            (cpu, wall, own + 0.01),
        ):
            with self.assertRaisesRegex(ValueError, "remaining budget"):
                job_limits(limits, *values)

    def test_missing_monitor_policy_fails_before_scientific_validation(self):
        limits = profile("scientific-v2")["resources"]
        with self.assertRaisesRegex(ValueError, "monitored worker policy"):
            validate_metadata(
                {
                    "rows": 10000,
                    "measurement_execution": {"status": "returned"},
                    "requested_mode_order": [0],
                },
                {"chunk_rows": 10000, "measured_modes": [0]},
                0,
                {},
                {},
                {},
                limits,
            )

    def test_recorded_overruns_and_malformed_monitor_are_rejected(self):
        limits = profile("scientific-v2")["resources"]
        record = {
            "requested_cpu_limit": 180,
            "cpu_seconds": 149,
            "exit_code": 0,
            "limit_failure": None,
            "requested_controller_cpu_ceiling": 20,
            "controller_whole_chunk_delta": {"process_cpu_seconds": 7},
            "wall_seconds": 150,
            "postflight_resource_violations": [],
            "supervision": {
                "requested_wall_ceiling_seconds": 240,
                "exit_code": 0,
                "limit_failure": None,
                "observed_since_launch_wall_seconds": 150,
                "controller_delta": {"process_cpu_seconds": 7},
                "requested_own_cpu_tripwire_seconds": 18,
                "cleanup_reaped": True,
                "tracked_child_wait4": {"cpu_seconds": 149},
                "child_cpu_monitor": {
                    "tripwire_cpu_seconds": 170,
                    "poll_seconds": 0.02,
                    "polls": 5000,
                    "affinity_count": 5,
                    "max_poll_gap_seconds": 0.4,
                    "last": {"own_cpu_seconds_tick_resolution": 149},
                },
            },
        }
        validate_monitored_record(record, limits)
        changes = [
            (("wall_seconds",), 241),
            (("controller_whole_chunk_delta", "process_cpu_seconds"), 21),
            (("supervision", "tracked_child_wait4", "cpu_seconds"), 181),
            (("supervision", "child_cpu_monitor", "polls"), -1),
            (("supervision", "child_cpu_monitor", "affinity_count"), 6),
            (("supervision", "child_cpu_monitor", "max_poll_gap_seconds"), float("nan")),
            (("supervision", "child_cpu_monitor", "max_poll_gap_seconds"), -1),
            (("supervision", "child_cpu_monitor", "max_poll_gap_seconds"), 1.1),
        ]
        for path, value in changes:
            bad = copy.deepcopy(record)
            target = bad
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                validate_monitored_record(bad, limits)

    def test_unmonitored_cannot_request_higher_worker_limit(self):
        from maintenance.detector_validation.collect import run_worker

        with self.assertRaisesRegex(ValueError, "invalid worker resource"):
            run_worker(None, [], b"", {}, "test", 180, 240)


@unittest.skipUnless(sys.platform == "linux", "Linux pidfd/CPU monitor")
class CpuMonitorTests(unittest.TestCase):
    def run_child(self, source, **options):
        def affinity():
            os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:5])

        proc = subprocess.Popen(
            [sys.executable, "-B", "-c", source],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=affinity,
        )
        try:
            return supervise(
                proc,
                {},
                started=time.monotonic(),
                wall_ceiling=4,
                cleanup_grace=0.2,
                own_cpu_tripwire=2,
                **options
            )
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait()

    def test_busy_child_stopped_and_reaped_at_cpu_tripwire(self):
        result = self.run_child("while True: pass", child_cpu_tripwire=0.1)
        self.assertEqual(result["limit_failure"], "child_cpu_headroom_tripwire")
        self.assertEqual(result["exit_code"], -signal.SIGKILL)
        self.assertTrue(result["cleanup_reaped"])
        self.assertLess(result["tracked_child_wait4"]["cpu_seconds"], 1)
        self.assertFalse(result["strict_cpu_enforcement_verified"])

    def test_valid_child_finishes_without_monitor_failure(self):
        result = self.run_child("import time; time.sleep(.05)", child_cpu_tripwire=1)
        self.assertIsNone(result["limit_failure"])
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["cleanup_reaped"])

    def test_missing_counters_fail_closed(self):
        with patch(
            "maintenance.detector_validation.supervision.process_stat",
            return_value={"unavailable": "test"},
        ):
            result = self.run_child("import time; time.sleep(3)", child_cpu_tripwire=1)
        self.assertEqual(result["limit_failure"], "child_cpu_monitor_unavailable")
        self.assertTrue(result["cleanup_reaped"])
