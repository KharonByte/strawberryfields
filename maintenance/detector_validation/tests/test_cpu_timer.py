"""Process CPU clock guard controls; no detector sampling."""

import copy
import os
import resource
import signal
import subprocess
import sys
import time
import unittest

from maintenance.detector_validation.fresh import validate_cpu_timer_records
from maintenance.detector_validation.fresh_profile import profile


class TimerEvidenceTests(unittest.TestCase):
    def test_science_and_old_resources_unchanged(self):
        old, new = profile("scientific-v2"), profile("scientific-v3")
        self.assertEqual(old["cases"], new["cases"])
        self.assertEqual(old["statistics"], new["statistics"])
        self.assertNotIn("worker_cpu_timer_seconds", old["resources"])
        for key, value in old["resources"].items():
            self.assertEqual(new["resources"][key], value)

    def test_absolute_timer_records_fail_closed(self):
        limits = profile("scientific-v3")["resources"]
        names = (
            "cpu_timer_before",
            "cpu_timer_before_measurement",
            "cpu_timer_after_measurement",
            "cpu_timer_after",
        )
        meta = {
            name: dict(
                observation_policy="three-reads-width-only-v1",
                observation_count=1,
                rejected_observations=[],
                timer="CLOCK_PROCESS_CPUTIME_ID",
                signal="SIGALRM",
                budget_seconds=170,
                native_default=True,
                native_unblocked=True,
                interval_seconds=0,
                observation_tolerance_seconds=0.01,
                cpu_before_seconds=i + 1,
                cpu_after_seconds=i + 1.001,
                remaining_seconds=169 - i,
            )
            for i, name in enumerate(names)
        }
        validate_cpu_timer_records(meta, limits)
        for key, bad in (
            ("native_default", False),
            ("native_unblocked", False),
            ("remaining_seconds", 0),
            ("remaining_seconds", float("nan")),
            ("remaining_seconds", 169),
            ("interval_seconds", 1),
            ("budget_seconds", 180),
            ("cpu_before_seconds", None),
            ("cpu_after_seconds", float("nan")),
            ("cpu_before_seconds", 100),
            ("cpu_after_seconds", 3),
            ("timer", "ITIMER_PROF"),
        ):
            wrong = copy.deepcopy(meta)
            wrong[names[-1]][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_cpu_timer_records(wrong, limits)
        for name in names:
            wrong = copy.deepcopy(meta)
            del wrong[name]
            with self.assertRaises(ValueError):
                validate_cpu_timer_records(wrong, limits)

    def test_command_and_timer_policy_must_agree(self):
        from maintenance.detector_validation.collect import run_worker

        for command, enabled in ((["python"], True), (["python", "--cpu-timer"], False)):
            with self.assertRaisesRegex(ValueError, "command/policy mismatch"):
                run_worker(
                    None, command, b"", {}, "test", 180, 240, monitored=True, cpu_timer=enabled
                )


@unittest.skipUnless(sys.platform == "linux", "Reviewed Linux process timer ABI")
class TimerExecutionTests(unittest.TestCase):
    imports = "from maintenance.detector_validation.supervision import start_worker_cpu_timer,worker_cpu_timer_state; "

    def test_busy_child_dies_while_parent_waits(self):
        code = self.imports + "start_worker_cpu_timer(.4)\nwhile True: pass"
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc = subprocess.Popen(
            [sys.executable, "-B", "-c", code],
            start_new_session=True,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CPU, (3, 3)),
        )
        try:
            time.sleep(1)
            self.assertEqual(proc.wait(timeout=2), -signal.SIGALRM)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.assertLess(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime, 1)
        finally:
            if proc.returncode is None:
                proc.kill()
                proc.wait(timeout=1)

    def test_sleep_does_not_exhaust_cpu_guard(self):
        code = (
            self.imports
            + "import time; start_worker_cpu_timer(.4); time.sleep(.15); worker_cpu_timer_state(.4)"
        )
        self.assertEqual(
            subprocess.run([sys.executable, "-B", "-c", code], timeout=3).returncode, 0
        )

    def test_signal_mutation_and_rearming_rejected(self):
        for mutation in (
            "signal.signal(signal.SIGALRM,signal.SIG_IGN)",
            "signal.pthread_sigmask(signal.SIG_BLOCK,[signal.SIGALRM])",
            "start_worker_cpu_timer(.4)",
        ):
            code = (
                self.imports
                + "import signal; start_worker_cpu_timer(.4)\ntry:\n "
                + mutation
                + '\n worker_cpu_timer_state(.4)\nexcept RuntimeError:\n pass\nelse:\n raise AssertionError("invalid timer accepted")'
            )
            with self.subTest(mutation=mutation):
                self.assertEqual(
                    subprocess.run([sys.executable, "-B", "-c", code], timeout=3).returncode, 0
                )
