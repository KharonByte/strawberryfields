"""Synthetic successful-receipt and pre-fork parent identity controls."""

import unittest
from unittest.mock import Mock, patch

from maintenance.detector_validation.collect import child_limits
from maintenance.detector_validation.execution_validation import validate_execution_record


def valid_record(cpu=120, wall=180, controller=5, tripwire=5):
    return {
        "requested_cpu_limit": cpu,
        "requested_controller_cpu_ceiling": controller,
        "exit_code": 0,
        "limit_failure": None,
        "postflight_resource_violations": [],
        "cpu_seconds": 1,
        "wall_seconds": 2,
        "controller_whole_chunk_delta": {"process_cpu_seconds": 0.1},
        "supervision": {
            "exit_code": 0,
            "limit_failure": None,
            "requested_wall_ceiling_seconds": wall,
            "requested_own_cpu_tripwire_seconds": tripwire,
            "cleanup_reaped": True,
            "tracked_child_wait4": {"cpu_seconds": 1},
            "observed_since_launch_wall_seconds": 2,
            "controller_delta": {"process_cpu_seconds": 0.1},
        },
    }


class ExecutionEvidenceTests(unittest.TestCase):
    def test_profile_boundaries_preserved(self):
        validate_execution_record(valid_record(), cpu_limit=120, wall_limit=180)
        validate_execution_record(valid_record(10, 20))
        validate_execution_record(valid_record(120, 240), cpu_limit=120, wall_limit=240)
        validate_execution_record(
            valid_record(180, 240, 20, 18),
            cpu_limit=180,
            wall_limit=240,
            controller_ceiling=20,
            controller_tripwire=18,
        )
        for record in (valid_record(121), valid_record(120, 181), valid_record(180, 240, 20, 18)):
            with self.assertRaises(ValueError):
                validate_execution_record(record)
        with self.assertRaises(ValueError):
            validate_execution_record(valid_record(119), cpu_limit=120, wall_limit=180)

    def test_all_observed_resource_fields_fail_closed(self):
        paths = [
            ("cpu_seconds",),
            ("wall_seconds",),
            ("controller_whole_chunk_delta", "process_cpu_seconds"),
            ("supervision", "tracked_child_wait4", "cpu_seconds"),
            ("supervision", "observed_since_launch_wall_seconds"),
            ("supervision", "controller_delta", "process_cpu_seconds"),
        ]
        for path in paths:
            for value in (999, -1, float("nan"), float("inf"), None, True):
                record = valid_record()
                target = record
                for name in path[:-1]:
                    target = target[name]
                target[path[-1]] = value
                with self.subTest(path=path, value=value), self.assertRaises(ValueError):
                    validate_execution_record(record, cpu_limit=120, wall_limit=180)
            del target[path[-1]]
            with self.assertRaises(ValueError):
                validate_execution_record(record)

    def test_cleanup_and_both_verdicts_required(self):
        for path, value in (
            (("exit_code",), 1),
            (("exit_code",), False),
            (("limit_failure",), "postflight"),
            (("postflight_resource_violations",), ["overrun"]),
            (("supervision", "exit_code"), 1),
            (("supervision", "limit_failure"), "timeout"),
            (("supervision", "cleanup_reaped"), False),
            (("requested_controller_cpu_ceiling",), 20),
            (("supervision", "requested_own_cpu_tripwire_seconds"), 18),
        ):
            record = valid_record()
            target = record
            for name in path[:-1]:
                target = target[name]
            target[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_execution_record(record)


class ParentIdentityTests(unittest.TestCase):
    def test_parent_missing_before_or_during_pdeath_setup_exits(self):
        for parents in ([1], [123, 1]):
            libc = Mock()
            libc.prctl.return_value = 0
            with patch(
                "maintenance.detector_validation.collect.os.getppid", side_effect=parents
            ), patch(
                "maintenance.detector_validation.collect.os._exit", side_effect=SystemExit
            ) as stop, patch(
                "maintenance.detector_validation.collect.ctypes.CDLL", return_value=libc
            ), patch(
                "maintenance.detector_validation.collect.resource.setrlimit"
            ) as limits:
                with self.assertRaises(SystemExit):
                    child_limits(120, expected_parent=123)
                stop.assert_called_once_with(125)
                limits.assert_not_called()

    def test_launcher_captures_controller_identity_before_fork(self):
        from maintenance.detector_validation.collect import run_worker

        captured = []

        def inspect_launch(*args, **kwargs):
            captured.append(kwargs["preexec_fn"].keywords["expected_parent"])
            raise OSError("synthetic launch failure; no child started")

        with patch("maintenance.detector_validation.collect.os.getpid", return_value=123), patch(
            "maintenance.detector_validation.collect.subprocess.Popen", side_effect=inspect_launch
        ):
            receipt, _ = run_worker(Mock(), ["python"], b"", {}, "synthetic", 120, 180)
        self.assertEqual(captured, [123])
        self.assertIsNotNone(receipt["limit_failure"])

    def test_expected_parent_admitted(self):
        libc = Mock()
        libc.prctl.return_value = 0
        with patch("maintenance.detector_validation.collect.os.getppid", return_value=123), patch(
            "maintenance.detector_validation.collect.ctypes.CDLL", return_value=libc
        ), patch("maintenance.detector_validation.collect.resource.setrlimit") as limits:
            child_limits(120, expected_parent=123)
            self.assertEqual(limits.call_count, 5)


class WorkflowParentIdentityTests(unittest.TestCase):
    def test_spawn_rejects_adoptive_noninit_parent(self):
        from maintenance.detector_validation import run

        libc = Mock()
        libc.prctl.return_value = 0

        def invoke_preexec(*args, **kwargs):
            kwargs["preexec_fn"]()

        with patch.object(run.os, "getpid", return_value=123), patch.object(
            run.os, "getppid", return_value=456
        ), patch.object(run.subprocess, "Popen", side_effect=invoke_preexec), patch.object(
            run.ctypes, "CDLL", return_value=libc
        ), patch.object(
            run.resource, "setrlimit"
        ) as limits:
            with self.assertRaisesRegex(RuntimeError, "parent vanished"):
                run.spawn(["python"], {}, 10, 1024)
            libc.prctl.assert_not_called()
            limits.assert_not_called()

    def test_spawn_preserves_parent_identity_through_setup(self):
        from maintenance.detector_validation import run

        for parents, admitted in (([123, 123], True), ([123, 456], False)):
            libc = Mock()
            libc.prctl.return_value = 0

            def invoke_preexec(*args, **kwargs):
                kwargs["preexec_fn"]()

            with patch.object(run.os, "getpid", return_value=123), patch.object(
                run.os, "getppid", side_effect=parents
            ), patch.object(run.subprocess, "Popen", side_effect=invoke_preexec), patch.object(
                run.ctypes, "CDLL", return_value=libc
            ), patch.object(
                run.resource, "setrlimit"
            ) as limits:
                if admitted:
                    run.spawn(["python"], {}, 10, 1024, gated=True)
                    self.assertEqual(limits.call_count, 5)
                else:
                    with self.assertRaisesRegex(RuntimeError, "parent vanished"):
                        run.spawn(["python"], {}, 10, 1024)
                    limits.assert_not_called()


class GitParentIdentityTests(unittest.TestCase):
    def test_git_binds_prefork_parent_and_rejects_adoption(self):
        from maintenance.detector_validation import provenance

        for parents, admitted in (([123, 123], True), ([456], False), ([123, 456], False)):
            libc = Mock()
            libc.prctl.return_value = 0

            def invoke_preexec(*args, **kwargs):
                self.assertEqual(kwargs["timeout"], 5)
                kwargs["preexec_fn"]()
                return b"synthetic Git result"

            with patch.object(provenance.os, "getpid", return_value=123), patch.object(
                provenance.os, "getppid", side_effect=parents
            ), patch.object(provenance.os, "_exit", side_effect=SystemExit) as stop, patch.object(
                provenance.subprocess, "check_output", side_effect=invoke_preexec
            ), patch.object(
                provenance.ctypes, "CDLL", return_value=libc
            ), patch.object(
                provenance.resource, "setrlimit"
            ) as limits:
                if admitted:
                    self.assertEqual(
                        provenance._git(["rev-parse", "HEAD"]), b"synthetic Git result"
                    )
                    limits.assert_called_once_with(provenance.resource.RLIMIT_CPU, (1, 1))
                else:
                    with self.assertRaises(SystemExit):
                        provenance._git(["rev-parse", "HEAD"])
                    stop.assert_called_once_with(125)
                    limits.assert_not_called()
