"""Timer rejection diagnostics preserve the same single-observation admission."""

import ctypes
import os
import signal
import sys
import unittest
from unittest.mock import patch, MagicMock

from maintenance.detector_validation import supervision as s, worker


class TimerDiagnosticsTests(unittest.TestCase):
    def observe(
        self,
        *,
        before=1.0,
        after=1.000001,
        remaining=169.0,
        interval=0,
        ignored=0,
        caught=0,
        blocked=0,
        handler=signal.SIG_DFL,
    ):
        class Libc:
            def timer_gettime(self, identifier, pointer):
                state = pointer._obj
                state.value.seconds = int(remaining)
                state.value.nanoseconds = round((remaining % 1) * 1e9)
                state.interval.seconds = interval
                return 0

        fields = f"SigIgn:\t{ignored:x}\nSigCgt:\t{caught:x}\nSigBlk:\t{blocked:x}\n"
        with patch.object(s, "_WORKER_CPU_TIMER", (Libc(), ctypes.c_void_p(), 170)), patch.object(
            s.Path, "read_text", return_value=fields
        ), patch.object(
            s.time, "clock_gettime", side_effect=[before, after]
        ) as clock, patch.object(
            s.signal, "getsignal", return_value=handler
        ):
            try:
                return s._worker_cpu_timer_observation()
            finally:
                self.assertEqual(clock.call_count, 2)  # no retry or new observation

    def test_valid_single_observation_keeps_historical_record(self):
        record = self.observe()
        self.assertEqual(record["budget_seconds"], 170)
        self.assertEqual(record["remaining_seconds"], 169)
        self.assertNotIn("failed_checks", record)

    def test_each_rejection_preserves_original_operands(self):
        bit = 1 << (signal.SIGALRM - 1)
        for kwargs, expected in [
            ({"handler": signal.SIG_IGN}, "python_signal_default"),
            ({"ignored": bit}, "native_signal_default"),
            ({"caught": bit}, "native_signal_default"),
            ({"blocked": bit}, "native_signal_unblocked"),
            ({"interval": 1}, "one_shot"),
            ({"remaining": 0}, "remaining_positive"),
            ({"after": 0.9}, "clock_order"),
            ({"after": 1.02}, "observation_width"),
            ({"remaining": 169.02}, "deadline_upper"),
            ({"remaining": 168.98}, "deadline_lower"),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(
                s.WorkerTimerObservationError
            ) as ctx:
                self.observe(**kwargs)
            obs = ctx.exception.observation
            self.assertIn(expected, obs["failed_checks"])
            self.assertEqual(obs["cpu_before_seconds"], 1)
            self.assertEqual(obs["cpu_after_seconds"], kwargs.get("after", 1.000001))
            self.assertAlmostEqual(obs["remaining_seconds"], kwargs.get("remaining", 169))
            self.assertEqual(obs["native_sigblk"], kwargs.get("blocked", 0))
            self.assertEqual(obs["observation_tolerance_seconds"], 0.01)

    def test_worker_preserves_structured_failure_without_capture(self):
        self.worker_failure(False)

    def test_worker_preserves_initial_arm_failure_without_capture(self):
        self.worker_failure(True)

    def worker_failure(self, initial):
        observation = {
            "failed_checks": ["observation_width"],
            "cpu_before_seconds": 1,
            "cpu_after_seconds": 1.02,
        }
        evidence = MagicMock()
        armed = {"rejected_observations": [observation]}
        failure = s.WorkerTimerObservationError(observation)
        failure.observation_history = [observation, observation, observation]
        with patch.object(
            sys, "argv", ["worker", "--case-sha256", "0" * 64, "--chunk", "0", "--cpu-timer"]
        ), patch.dict(os.environ, {"DETECTOR_EVIDENCE_FD": "3"}), patch.object(
            s,
            "start_worker_cpu_timer",
            return_value=armed,
            side_effect=failure if initial else None,
        ), patch.object(
            s, "worker_cpu_timer_state", side_effect=s.WorkerTimerObservationError(observation)
        ), patch.object(
            worker.sys, "addaudithook"
        ), patch.object(
            worker, "EvidenceStream", return_value=evidence
        ), patch.object(
            worker, "started_metadata", return_value={}
        ), patch.object(
            worker, "preflight", return_value={}
        ), patch.object(
            worker, "capture"
        ) as capture:
            with self.assertRaises(s.WorkerTimerObservationError):
                worker.main()
        import json

        final = json.loads(evidence.emit.call_args.args[1])
        self.assertEqual(evidence.emit.call_args.args[0], 5)
        self.assertEqual(final["status"], "error")
        self.assertEqual(final["cpu_timer_validation_failure"], observation)
        capture.assert_not_called()
        if initial:
            self.assertEqual(evidence.emit.call_count, 2)
            self.assertEqual(final["cpu_timer_observation_history"], failure.observation_history)
        else:
            self.assertEqual(final["cpu_timer_armed"], armed)

    def test_initial_failure_round_trips_real_pipe_and_parser(self):
        from maintenance.detector_validation.collect import FrameParser
        import json

        observation = {
            "failed_checks": ["observation_width"],
            "cpu_before_seconds": 1,
            "cpu_after_seconds": 1.02,
        }
        error = s.WorkerTimerObservationError(observation)
        error.observation_history = [observation] * 3
        read_fd, write_fd = os.pipe()
        try:
            with patch.object(
                sys, "argv", ["worker", "--case-sha256", "0" * 64, "--chunk", "0", "--cpu-timer"]
            ), patch.dict(os.environ, {"DETECTOR_EVIDENCE_FD": str(write_fd)}), patch.object(
                s, "start_worker_cpu_timer", side_effect=error
            ):
                with self.assertRaises(s.WorkerTimerObservationError):
                    worker.main()
            os.close(write_fd)
            write_fd = None
            store = MagicMock()
            parser = FrameParser(store, "synthetic/00")
            while True:
                block = os.read(read_fd, 7)  # split both headers and payloads
                if not block:
                    break
                parser.feed(block)
            parser.finish()
            self.assertEqual(parser.seen, {1, 5})
            self.assertEqual(
                parser.final_metadata["cpu_timer_observation_history"], error.observation_history
            )
            self.assertEqual(parser.final_metadata["status"], "error")
            startup = json.loads(store.put.call_args_list[0].args[1])
            self.assertEqual(startup["status"], "startup_unqualified")
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)


class BoundedObservationTests(unittest.TestCase):
    def observation(self, t=1):
        return dict(
            schema="worker-cpu-timer-rejection-v1",
            failed_checks=["observation_width"],
            budget_seconds=170,
            cpu_before_seconds=t,
            cpu_after_seconds=t + 0.02,
            remaining_seconds=170 - t,
            interval_seconds=0,
            python_signal_default=True,
            native_sigign=0,
            native_sigcgt=0,
            native_sigblk=0,
            observation_tolerance_seconds=0.01,
        )

    def good(self, t=2):
        return dict(
            timer="CLOCK_PROCESS_CPUTIME_ID",
            signal="SIGALRM",
            budget_seconds=170,
            cpu_before_seconds=t,
            cpu_after_seconds=t + 0.001,
            remaining_seconds=170 - t,
            interval_seconds=0,
            native_default=True,
            native_unblocked=True,
            observation_tolerance_seconds=0.01,
        )

    def test_zero_one_two_rejections_then_valid(self):
        for n in range(3):
            rejected = [self.observation(1 + i * 0.03) for i in range(n)]
            with patch.object(
                s,
                "_worker_cpu_timer_observation",
                side_effect=[*(s.WorkerTimerObservationError(o) for o in rejected), self.good()],
            ) as read:
                result = s.worker_cpu_timer_state()
            self.assertEqual(read.call_count, n + 1)
            self.assertEqual(result["rejected_observations"], rejected)
            self.assertEqual(result["observation_count"], n + 1)

    def test_third_wide_observation_exhausts_bound(self):
        errors = [s.WorkerTimerObservationError(self.observation(1 + i * 0.03)) for i in range(3)]
        with patch.object(s, "_worker_cpu_timer_observation", side_effect=errors) as read:
            with self.assertRaises(s.WorkerTimerObservationError) as ctx:
                s.worker_cpu_timer_state()
        self.assertEqual(read.call_count, 3)
        self.assertEqual(ctx.exception.observation_history, [e.observation for e in errors])

    def test_other_failures_are_never_reobserved(self):
        for extra in ("deadline_upper", "native_signal_default", "one_shot"):
            obs = self.observation()
            obs["failed_checks"].append(extra)
            with patch.object(
                s, "_worker_cpu_timer_observation", side_effect=s.WorkerTimerObservationError(obs)
            ) as read:
                with self.assertRaises(s.WorkerTimerObservationError):
                    s.worker_cpu_timer_state()
                self.assertEqual(read.call_count, 1)
        with patch.object(
            s, "_worker_cpu_timer_observation", side_effect=OSError("read failed")
        ) as read:
            with self.assertRaises(OSError):
                s.worker_cpu_timer_state()
            self.assertEqual(read.call_count, 1)

    def test_report_recomputes_trace_and_trajectory(self):
        import copy
        from maintenance.detector_validation.fresh import validate_cpu_timer_records
        from maintenance.detector_validation.fresh_profile import profile

        names = (
            "cpu_timer_armed",
            "cpu_timer_before",
            "cpu_timer_before_measurement",
            "cpu_timer_after_measurement",
            "cpu_timer_after",
        )
        meta = {}
        for i, name in enumerate(names):
            meta[name] = dict(
                self.good(2 + i),
                observation_policy="three-reads-width-only-v1",
                observation_count=1,
                rejected_observations=[],
            )
        first = meta[names[0]]
        first.update(
            observation_count=3, rejected_observations=[self.observation(1), self.observation(1.03)]
        )
        limits = profile("scientific-v3")["resources"]
        validate_cpu_timer_records(meta, limits)
        bads = []
        for key, value in (
            ("native_sigblk", 1 << (signal.SIGALRM - 1)),
            ("remaining_seconds", 0),
            ("remaining_seconds", float("nan")),
            ("cpu_before_seconds", True),
            ("interval_seconds", 1),
            ("budget_seconds", 180),
            ("failed_checks", []),
            ("cpu_after_seconds", 1.001),
        ):
            bad = copy.deepcopy(meta)
            bad[names[0]]["rejected_observations"][0][key] = value
            bads.append(bad)
        bad = copy.deepcopy(meta)
        bad[names[0]]["rejected_observations"].reverse()
        bads.append(bad)
        bad = copy.deepcopy(meta)
        bad[names[0]]["rejected_observations"].pop()
        bads.append(bad)
        bad = copy.deepcopy(meta)
        del bad[names[0]]["rejected_observations"]
        bads.append(bad)
        bad = copy.deepcopy(meta)
        bad[names[0]]["rejected_observations"][0]["extra"] = 1
        bads.append(bad)
        for bad in bads:
            with self.subTest(bad=bad[names[0]]), self.assertRaises(ValueError):
                validate_cpu_timer_records(bad, limits)
