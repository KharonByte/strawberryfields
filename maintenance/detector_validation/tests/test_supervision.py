"""Bounded synthetic child cleanup controls; no simulator imports or samples."""

import os
import selectors
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from maintenance.detector_validation.supervision import supervise, terminate_group


@unittest.skipUnless(sys.platform == "linux", "Linux pidfd supervisor")
class CleanupTests(unittest.TestCase):
    def test_term_drains_buffer_and_returns_after_reaping(self):
        for use_pipe in (False, True):
            with self.subTest(pipe=use_pipe):
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        "import os,time; os.write(1,b'readydata'); time.sleep(5)",
                    ],
                    stdout=subprocess.PIPE if use_pipe else subprocess.DEVNULL,
                    start_new_session=True,
                )
                captured = bytearray()
                callbacks = {}
                if use_pipe:
                    # The ready marker proves the synthetic child reached its sleep.
                    captured.extend(os.read(proc.stdout.fileno(), 5))
                    callbacks[proc.stdout.fileno()] = (
                        selectors.EVENT_READ,
                        lambda mask: self.consume(proc.stdout.fileno(), captured),
                    )
                try:
                    started = time.monotonic()
                    result = supervise(
                        proc,
                        callbacks,
                        started=started,
                        wall_ceiling=2,
                        cleanup_grace=0.2,
                        own_cpu_tripwire=0,
                    )
                    self.assertTrue(result["cleanup_reaped"])
                    self.assertEqual(result["exit_code"], -signal.SIGTERM)
                    self.assertLess(time.monotonic() - started, 1.5)
                    if use_pipe:
                        self.assertEqual(captured, b"readydata")
                        self.assertGreater(result["counters"]["callback_calls"], 0)
                finally:
                    if proc.returncode is None:
                        proc.kill()
                        proc.wait()
                    if proc.stdout:
                        proc.stdout.close()

    @staticmethod
    def consume(fd, captured):
        block = os.read(fd, 4096)
        captured.extend(block)
        return bool(block)

    def test_reaped_group_identifier_is_never_signalled(self):
        class Reaped:
            pid = 12345
            returncode = 0

        with patch("os.killpg") as kill:
            terminate_group(Reaped(), signal.SIGKILL)
        kill.assert_not_called()
