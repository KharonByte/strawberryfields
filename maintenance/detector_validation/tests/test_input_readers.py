"""Parity and boundary controls for shared evidence input primitives."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from maintenance.detector_validation import fresh, reanalyse, retained, run, storage


class InputReaderTests(unittest.TestCase):
    def test_adapters_share_implementation(self):
        self.assertIs(fresh.raw_bytes, storage.read_bounded)
        self.assertIs(retained.bounded_bytes, storage.read_bounded)
        self.assertIs(reanalyse.secure_open, storage.open_regular)
        self.assertIs(run.safe_path, storage.regular_path)

    def test_bytes_and_caps_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input"
            path.write_bytes(b'{"x":1}\n')
            for read in (storage.read_bounded, fresh.raw_bytes, retained.bounded_bytes):
                self.assertEqual(read(path, 8), b'{"x":1}\n')
                with self.assertRaises(ValueError):
                    read(path, 7)
            with reanalyse.secure_open(path) as stream:
                self.assertEqual(stream.read(), path.read_bytes())
            self.assertEqual(run.bounded_json(path, 8)[0], {"x": 1})

    def test_symlinks_directories_and_fifo_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "input"
            path.write_bytes(b"ok")
            (root / "link").symlink_to(path)
            (root / "directory-link").symlink_to(root, target_is_directory=True)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            for candidate in (root, root / "link", root / "directory-link" / "input", fifo):
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    storage.read_bounded(candidate)

    def test_short_or_growing_read_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input"
            path.write_bytes(b"ok")
            with path.open("rb") as real:
                stream = Mock()
                stream.fileno.return_value = real.fileno()
                for payload in (b"o", b"overflow"):
                    stream.read.return_value = payload
                    context = Mock()
                    context.__enter__ = Mock(return_value=stream)
                    context.__exit__ = Mock(return_value=False)
                    with patch.object(storage, "open_regular", return_value=context):
                        with self.assertRaisesRegex(ValueError, "size changed"):
                            storage.read_bounded(path, 2)

    def test_launcher_report_retains_explicit_sixteen_mib_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            # Large valid JSON is not squeezed into the default metadata limit.
            path.write_text('"' + "x" * (5 * 1024**2) + '"')
            self.assertEqual(len(run.bounded_json(path, run.REPORT_BYTES)[0]), 5 * 1024**2)
            with self.assertRaises(ValueError):
                storage.read_bounded(path)
