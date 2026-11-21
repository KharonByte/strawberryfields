"""Shared parsing mechanics preserve explicit per-format bounds."""

import tempfile
from pathlib import Path
import unittest

from maintenance.detector_validation import provenance, reanalyse, run, storage


class JsonPolicyTests(unittest.TestCase):
    def test_public_readers_reject_ambiguous_or_nonfinite_json(self):
        for payload in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}'):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "input.json"
                path.write_bytes(payload)
                for reader in (
                    provenance.read_json,
                    reanalyse.read_json,
                    lambda p: run.bounded_json(p, 4096),
                ):
                    with self.assertRaises(ValueError):
                        reader(path)

    def test_historical_parser_bounds_preserved(self):
        depth31 = b"[" * 31 + b"0" + b"]" * 31
        provenance.parse_json(depth31)
        with self.assertRaises(ValueError):
            reanalyse.parse_json(depth31)
        many = b"[" + b"0," * 10000 + b"0]"
        reanalyse.parse_json(many)
        with self.assertRaises(ValueError):
            provenance.parse_json(many)

    def test_encoding_identity_preserved(self):
        self.assertIs(run.encoded, storage.encoded)
        self.assertEqual(
            run.encoded({"z": [1, False], "a": "value"}), b'{"a":"value","z":[1,false]}\n'
        )
