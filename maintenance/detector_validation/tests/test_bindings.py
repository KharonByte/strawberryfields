"""Prospective identity and sampler metadata controls; no detector sampling."""

import copy
import unittest

from maintenance.detector_validation.provenance import (
    SINGLE_MODE_SOURCE,
    sampler_configuration,
    validate_sampler_configuration,
    validate_candidate_binding,
)


class BindingTests(unittest.TestCase):
    def test_explicit_revision_not_historical_default(self):
        for candidate in ("a" * 40, "b" * 40):
            validate_candidate_binding({"simulator_commit": candidate}, {"candidate": candidate})

    def test_missing_mutable_and_mismatched_revisions_fail(self):
        for candidate in (None, "master", "a" * 39, "A" * 40, 123):
            with self.assertRaises(ValueError):
                validate_candidate_binding(
                    {"simulator_commit": candidate}, {"candidate": candidate}
                )
        with self.assertRaises(ValueError):
            validate_candidate_binding({"simulator_commit": "a" * 40}, {"candidate": "b" * 40})

    def test_historical_single_mode_still_uses_inherited_sampler(self):
        config = sampler_configuration({"measured_modes": [0]}, False)
        self.assertEqual(config["pnr"]["cutoff"], 5)
        self.assertEqual(config["threshold"]["fanout"], 10)

    def test_corrected_single_mode_and_marginal(self):
        for mode in (0, 3):
            config = sampler_configuration({"measured_modes": [mode]}, True)
            self.assertEqual(config["implementation"], "single_mode_Gaussian")
            self.assertEqual(config["pnr_max_cutoff"], 512)
            self.assertEqual(config["threshold"], "analytic_vacuum_overlap")

    def test_multimode_remains_inherited_on_both_revisions(self):
        self.assertEqual(
            sampler_configuration({"measured_modes": [1, 0]}, True),
            sampler_configuration({"measured_modes": [1, 0]}, False),
        )


class SamplerProvenanceTests(unittest.TestCase):
    def test_old_and_corrected_metadata(self):
        case = {"measured_modes": [0]}
        for available in (False, True):
            source = {"files": {SINGLE_MODE_SOURCE: "a" * 64} if available else {}}
            meta = {
                "sampler_defaults": sampler_configuration(case, available),
                "imports": (
                    [{"path": "/checkout/" + SINGLE_MODE_SOURCE, "sha256": "a" * 64}]
                    if available
                    else []
                ),
            }
            validate_sampler_configuration(meta, case, source)
            wrong = copy.deepcopy(meta)
            wrong["sampler_defaults"] = sampler_configuration(case, not available)
            with self.assertRaises(ValueError):
                validate_sampler_configuration(wrong, case, source)
            if available:
                for imports in (
                    [],
                    meta["imports"] * 2,
                    [{"path": "/checkout/" + SINGLE_MODE_SOURCE, "sha256": "b" * 64}],
                ):
                    wrong = dict(meta, imports=imports)
                    with self.assertRaises(ValueError):
                        validate_sampler_configuration(wrong, case, source)
            else:
                wrong = dict(
                    meta, imports=[{"path": "/checkout/" + SINGLE_MODE_SOURCE, "sha256": "a" * 64}]
                )
                with self.assertRaises(ValueError):
                    validate_sampler_configuration(wrong, case, source)
