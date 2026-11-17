# Repeatable terminal-detector validation

This validation-only workflow does not change simulator behavior, detector defaults,
APIs or supported input domains. Run from this source checkout with Python 3.12.3
and the exact maintenance pins. Use `python -B` without `-O`; optimized Python is
rejected because it disables integrity assertions. CPU-only numerical thread flags
are fixed to one by the collector's sanitized worker environment.

Repository CI and prospective fixed-shard full studies are documented in [CI.md](CI.md).
Automatic PR checks use synthetic controls only; hosted full collection requires a separately
reviewed release and owner-merged policy. The separately reviewed local path and its
prospective resource profile are documented in [LOCAL-STUDY.md](LOCAL-STUDY.md).

Three modes separate execution checks from scientific evidence:

| Mode | Purpose | Candidate sampling |
| --- | --- | --- |
| `smoke` | Supervised synthetic workers, typed raw framing, deliberate worker failure, analytic reference controls | None |
| `retained` | Recompute the original frozen detector criteria from the preserved interrupted attempt | None |
| `full` | Execute the full frozen scientific collection after a new reviewed source/runtime/resource release | Yes; not authorized by the test-only amendment |

## Bounded operational smoke

Choose a new output path; existing outputs are never overwritten:

```sh
python -B -m maintenance.detector_validation.workflow smoke --out smoke.json
```

This launches bounded synthetic subprocesses through the collector and checks their
raw rows and failure receipts. Analytic controls cover vacuum and nonzero coherent, thermal, squeezed,
correlated two-mode and interference references as applicable,
plus threshold reference, wrong-distribution and insufficient-sample controls.
It does not exercise real probability kernels or establish detector accuracy.
Requested limits are 20 CPU seconds, 30 wall seconds, 14 GiB address space and a
1 MiB report. These are practical limits/watchdogs, not an exact OS CPU guarantee.
Reports distinguish requested from observed usage and preserve the expected planted
failure separately from an unexpected workflow failure.

## Reanalyse the retained real evidence

Set `EVIDENCE_ROOT` to the preserved task-evidence directory. It contains the
original container, frozen manifest/release and deterministic reference certificates.
Do not copy, modify or regenerate those inputs. The command is deliberately pinned
to that attempt; it verifies identities and raw member hashes before reporting.

```sh
python -B -m maintenance.detector_validation.workflow retained \
  --container "$EVIDENCE_ROOT/candidate-attempt-2/evidence.bin" \
  --manifest "$EVIDENCE_ROOT/preparation/frozen-collection-v2/manifest.json" \
  --release "$EVIDENCE_ROOT/preparation/frozen-collection-v2/release.json" \
  --release-sha256 27f8507c67063beab99af73909a18980899923d0384f7752269335a0efaa73bb \
  --references "$EVIDENCE_ROOT/preparation/deterministic-controls-final" \
  --out retained-report.json
```

This is independent reanalysis of raw data, not a sampler rerun. Its implementation
adapts the previously frozen assessor audit; copying that audit does not create a
second independent implementation. Separate QA checks use the original production
PGF/interval implementation, direct raw counting and the closed-form squeezed tail.
Requested limits are 60 CPU/120 wall seconds, 14 GiB address space and a 1 MiB report.
Retained coverage is INCOMPLETE, with 15 complete, one partial and 52 untested
cases. The complete scientific suite has not passed.

## Explicit full scientific mode

```sh
python -B -m maintenance.detector_validation.workflow full \
  --manifest frozen/manifest.json \
  --release frozen/release.json --release-sha256 RELEASE_SHA256 \
  --resource-envelope frozen/resource-envelope.json \
  --resource-envelope-sha256 ENVELOPE_SHA256 --out NEW_ATTEMPT_DIRECTORY
```

This is a real collector entry point, not a smoke alias. No usable operational
release is supplied by the test-only amendment. A flag is not approval. Before
execution, independent scientific, QA and Security reviews must bind the exact
source, runtime, unchanged manifest, resource envelope and cumulative preservation
ledger. The v6 release must identify `full_scientific_collection`, explicitly reject
a test-only amendment as authorization, and bind three distinct review records.
Historical release files cannot authorize changed code. Envelope and release
validation is in `envelope.py`, `budget.py` and `provenance.py`.

The full matrix remains 68 cases, 100,000 returned rows each, ten fixed-seed chunks
per case, and 1,022 original criteria (920 cell/event, 68 joint-TV, 34 PNR-tail).
Keep all failures and partial outputs; no rerolls, threshold changes, smaller
multiplicity or reduced-sample acceptance. Prior evidence remains counted in the
4 GiB storage ceiling. Full-mode resource settings are a prospective proposal,
not proof that all cases will finish or permission to start another attempt.
Practical watchdogs may overshoot; postflight violations fail, and requested versus
observed limits and cleanup limitations must remain visible. No OS settings change
is required by this workflow.

## Machine-readable report contract

Smoke and retained reports identify `schema: detector-test-report-v6`, mode,
`execution_status`, `scientific_verdict`, `coverage_status` and resource observations.
Smoke uses `NOT_EVALUATED` and `SYNTHETIC_ONLY`; successful smoke is never accuracy
acceptance. Retained reports enumerate all 68 `case_reports`, with row counts,
missing chunks, integrity status, coverage, scientific status, per-criterion states
and whether each criterion was actually evaluated. Scientific states are PASS,
FAIL, INCOMPLETE and UNTESTED. Partial cases do not receive full-N acceptance.
Invalid evidence cannot produce a scientific PASS. Full collection retains its
versioned container, command/exit/provenance records and reanalysis report.

Exit 0 means the requested smoke controls passed or the relevant scientific mode
passed. Exit 2 means a diagnostic scientific failure or incomplete result; it is
not an execution crash and must not be converted to a pass. Exit 1 records an
execution/input failure where a bounded receipt can be written. Signal termination
may prevent a final receipt; outer supervisors must retain exit/stdout/stderr and
partial evidence. Reports are bounded and fail closed on oversized output.

## Coverage and limits

The complete 68-case suite has not passed. Operational execution success is not
scientific acceptance. Evidence concerns terminal returned sample laws, not
physical devices or conditional-state updates.

Pinned Walrus PNR state sampling defaults to cutoff 5 and max_photons 30. Threshold
sampling also names its cap max_photons, but it counts clicks, with fanout 10 and
cutoff 1. Both wrappers retry rejected proposals without an internal attempt-count
bound; external timeouts are not probability certificates. In this one-/two-
measured-mode matrix the default cap 30 is unreachable for both families. Thus
cap-boundary numerical behavior is untested here. This source-verified limitation
is not an additional reproduced defect. Existing deterministic PNR support/cap
controls are separate control-flow evidence, not distribution-accuracy tests.
