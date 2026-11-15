# Portable fresh experiments

This path creates a new experiment. It is not continuation, recovery or acceptance
of any earlier experiment. Prior FAIL/INCOMPLETE results remain unchanged; do not
pool rows or claim that a new identity restores another experiment's resource
budget. Use the historical workflow documented in README.md for that evidence.
No private historical container is needed by this fresh path.

## Environment

Use Linux x86-64 and Python 3.12.3 in the candidate source checkout with this
validation-only patch applied. The runtime checker currently verifies all applicable
pins in the four files below, even though the operational profile samples only the
Gaussian backend. No cloud account, hardware or GPU is used.

```sh
python3.12 -c 'import sys; assert sys.version_info[:3] == (3, 12, 3)'
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install \
  -r maintenance/requirements-python312.txt \
  -r maintenance/requirements-core.txt \
  -r maintenance/requirements-cloud.txt \
  -r maintenance/requirements-tensorflow.txt
```

Do not install another Strawberry Fields distribution. Run from this checkout with
`python -B`; do not use `-O`. The worker applies CPU-only, one-thread numerical
settings. Linux process supervision is required; other platforms are unvalidated.

## Profiles

`operational` is a fixed eight-row runtime plumbing check: four direct Gaussian
vacuum PNR rows and four asymmetric complex coherent threshold rows, with fixed
seeds, reversed measured order in the threshold case, specified loss and units.
It checks shape, valid count/click domain, frozen input identity and provenance.
It contains no statistical accuracy criteria and always labels scientific accuracy
UNTESTED. No particular vacuum/click frequency is required. There are no rerolls.

`scientific` is the full original 68-case, 100000-row-per-case specification with
1022 unchanged cell/event, joint-TV and tail criteria. It is an executable profile,
not evidence of completed validation. Missing cases cannot pass; a criterion failure
must remain FAIL even if other cases pass. A new scientific experiment requires its
own justified resource allocation and independent review before collection.

## Public review and freeze procedure

The command sequence is `init`, independent review, `freeze`, `run`, `report`.
Initialization generates the manifest, source/runtime identities, local ledger and
pending review templates. It does not approve collection. A new directory is
required; the generated experiment ID and lineage must not be edited.

Each independent reviewer examines the exact manifest/profile, source, runtime,
resource accounting and expected interpretation. QA checks execution/coverage;
scientific review checks expectations and accuracy-claim scope; Security checks
execution, inputs and evidence handling. Each reviewer supplies their own completed
JSON review with a distinct identity, exact bindings, scope, decision and notes.
Pending templates are not approvals. JSON identities and hashes provide traceability,
not cryptographic proof of who reviewed them. Authenticity and authorization remain
an external responsibility; do not self-approve all three roles.

Freeze rejects pending, stale, wrong-experiment or mismatched reviews. Record the
release SHA-256 returned by freeze and supply that exact value to run and report.
Changing source, profile or bound accounting requires new reviews and a new release;
do not edit a started experiment or reuse it for a retry. The one-attempt marker
preserves failed/partial execution. New outcomes cannot overwrite historical data.

For a genuinely new independent operational budget, create `accounting.json`:

```json
{
  "schema": "detector-public-accounting-v7",
  "scope": "new_independent_budget",
  "scope_id": "replace-with-your-reviewed-budget-identity",
  "prior_cpu_seconds": 0,
  "prior_wall_seconds": 0,
  "historical_status": "not_applicable",
  "prior_accounting_status": "new_budget_no_prior_execution",
  "retained_workspace_bytes": 0,
  "activity_allowance": {
    "cpu_seconds": 300,
    "wall_seconds": 600,
    "capacity_bytes": 8388608,
    "worker_cpu_seconds": 120,
    "worker_wall_seconds": 240
  }
}
```

Zero prior use is valid only for a genuinely new budget. Set retained workspace
bytes to the actual enclosing preservation obligation, not automatically zero.
If previous work shares the budget, use `existing_budget_new_activity`, preserve
its status (`prior_results_preserved` or `FAIL_INCOMPLETE_preserved`), and record
known prior CPU/wall usage. Unknown usage must be `null` with
`prior_accounting_status: unreconciled_explicit_new_activity_allowance`; reviewers
must verify explicit authorization for that separate allowance. This declaration
is not proof of historical resource use and cannot reset it.

Initialize and have each external reviewer complete their own pending template:

```sh
python -B -m maintenance.detector_validation.fresh init \
  --profile operational --accounting accounting.json --out experiment
```

Reviewers fill `reviewer`, `decision` and `notes` in their respective
`experiment/review-qa.json`, `review-scientific.json` and `review-security.json`.
Only an actually supported review may use `decision: approved`; retain all exact
bindings and scope. Then freeze:

```sh
python -B -m maintenance.detector_validation.fresh freeze \
  --experiment experiment \
  --qa experiment/review-qa.json \
  --scientific experiment/review-scientific.json \
  --security experiment/review-security.json
```

Use the returned digest unchanged (replace the placeholder below):

```sh
python -B -m maintenance.detector_validation.fresh run \
  --experiment experiment --release-sha256 RELEASE_SHA256
python -B -m maintenance.detector_validation.fresh report \
  --experiment experiment --release-sha256 RELEASE_SHA256 --out report.json
```

For a full scientific experiment, use a different new directory and `--profile
scientific` at initialization. Its reviewed activity allowance is 43200 CPU seconds,
57600 wall seconds and 805306368 evidence bytes. Repeat independent review and
freeze for that exact scientific profile; operational approval cannot release it.
Neither this procedure nor an example JSON grants budget or execution permission.
No historical containers or service credentials are required. Review records and
freezes are generated from these public local inputs.

## Resource and result interpretation

The revised operational collection target is 300 CPU seconds, 600 wall seconds,
14 GiB address space and 8 MiB evidence capacity. Each worker is admitted only
with its complete 120 CPU-second / 240 wall-second allocation available, plus
reserved supervisor/cleanup overhead. See OPERATIONS.md for prior attempts and
the prospective diagnostic amendment. Setup, review tooling and independent
analysis are accounted separately in the enclosing allocation. Practical watchdogs
can overshoot; requested and observed usage must be reported honestly. These are
not promises of an exact OS quota. Cold compilation can exhaust the budget; retain
that failed attempt, rather than rerolling until it succeeds.

Report execution, integrity, coverage and scientific status separately. Successful
operational collection does not establish photon or threshold probability accuracy,
conditional-state behavior, physical-device accuracy or the apps route. Its raw
rows and provenance are retained for independent inspection. Full scientific
accuracy requires the original criteria, complete required evidence and independent
assessment, not a successful operational profile.

Historical scientific coverage remains incomplete, including 52 untested cases.
A successful eight-row operational run does not establish scientific accuracy.
No detector default, simulator implementation, API or supported domain is changed.
