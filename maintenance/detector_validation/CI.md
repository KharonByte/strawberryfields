# Repository-owned detector checks

`Detector harness checks` runs on pull requests and pushes to `master`. It runs
stdlib adversarial tests and the supervised synthetic smoke, without detector
sampling. Its green result is **harness evidence only**. The original eight-row
operational profile remains a separate plumbing check; neither establishes accuracy.

`Reviewed detector accuracy study` is a separate, manual workflow. No approved
study release is included. Adding these files does not authorize another full
experiment, change an existing frozen study, or establish that detectors pass.
The existing `fresh` and historical entry points remain unchanged.

## Why fixed shards

A serial full revision has 680 cold workers and is expected to take roughly 8–12
hours from prior operational observations, exceeding the
[GitHub-hosted six-hour job limit](https://docs.github.com/en/actions/reference/limits).
The public repository's existing runs use standard GitHub Actions runners.
Account-specific Actions/storage settings were not readable through the available
integration (403); free storage or unlimited concurrency must not be assumed.

The new design uses 17 shards of four complete cases, at most four concurrent jobs,
with no fail-fast cancellation. Each case retains all ten preregistered seeds and
100,000 rows. Partitioning changes scheduling only. Every evaluator receives the
original statistics: 920 cell/event comparisons, 68 joint-TV criteria, 34 tail
criteria, and alpha allocation 0.004/0.003/0.003. The total is 1,022 criteria with
simultaneous 99% coverage **within each revision**. This does not imply a 99%
two-revision joint guarantee. The original PNR 3.6%, threshold 3%, and tail 0.003
acceptance criteria remain unchanged; PNR's bound includes the existing tail term.

Per shard collection limits are 2,500 family CPU seconds, 3,300 wall seconds,
40 MiB raw container capacity, and the existing 14 GiB per-process address-space
limit. Each worker reserves 120 CPU/180 wall seconds plus finalization headroom.
Seventeen shard allowances total 42,500 CPU seconds, 56,100 summed wall seconds,
and 680 MiB raw capacity per revision, within the original scientific collection
allocation. Setup, dependency installation, analysis and transfers are separate
hosted-job overhead and must be included in the release's enclosing budget.
Shard jobs allow 75 minutes including setup/upload; collection has a 58-minute
outer timeout. Aggregation allows 40 minutes. These are bounded allowances, not
measured full-study completion guarantees or exact kernel CPU quotas.

The raw pair occupies up to 1,360 MiB before compression, plus reports and overhead.
A revision aggregator downloads up to 680 MiB. Dependencies and runner OS consume
additional disk. Compression may reduce upload use but is not an allocation promise.
At approximately 40 seconds per cold worker, four-case shards imply about 27 minutes
of collection and five scheduling waves per revision, excluding queuing/setup.
The workflow is designed for that scale; full hosted collection remains untested.

## Prospective review and dispatch

1. Use a new release identity and budget; preserve every prior study and failed
   dispatch. Do not copy a previous release to reroll seeds or pool partial data.
2. From the intended suite checkout, create a pending template outside the source:

   ```sh
   python -B -m maintenance.detector_validation.ci draft \
     --base FULL_BASE_COMMIT --candidate FULL_CANDIDATE_COMMIT \
     --out /tmp/study-release.json
   ```

3. Fill the enclosing `budget_scope` (including prior evidence obligations) and
   `archive_plan` (durable destination, custodian and expiry deadline). Each of
   three genuinely independent QA, Scientific and Security reviewers assesses
   the exact proposed policy. Each supplies their own identity, evidence locator,
   decision and `binding_sha256`: SHA-256 of canonical `encoded()` JSON with the
   `reviews` field omitted. Changing any bound field invalidates every old review.
   The command creates **pending templates only**. No workflow manufactures approvals.
   Reviewer strings/digests provide traceability, not identity authentication;
   authentic reviewer evidence and the owner's merge are external trust requirements.
4. Commit the reviewed release under
   `maintenance/detector_validation/releases/<identity>.json`, in an owner-reviewed
   PR. It binds immutable base/candidate commits, every harness/test file, workflow,
   requirement file, canonical profile, seed partition and resource limits. It
   binds suite content rather than its enclosing commit to avoid a self-hash cycle.
   Changes to these files require fresh reviews. Do not change simulator behavior
   to clear a validation failure.
5. After the owner merges the policy and workflows, verify the complete release
   file SHA-256 independently and dispatch `detector-study.yml` from the default
   branch with that path and digest. Full studies cannot be dispatched from a PR
   branch. No workflow changes branch protections or auto-merges anything.

Admission uses read-only Actions history, rejects repeated dispatches of the same
release, and rejects every `GITHUB_RUN_ATTEMPT` other than 1. History must be
complete within 1,000 entries, include the current run, and have no duplicate IDs.
Unknown/changing history fails closed and the rejected run is retained. Release-key
concurrency avoids overlapping execution. GitHub may replace queued pending runs;
`cancel-in-progress: false` protects the active run, not every queued dispatch.
Deleting run history defeats a history-based single-use check. The owner/custodian
must retain a durable release/run registry and must not reapprove equivalent
releases as retries. No automatic recovery, continuation or sampling retry exists.

Each job makes an isolated detached checkout of the exact simulator commit and
overlays only reviewed harness/workflow/requirement files. Dependencies are installed
in an external virtual environment. Checkout credentials are not persisted; jobs
use read-only permissions, no secrets, no privileged PR triggers and no cache shared
with untrusted PR code. Python 3.12.3, Linux x86-64, exact installed distribution
content hashes and numerical thread policy remain mandatory. Per-job absolute paths
are retained. Aggregation translates only verified source/runtime location prefixes;
all imported file hashes and execution-policy checks remain intact. Runtime content
must match across shards and across base/candidate. A mismatch fails; normalization
never removes package/file hashes. Cross-runner full-runtime equivalence is a
prospective requirement, not an already demonstrated full collection result.

## Verdicts, failures and preservation

Base aggregation recomputes from raw arrays, checks every raw member hash and exact
case/chunk/seed/source/runtime/release/run binding, and rejects missing, extra,
duplicate or mismatched evidence. It preserves the full global statistical family.
The candidate begins only after base integrity/completeness succeeds. A complete
base scientific FAIL still permits candidate collection, so a real defect does not
silently prevent comparison. An incomplete base blocks the candidate.

The final `detector-accuracy` job separately requires complete evidence and checks
actual scientific verdicts. Scientific FAIL/INCOMPLETE returns a failing exit; it is
never converted to green accuracy. Missing input creates an explicit incomplete
receipt when the controller can run. Known failures remain FAIL even when other
shards are missing. Invalid shards cannot contribute acceptance evidence.

Uploads run with `always()` and unique run/attempt/revision/shard names. They retain
raw containers, partial checkpoints, command/exit/stdout/stderr records, source and
runtime identities, observed usage and SHA-256 inventories. Downloaded artifacts
stay in separate directories; aggregate reports are recomputed, not trusted as
substitutes for raw data. Admission/rejection, per-revision reports and comparison
receipts are separate artifacts. Abrupt runner loss, job timeout, upload service
failure or checkout failure can prevent final receipts/uploads; that is missing
evidence and blocks acceptance. No recovery promise can make lost bytes permanent.

**Artifacts expire after 30 days** (or earlier deletion/service loss). GitHub Actions
is not permanent archival storage. Before expiry, the named custodian must export
all raw and partial artifacts, reviews, exact release/suite sources and reports to
the approved durable destination, verify checksums and retain a compact receipt with
run URLs, artifact IDs/digests, source commits and archive location. This workflow
does not provision external storage or publish scientific results/releases.

## Verification scope

Run the same no-sampling checks locally:

```sh
python -B -m unittest discover -s maintenance/detector_validation/tests -v
python -B -m maintenance.detector_validation.workflow smoke --out /tmp/new-smoke.json
```

The raw transport fixtures exercise real bounded containers, checksum verification,
raw parsing, case/seed/command bindings and failure aggregation. They explicitly
mock import-metadata validation and scoring to isolate orchestration. They do not
qualify scientific distributions or substitute for a full runtime study. Separate
synthetic controls exercise the real analytic evaluator and subprocess supervision.

Initial Python 3.12.3 local checks completed 13 adversarial tests in 3.34 seconds.
Synthetic smoke used 1.15 wall/0.88 process CPU seconds. A separate no-sampling
benchmark of all 68 analytic evaluator calls used 0.95 wall/0.95 CPU seconds;
that excludes raw parsing, installation, transfer and runtime hashing. Hashing the
73 pinned installed distributions took 8.39 seconds locally. These are measured
component costs, not extrapolated scientific success or hosted throughput claims.

Security review also identified upstream action dependency advisories affecting
the pinned action releases. Existing use elsewhere does not approve these new
workflow operations. Retain the separate Security review and owner disposition;
a green hosted check cannot waive that readiness blocker.
