# Software-only coverage audit

## Issue #26 reconciliation, 22 November 2026

This is an inventory tranche, not completion of [#26](https://github.com/KharonByte/strawberryfields/issues/26)
or numerical acceptance. No exclusion, simulator implementation, resource setting
or numerical tolerance changes. The sections below this checkpoint retain the
historical #25 audit and its evidence limits.

Baseline: `d2e18ea769fa47bc4f438e8cab138f16208c578f`, after #49. The exact
[push run 44964564817](https://github.com/KharonByte/strawberryfields/actions/runs/44964564817)
passed all 15 configurations and `core-tests`. All 15 retained archives were
downloaded, checked against GitHub's SHA-256 digests, and their JUnit reports
rechecked with the unchanged coverage guard. These are retained hosted results,
not newly executed local numerical tests. [The receipt](skip-audit-baseline.json)
records archive/report hashes, outcome-identity hashes and coverage totals.

| Profile, on each of Linux, Windows and macOS | Passed | Skipped | Xfail | Required checks |
| --- | ---: | ---: | ---: | ---: |
| All backends | 13,269 | 2,990 | 8 | 3,532 |
| TF batch pure | 1,111 | 274 | 0 | 418 |
| TF batch mixed | 1,106 | 279 | 0 | 417 |
| Without TensorFlow | 10,819 | 2,275 | 8 | 2,657 |
| Without TensorFlow/cloud | 10,804 | 2,276 | 8 | 2,657 |

The sorted `(classname, name, outcome)` sets agree across the three operating
systems for each profile, with no duplicate identities. Branch coverage differs
by one covered branch in Linux TF mixed (0.2925 versus 0.2921); identity equality
does not imply identical executed branches. These reports cover Python 3.12.3,
the published pinned CPU environments and seed 42. They do not establish
assertion quality or independent correctness merely by passing.

### Historical 184 and the 72-row delta

The historical follow-up categories sum to **184 = 81 + 91 + 12**. Their original
row-to-category mapping was not retained. [The named-family ledger](skip-audit-families.json)
records current identities for the described families, but is deliberately **not
claimed to partition those 184 rows**. Reconstructing or replacing that missing
mapping with individually reviewed classifications remains unfinished.

The added 72 unbatched exclusions are the exact Cartesian set introduced by
[#44](https://github.com/KharonByte/strawberryfields/pull/44):
`TestBoundedGaussianGates.test_{Pgate,CXgate,CZgate}` × strengths
`{-0.25,0.0,0.25}` × hbar `{1.0,2.0}` × storage starts `{True,False}` ×
`{gaussian,bosonic}`, at cutoff 20. Their Fock/TF backend marker excludes these
Gaussian/Bosonic combinations. Every expected identity is present and skipped.
They are unrelated to the historical 72 Bosonic thermal-continuation rows.
[#47](https://github.com/KharonByte/strawberryfields/pull/47) already reported
2,990 unbatched skips; this increase was not introduced by #49.

Across added test modules since #25 these are the only new skipped/xfail rows in
the three named profiles. Source comparison found no change to the original
exclusion-bearing modules, shared fixtures or selection/dependency configuration;
changes to pre-existing subgraph/guard tests remove no exclusion path. This
supports the added-source explanation. Without the old raw JUnit, exact historical
outcome-set equality and all possible runtime offsets remain unverified.

Added bounded P/CX/CZ, distinct number-state moments, nonzero photon variance,
and correlated Gaussian preparation tests provide additional guarded coverage.
They do not remove the original high-squeezing/random-input exclusions or repair
every legacy assertion. In particular, the legacy photon-statistics comparisons
still permit scalar broadcasting; separate shape/distinct-member checks must
not be described as a rewrite of those assertions.

The original #24 head `3c32e19fd7f7db735bc1db6a88ac6cd3075b2331` is tree-identical
to merge `1155e8c078e4b41e123f84c015dd72d29e89a3a0`. #25 was based on that merge;
head `4b116dc4b6a167591e7b4911fb85e1de91621c71` is tree-identical to merge
`3695e539121be0bc708f6a883905ceb1c356affc`. Their fixtures, tests and guards are
reused. Historical body-reported test refs are not substituted for these merged
contents or for the exact baseline run above.

### Classification criteria and remaining work

Classify each identity only after inspecting the applicable API/compiler contract:

- **Unsupported contract:** cite the unavailable operation or representation;
  a reason string or inherited marker is insufficient.
- **Precondition/dependency:** name the input or installed-dependency condition,
  with its corresponding supported profile/control.
- **Resource-sensitive:** calculate or measure the allocation, retain per-test
  limits, and investigate a bounded independent oracle before enabling it.
- **Supported defect:** retain an actual reproducer and exact environment,
  distinct from an unsupported feature. Historical observations stay historical.
- **Assertion/coverage gap:** the path may execute, but independent expectations,
  full shapes or every batch member are not established.
- **Unresolved:** missing contracts, identity mapping, qualification or evidence
  cannot be converted into support merely to reduce skip counts.

No new numerical decision is made here. Enabling numerical cases still requires
qualified independent assessment and expectations fixed before execution, with
every original tolerance and both TF batch/resource constraints preserved.
The old resource estimate for a six-mode dense state remains a lower-bound
allocation calculation, not a measured execution budget.

Bosonic zero-mode insertion and registration of multiple inserted modes relate
to existing [draft #50](https://github.com/KharonByte/strawberryfields/pull/50),
inspected at `1e38992363aef22c1e030edecb80e84e61deb8c1`. It is unmerged and is
not treated as a baseline fix. Empty-system density conversion and final-mode
deletion/fidelity are separate questions outside that draft.

Source inventory also found `BackendOnly.test_squeeze_variance` in
`tests/integration/test_hbar_integration.py`: its class name prevents collection.
It is neither a runtime skip nor a deselected test. Missing TensorFlow removes
fixture parametrizations and skips whole modules; missing cloud skips the remote
API module. A complete audit must include these collection omissions as well
as the recorded runtime rows, semantic duplicate parameter tuples, batch-member
assertions and the eight existing expected failures. The named family ledger
does not close any of those obligations.

### Local TDM contract diagnosis

At `26f674a183eeee8f01d6267648d72b16a1361a41`, the single skipped Borealis
`test_compile_with_create_program` row is a local contract investigation, not
evidence of a hardware requirement. A bounded execution of its original body
with NumPy seed 42 first fails range validation for `r0`; the separately checked
`loop0_phase=4` also exceeds its allowed range. The fixture additionally mixes
loop-array names 1–3 with scalar/range names 0–2, rank-one inputs and scalar
placeholders for array values. These are distinct from the public API failure.

With Blackbird 0.5.0, a two-bin control using distinct rank-two array bindings
and a separately named scalar preserves every expected value and position.
Rank-one input is rejected; using the same keyword as an array and a bare scalar
is also rejected. A corrected 259-bin local fixture binds successfully and its
explicit `compile(device=...)` control returns the expected distinct compiled
program. The public `create_program` path on that same fixture fails from fresh
compiler state because it supplies no device layout. An explicitly warmed
diagnostic instead reaches a missing-device-certificate error. Neither is a
successful public compilation. A separate supported Xcov control confirms that
`create_program` returns the original program rather than the linked compiled
copy. The skipped row remains excluded; production API changes, phase-compensation
accuracy and remote hardware execution are outside this audit tranche.

The required-case guard also rejects missing or duplicate consumed setup fields,
including identical duplicates. A conflicting value must not become acceptable
merely because the correct value appears last in JUnit properties. Regression
controls cover both duplicate orders in the unbatched and both TF batch profiles;
unrelated repeated note properties remain allowed. The retained passing reports
contained no duplicates in these configuration fields, so this guard repair
does not revise their numerical outcomes or relax any test criterion.

### Reproducible observation ledger

The CI test command now loads `maintenance.skip_audit`. Its JSONL artifact records
pre-selection identities/parameters/markers, deselection, collection skips,
setup/call/teardown reports and unmodified user properties. It records only an
allowlist of environment settings. A missing final event or unsuccessful exit
is incomplete evidence. Requested profile/environment settings are distinct from
reported setup/state settings; absent settings remain unknown. Objects whose
parameters cannot be serialized are explicitly labelled, not silently treated
as semantic duplicates.

`maintenance/summarize_skip_audit.py` reconciles the ledger with JUnit, rejecting
missing/duplicate identities, inconsistent selection, failed execution and
contradictory runtime outcomes. It leaves support classification **unreviewed**.
Collection-only ledgers can be joined to externally verified exact-revision
JUnit; this is labelled retained runtime evidence rather than a new local run.
An externally supplied revision label is not cryptographic provenance.

Example for the published TF pure profile, from the repository root:

```bash
env -u BATCHSIZE BATCHED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
  TF_NUM_INTEROP_THREADS=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=-1 \
  SYMPY_USE_CACHE=no timeout --signal=TERM --kill-after=10s 900s \
  .venv/bin/python -m pytest tests -q --tb=short -p no:warnings \
  --randomly-seed=42 -m 'tf and pure' -p maintenance.skip_audit \
  --skip-audit=maintenance/artifacts/pure-ledger.jsonl \
  --audit-profile=tensorflow-batched-pure \
  --junitxml=maintenance/artifacts/pure.xml
.venv/bin/python maintenance/summarize_skip_audit.py \
  maintenance/artifacts/pure-ledger.jsonl maintenance/artifacts/pure.xml \
  --report-revision="$(git rev-parse HEAD)" \
  --output=maintenance/artifacts/pure-summary.json
```

Use `BATCHED=0` and `-m ''` for all backends, or `BATCHED=1` and `-m 'tf and mixed'`
for mixed batching. Explicitly clear `BATCHSIZE`, which overrides `BATCHED=0`.
For collection-only reconciliation add `--collect-only` and use a retained,
verified JUnit report; collection alone is not a passing numerical baseline.
The 900-second local checkpoint is operational, not a new support limit; an
interrupted run remains incomplete. CI retains its existing 120-minute job limit.

## Evidence scope

The numerical error values and diagnostic outcomes below are historical audit
reports, not newly reproduced results. Their complete original input arrays,
environment receipts and execution logs are not retained with this document,
so they remain unverified here. In particular, random-input covariance maxima
must not be interpreted as bounds over all inputs. Fixed cutoffs and passing
assertions do not establish convergence or general scientific accuracy. The
maintained tests and guard specify the current checks; validating a quoted
measurement requires its exact inputs, source/environment versions and retained
results. This documentation correction changes no test or acceptance threshold.

## Enabled configurations

The first 704 enabled configurations comprise 584 Gaussian/Bosonic normalization
checks, 60 Bosonic squeezed-state parity checks, and 60 TensorFlow/edge cases.
They need no physical quantum hardware. Original numerical tolerances are
preserved; the new Bosonic parity assertion checks odd-index density entries
at cutoff 10 against `100 * float64.eps`, with zero relative tolerance. This
is a regression threshold, not a proven general bound on floating-point
cancellation or conversion error. Other backends retain exact-zero checks.

Another **42 configurations** are enabled: 18 TF batch channel-reference
comparisons, 16 Gaussian/Bosonic utility-state density comparisons, two Gaussian
mode-addition checks and six cutoff-aware Fock/TF vacuum-density checks. Channel
extraction still uses Fock; TF independently executes the reference circuit.

Another **15 TensorFlow batch gradients** are enabled: eight pure and seven
mixed. Analytic values and full Jacobians cover distinct batch members and
zero cross-member derivatives, at the existing cutoff and tolerance. Mixed
two-mode tests prepare density matrices and verify their storage. The coherent
ket gradient retains its mixed-storage exclusion.

Another **26 state-decomposition configurations** are enabled: 18 unbatched
and four in each TF batch profile. Finite-density moments and squeezed-state
projections preserve the covariance, mean, fidelity and instruction-count
assertions. They check batch shapes and normalization without rescaling away
cutoff loss; analytic controls cover complex correlations and the cutoff boundary.

Another **38 configurations** are enabled: six thermal decompositions, 30
passive-optics decompositions and two Gaussian rotated-squeezed projections.
Thermal tests use cutoff 12; the five passive-optics tests use cutoff 16. All
other tests retain their own cutoff, with the default still 6. The historical audit reported covariance errors of at most 0.00000653 and
0.000124 respectively for its sampled inputs, below the unchanged 0.001
tolerance. These reported maxima are not established bounds for the randomly
generated inputs in later runs. Passive tests check the analytic initial
and final covariance as well as propagation of the simulated covariance.
Rotated-squeezed references use phase `2 * phi` and check complex density entries,
normalization and fidelity. Separate low-energy TF controls cover distinct batch
inputs and actual pure/mixed storage for all five passive decompositions.

Another **78 Gaussian-utility moment comparisons** are enabled: 52 unbatched
Fock/TF cases and 13 in each TF batch profile. The existing Gaussian/Bosonic
comparisons remain. Vacuum and coherent states retain cutoff 6; the two squeezed
state tests use cutoff 8 with their original inputs and 0.001 tolerance. The historical audit reported 12 cases exceeding that tolerance at cutoff 6,
a maximum covariance error of 0.0000617 at cutoff 8, and 0.00000353 at cutoff
10. These measurements remain subject to the evidence limits above. Independent controls
check units, complex moments, normalization and distinct TF batch members.
The guard requires every mesh value and distinct utility parameter tuples,
so pytest's numbered IDs for duplicated values cannot replace missing coverage.

Another **two Fock graph-embedding configurations** are enabled, covering pure
and mixed storage. The two-mode graph and photon-number tolerance are unchanged.
Finite-density moments use cutoff 48. The historical audit reported total
photon-number errors of 0.0000146 at cutoff 40, 0.000000985 at 48 and
0.0000000660 at 56; those measurements have not been independently reproduced
for this documentation correction. Regression
controls reject biased photon counts, lost normalization, incorrect units,
displacement and malformed densities. The guard requires both storage modes
and the existing Gaussian controls at the explicit cutoff.

Another **one pure-Fock three-mode graph check** is enabled at cutoff 72.
A ket-only moment adapter avoids constructing the full density matrix and
retains the original complex graph, matrix-block assertions and tolerance.
Independent covariance, zero-mean and normalization checks prevent a correct
graph ratio from hiding truncation loss. The historical audit reported covariance errors of 0.0001253 at cutoff 72
and 0.0000351 at 80. These are unverified reported observations, not a
convergence guarantee. Regression controls cover boundary terms,
complex phases, units, mode order, storage and malformed states. The guard
requires pure Fock and both Gaussian controls, with no new TF requirements.
Mixed Fock remains excluded before execution: its density matrix at cutoff 72
alone would require about 2,076 GiB. No simulator code or default cutoff changes.

The historical #25 inventory contained **2,918 unbatched skips, 274 TensorFlow pure
batch skips and 279 mixed batch skips**. These are configuration rows,
not distinct unsupported features. Eight existing expected
failures are separate from these totals.

| Remaining classification | Configurations |
| --- | ---: |
| Backend API or compiler restriction | 1,982 |
| Representation-specific or duplicate contract | 1,245 |
| Parameter or execution precondition | 59 |
| Correct dependency-profile exclusion | 1 |
| Investigated contracts still requiring follow-up test work (below) | 81 |
| Families blocked by reproduced failures, accuracy gaps or empty-state questions | 91 |
| Resource-sensitive cases with a Gaussian-only oracle | 12 |

This is a source-contract audit with targeted executions, not a claim that every
incompatible case was forcibly run. The simulator source was unchanged. Hosted
JUnit artifacts retain every case's outcome and skip reason; the coverage guard
requires the newly enabled configurations to execute and pass.

## Investigation of the other 167 configurations

All 167 original configurations were executed diagnostically with their backend
exclusions bypassed. They produced 143 failures, 15 passes and nine remaining
pure-storage skips. Nine of those passes executed no numerical assertions;
the other six used incomplete batch indexing. These outcomes do not certify
support. Independent adapters were then tested without changing simulator code
or relaxing the original tolerances:

- **15 batch gradients, now enabled:** maintained tests check analytic per-member
  Jacobians and explicit cross-member derivatives with distinct parameters.
  Thermal preparation produces mixed storage even when the engine starts in
  pure mode. Regression tests reject summed, disconnected, duplicated and
  cross-coupled derivatives; the CI guard requires every supported configuration.
- **66 state-decomposition cases, 34 now enabled:** maintained finite-density
  adapters cover the verified states at explicit cutoffs. Twenty Bosonic cases
  execute native Gaussian preparation, so their decomposed-instruction-count
  assertion does not apply. Twelve other finite-cutoff cases still need accuracy
  and resource budgets. The two Gaussian rotated-squeezed cases now use the
  corrected ket phase and full complex-density assertions; the old fidelity-only
  check accepted a density error of 0.0162.
- **68 phase-space cases, 33 now enabled:** passive transformations and four
  interferometer meshes use finite-density moments at cutoff 16. Cutoff 12 can
  pass the comparison between two simulated covariances while missing the
  independent analytic answer; the maintained tests require both. Random active
  transforms, other graph configurations, teleportation and 500-shot cloning still need
  separate precision/statistical budgets. Matching clone marginals alone hides
  severe normalization loss (observed traces 0.2200 to 0.8999); those six
  configurations remain excluded pending convergence checks. Two Fock graph
  photon-number checks now use cutoff 48; the four TF variants remain excluded
  because of the separate precision failure below. Bosonic rejects
  `GraphEmbed` at compilation in two configurations. The pure-Fock three-mode
  graph check now uses direct ket moments at cutoff 72.
- **18 P/CX/CZ gate cases:** the original assertions have no Fock/TF branch.
  Full moment comparisons expose severe truncation at squeezing 3 and cutoff 6;
  removing the skip alone gives false confidence. Meaningful finite-Fock
  fixtures must retain the Gaussian/Bosonic high-squeezing checks separately.
  Bounded-input Fock and mixed/batched TF controls pass at cutoff 12 with the
  original 0.001 tolerance.

The remaining 81 diagnostic configurations need follow-up tests, coverage
guards and cross-platform validation. Existing batch photon
statistics also need stronger shape and independent-member assertions.

The general squeezed-state oracle's zero-squeezing exclusion is already covered
by `test_displaced_squeezed_fock_no_squeezing`; it is not a missing physical limit.
The missing-TensorFlow import test belongs to the no-TensorFlow CI profile.

## Observed software issues

- **TensorFlow graph precision:** the two-mode graph photon-number check passes
  at cutoff 40 but misses its original tolerance at 48 using default complex64,
  with an error of about 0.000672. All four TF variants pass with explicit
  complex128 at cutoffs 48 and 56. This does not resolve the default-precision
  failure; those configurations stay excluded and are not counted as restored.
  For the three-mode graph, pure TF (including batching) passes at cutoff 72
  but covariance error grows to about 0.01185 at 80, above its 0.001 tolerance.
  Those exclusions remain; passing a lower cutoff alone does not certify support.
- **Bosonic continuation:** displacement 0.3 followed by a separate identity-loss
  program changes mean photon number from 0.09 to zero. Gaussian, Fock and TF
  controls preserve it. This blocks 72 thermal-moment configurations.
- **Mode boundaries:** Bosonic `add_mode(0)` raises a type error. Empty-system
  density conversion raises a Walrus/Numba error for Gaussian and returns zero
  for Bosonic in the finite-density probe. Fock vacuum fidelity after deleting
  the last mode raises an `einsum` error. These need API/normalization diagnosis.
- **Bosonic Fock/cat accuracy:** full-density utility comparisons at cutoff 6
  exceed the original 0.001 tolerance by about 0.00749 and 0.43438 respectively.
  Fock preparation is approximate; its convergence budget must be assessed.
  The two boson-sampling reference cases also exceed their existing tolerance.
- **TDM template construction:** the original skipped test supplies an out-of-range
  phase. Valid zero-valued inputs separately pass parameter validation but fail
  template expansion on array dimensions or an unresolved `s_0_0`. No device or
  remote service is involved in these failures.
- **State accessor contracts:** 38 excluded configurations request methods absent
  from that state's representation. The TF weights probes also hit an integer
  coherent-amplitude dtype error before the unavailable weights accessor.
  Non-vacuum TF fidelity remains a separate numerical investigation.

These failures are not hardware limitations. This coverage change does not fix
them or relax their numerical requirements.

## Resource-sensitive exclusions

Twelve configurations (eight unbatched, four TF batched) target six-mode graph
embedding tests whose `Amat()` assertions are specific to the Gaussian circuit.
They were not force-run. At cutoff 6, a dense mixed state alone contains
`6**12` entries: approximately 32.4 GiB for complex128, or 16.2 GiB for complex64
before batching and working memory. A suitable backend-independent oracle is
needed before deciding whether any larger machine is useful.

No remaining case in this inventory requires buying a physical quantum device.
Completing the software-only coverage work remains a prerequisite for considering
library logic or numerical changes for merge.
