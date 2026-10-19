# Software-only coverage audit

The first 704 enabled configurations comprise 584 Gaussian/Bosonic normalization
checks, 60 Bosonic squeezed-state parity checks, and 60 TensorFlow/edge cases.
They need no physical quantum hardware. Original numerical tolerances are
preserved; the new Bosonic parity assertion explicitly bounds floating-point
cancellation without weakening the exact-zero checks for other backends.

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
other tests retain their own cutoff, with the default still 6. At these cutoffs,
the audited covariance errors were at most 0.00000653 and 0.000124 respectively,
below the unchanged 0.001 tolerance. Passive tests check the analytic initial
and final covariance as well as propagation of the simulated covariance.
Rotated-squeezed references use phase `2 * phi` and check complex density entries,
normalization and fidelity. Separate low-energy TF controls cover distinct batch
inputs and actual pure/mixed storage for all five passive decompositions.

Another **78 Gaussian-utility moment comparisons** are enabled: 52 unbatched
Fock/TF cases and 13 in each TF batch profile. The existing Gaussian/Bosonic
comparisons remain. Vacuum and coherent states retain cutoff 6; the two squeezed
state tests use cutoff 8 with their original inputs and 0.001 tolerance. At
cutoff 6, 12 cases exceed that tolerance. At cutoff 8, their maximum covariance
error is 0.0000617, falling to 0.00000353 at cutoff 10. Independent controls
check units, complex moments, normalization and distinct TF batch members.
The guard requires every mesh value and distinct utility parameter tuples,
so pytest's numbered IDs for duplicated values cannot replace missing coverage.

The remaining inventory contains **2,921 unbatched skips, 274 TensorFlow pure
batch skips and 279 mixed batch skips**. These are configuration rows,
not distinct unsupported features. Eight existing expected
failures are separate from these totals.

| Remaining classification | Configurations |
| --- | ---: |
| Backend API or compiler restriction | 1,982 |
| Representation-specific or duplicate contract | 1,245 |
| Parameter or execution precondition | 59 |
| Correct dependency-profile exclusion | 1 |
| Investigated contracts still requiring follow-up test work (below) | 88 |
| Families blocked by reproduced failures, accuracy gaps or empty-state questions | 87 |
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
- **68 phase-space cases, 30 now enabled:** passive transformations and four
  interferometer meshes use finite-density moments at cutoff 16. Cutoff 12 can
  pass the comparison between two simulated covariances while missing the
  independent analytic answer; the maintained tests require both. Random active
  transforms, graph embedding, teleportation and 500-shot cloning still need
  separate precision/statistical budgets. Matching clone marginals alone hides
  severe normalization loss (observed traces 0.2200 to 0.8999); those six
  configurations remain excluded pending convergence checks. Bosonic rejects
  `GraphEmbed` at compilation in two configurations.
- **18 P/CX/CZ gate cases:** the original assertions have no Fock/TF branch.
  Full moment comparisons expose severe truncation at squeezing 3 and cutoff 6;
  removing the skip alone gives false confidence. Meaningful finite-Fock
  fixtures must retain the Gaussian/Bosonic high-squeezing checks separately.
  Bounded-input Fock and mixed/batched TF controls pass at cutoff 12 with the
  original 0.001 tolerance.

The remaining 88 diagnostic configurations need follow-up tests, coverage
guards and cross-platform validation. Existing batch photon
statistics also need stronger shape and independent-member assertions.

The general squeezed-state oracle's zero-squeezing exclusion is already covered
by `test_displaced_squeezed_fock_no_squeezing`; it is not a missing physical limit.
The missing-TensorFlow import test belongs to the no-TensorFlow CI profile.

## Observed software issues

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
