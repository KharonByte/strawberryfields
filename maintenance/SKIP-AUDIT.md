# Software-only coverage audit

The 704 enabled configurations comprise 584 Gaussian/Bosonic normalization
checks, 60 Bosonic squeezed-state parity checks, and 60 TensorFlow/edge cases.
They need no physical quantum hardware. Original numerical tolerances are
preserved; the new Bosonic parity assertion explicitly bounds floating-point
cancellation without weakening the exact-zero checks for other backends.

The remaining inventory contains **3,039 unbatched skips, 315 TensorFlow pure
batch skips and 319 mixed batch skips**. These are configuration rows across
255 test functions, not distinct unsupported features. Eight existing expected
failures are separate from these totals.

| Remaining classification | Configurations |
| --- | ---: |
| Backend API or compiler restriction | 1,982 |
| Representation-specific or duplicate contract | 1,323 |
| Parameter or execution precondition | 59 |
| Correct dependency-profile exclusion | 1 |
| Additional adapters validated locally | 42 |
| Test adaptation or contract review still needed | 167 |
| Families blocked by reproduced failures, accuracy gaps or empty-state questions | 87 |
| Resource-sensitive cases with a Gaussian-only oracle | 12 |

This is a source-contract audit with targeted executions, not a claim that every
incompatible case was forcibly run. The simulator source was unchanged. Hosted
JUnit artifacts retain every case's outcome and skip reason; the coverage guard
requires the newly enabled configurations to execute and pass.

## Further test work

The following **42 remaining configurations** passed bounded local CPU probes
with adapted assertions. They still need maintained tests, coverage guards and
cross-platform validation before inclusion:

- 18 TensorFlow batch channel-reference comparisons, including distinct positive
  density inputs and complete batch shapes. Extraction itself uses Fock.
- 16 Gaussian/Bosonic utility-state comparisons using complete density matrices
  instead of assuming a pure-state ket representation.
- Two Gaussian mode-addition checks using finite-density traces.
- Six Fock/TF vacuum-density checks using the initialized cutoff and batch shape,
  replacing the inappropriate hard-coded 10-by-10 reference.

The other 167 open review rows cover phase-space assertions (68), state
decomposition/compiler contracts (66), batch gradients (15), and tests whose
numerical branches currently omit the selected backend (18). A forced pass in
that last group would not constitute numerical coverage. Existing batch photon
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
