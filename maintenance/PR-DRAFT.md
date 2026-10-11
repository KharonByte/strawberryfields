# Fix bosonic heterodyne postselection amplitude scaling

**Destination:** `KharonByte/strawberryfields`, branch
`fix/bosonic-heterodyne-postselection` into `master`. Local draft; not submitted.

`MeasureHeterodyne(select=alpha)` reports `alpha` but the bosonic backend
conditions the state on `alpha/2`. Convert the selected coherent amplitude to
`2 * [real(alpha), imag(alpha)]` before conditioning, consistent with its
internal hbar=2 convention. Returned samples retain their documented meaning.

Add an exact two-mode conditional-state regression and correct the old
low-level test's quadrature expectation. Cover zero, real, imaginary and
complex outcomes, both measured modes, two frontend hbar conventions, the
remaining state, measured-mode reset and returned samples. Include a short
demo and a script that runs identical tests against original and patched source.

Validation on a fresh pinned Python 3.12.3 / Ubuntu 24.04 WSL2 environment:

- Same regression cases: 43 passed / 42 failed before; 85 passed after, with
  the same 32 unsupported-backend cases skipped in each run.
- Related suite: 1,203 passed, 60 skipped, 104 deselected, no failures.
- Demo amplitude error: 0.088100986 before; 2.78e-17 after. The Gaussian
  control passes in both runs and the reported measurement is unchanged.

Results and source/test hashes are recorded in `maintenance/validation.json`;
reproduction instructions and a short comparison are in `maintenance/RESULTS.md`.
The fork notice, support scope and CI configuration limit this initial effort
to the selected CPU tests and pinned Linux environment. Legacy upstream jobs
are gated to upstream.

Users compensating by doubling `select` must remove that compensation. Full
upstream CI, TensorFlow/GPU execution, other platforms, quantum hardware,
hosted CI and external review have not been validated. This is a numerical
simulator fix; no quantum hardware is required for the demonstrated behavior.

This is a PR within an independent fork of an archived repository. No matching
customer incident or upstream acceptance is claimed.
