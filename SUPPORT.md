# KharonByte maintenance scope

This is an independent, best-effort maintenance fork of Strawberry Fields.
Xanadu's authorship and Apache-2.0 license are retained. This fork is not an
official Xanadu release or evidence of endorsement by any laboratory.

## Initial scope

| Area | Commitment for this first portfolio patch |
| --- | --- |
| Behavior | Correct coherent-amplitude scaling in bosonic `MeasureHeterodyne(select=alpha)` |
| Regression checks | Exact conditional states, reported samples, measured-mode reset, zero/real/imaginary/complex outcomes, either mode, frontend hbar 1 and 2 |
| Environment | Python 3.12.3, Linux x86-64, the exact dependencies in `maintenance/requirements-python312.txt` |
| Validation platform | Ubuntu 24.04 under WSL2 on a Ryzen 9 5950X, CPU only |
| Controls | Gaussian backend for the same analytic circuit, plus the related bosonic and measurement test selection |

The initial scope covers this correction and keeping its reproducer and
regression tests usable in that environment. Additional maintenance requires
a reproducible problem and an explicitly agreed environment and test boundary.
Passing the selected tests is not comprehensive validation of either backend.

## Outside this scope

- TensorFlow, automatic differentiation, GPU execution, and general Fock-backend support.
- Other operating systems, architectures, Python versions, or unpinned dependency combinations.
- Quantum hardware access, retired cloud services, and instrument/control software.
- New algorithms, general simulator correctness certification, or validation of a user's research conclusions.
- A whole-project security audit, guaranteed response times, or a paid support SLA.

No quantum board or GPU is needed to reproduce this numerical defect. Hosted CI
is configured separately; the checked-in result records local validation only.

## Compatibility note

Code that previously compensated for the bug by doubling `select` must remove
that compensation. Supply the intended coherent amplitude directly. The fix
preserves the requested value in both the measurement register and returned samples.

For an in-scope report, provide a minimal circuit, expected result or independent
reference, actual result, and the environment versions. Use this fork's issue
tracker if enabled. Public reports must omit credentials and private research data.
