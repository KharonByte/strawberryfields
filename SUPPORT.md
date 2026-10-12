# KharonByte maintenance scope

This is an independent, best-effort maintenance fork of Strawberry Fields.
Xanadu's authorship and Apache-2.0 license are retained. This fork is not an
official Xanadu release or evidence of endorsement by any laboratory.

## Initial scope

| Area | Initial maintenance boundary |
| --- | --- |
| Behavior | Selected bosonic and Gaussian measurement paths; individual corrections require their own regression tests and review |
| Regression checks | The bosonic and measurement test selection documented in `maintenance/README.md` |
| Environment | Python 3.12.3, Linux x86-64, the exact dependencies in `maintenance/requirements-python312.txt` |
| Validation platforms | Ubuntu 24.04 under WSL2 and GitHub-hosted Ubuntu 24.04 runners, CPU only |
| Controls | Analytic reference circuits for numerical fixes where available |

The initial scope covers keeping this test environment usable and reviewing
small, reproducible measurement corrections within it. Additional maintenance requires
a reproducible problem and an explicitly agreed environment and test boundary.
Passing the selected tests is not comprehensive validation of either backend.

## Outside this scope

- TensorFlow, automatic differentiation, GPU execution, and general Fock-backend support.
- Other operating systems, architectures, Python versions, or unpinned dependency combinations.
- Quantum hardware access, retired cloud services, and instrument/control software.
- New algorithms, general simulator correctness certification, or validation of a user's research conclusions.
- A whole-project security audit, guaranteed response times, or a paid support SLA.

No quantum board or GPU is needed for these simulator tests. Passing CI is not
evidence that every known numerical defect has been corrected. Each fix must
document its behavior change, validation and compatibility limits separately.

For an in-scope report, provide a minimal circuit, expected result or independent
reference, actual result, and the environment versions. Use this fork's issue
tracker if enabled. Public reports must omit credentials and private research data.
