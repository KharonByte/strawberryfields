# KharonByte Strawberry Fields demo

Locally verified on 12 October 2026 (Europe/Warsaw). Prepared for review before
pushing to `KharonByte/strawberryfields`; hosted CI has not run.

## The defect and the fix

A selected heterodyne measurement returned the value the caller requested, but
the bosonic simulator produced the remaining state for half that value. The
patch adds the missing amplitude-to-quadrature conversion: one executable
production line changes. The public API and reported measurement stay the same.

## Short before/after demo

The circuit prepares a two-mode squeezed vacuum and selects `0.25-0.15j` on
one mode. An exact analytic calculation predicts the state left in the other
mode. The same circuit, reference and tolerances are used before and after.

| Check | Original source | Patched source |
| --- | --- | --- |
| Remaining coherent amplitude | `0.058776607 + 0.065628456j` | `0.117553214 + 0.131256911j` |
| Analytic expected amplitude | `0.117553214 + 0.131256911j` | Same reference |
| Absolute amplitude error | `0.088100986` | `2.78e-17` |
| Fidelity to the expected full state | `0.992268261` | `1.000000000` (rounded) |
| Returned measurement | `0.25-0.15j` | `0.25-0.15j` |
| Gaussian control | Pass | Pass |
| Identical focused regression cases | **43 passed, 42 failed** | **85 passed, 0 failed** |

Both focused runs skip the same 32 Fock-backend combinations, which do not
support heterodyne measurement. No skips were added to make the patch pass.

The broader related suite on the patched source: **1,203 passed, 60 skipped,
104 deselected, no failures or errors**. This total includes the 85 focused
cases; it is not the full upstream suite.

## Run it

Follow the [pinned environment setup](README.md#reproduce), then run:

```bash
.venv/bin/python maintenance/heterodyne_demo.py
.venv/bin/python maintenance/check.py --related
```

The first command prints the patched numerical result. The second reproduces
the original failure in a temporary copy of the exact upstream revision and
reruns the identical tests against the patch. It saves both numerical demo
outputs, test logs, JUnit XML and an environment/source record under
`maintenance/artifacts/`.

[Recorded results and source/test hashes](validation.json) are from a fresh
isolated dependency installation on Ubuntu 24.04 / WSL2, Python 3.12.3,
Linux x86-64, Ryzen 9 5950X CPU. Dependency consistency and formatting of the
changed Python files also passed. No GPU or quantum hardware was used.

## Limits

The [support scope](../SUPPORT.md) covers this measurement correction and its
reproducer in the pinned environment. Full upstream CI, other platforms,
TensorFlow/GPU execution and external review remain unverified. The scoped
GitHub Actions workflow is prepared but has not run on GitHub.

This is an independent maintenance demonstration, not a customer incident or
proof of a paid requirement. Code that doubled `select` to compensate for the
old bug must remove that compensation when using this patch.
