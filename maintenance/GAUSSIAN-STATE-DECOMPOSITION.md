# Bounded correlated Gaussian preparation

This issue #26 regression supplements the Gaussian-only random pure/mixed tests
in `tests/integration/test_decompositions_integration.py`. Those tests, their
random inputs, tolerances and exclusions remain unchanged. The new deterministic
cases exercise the general `Gaussian(..., decomp=True)` path with small correlated
inputs on Gaussian, Fock and TensorFlow representations. They do not enable the
original three-mode random configurations or expand `SUPPORT.md`.

## Independent targets

Use quadratures `(x0, x1, p0, p1)` with `[xj, pk] = i*hbar*delta(j,k)`.
Set `r=0.12`, `phi=0.31`, `c=cosh(r)`, `s=sinh(r)` and
`vj=hbar*(nj+1/2)`. The physically pure target has `(n0,n1)=(0,0)`;
the mixed target has `(0.03,0.07)`. Define

```text
A = c*c*v0 + s*s*v1
B = s*s*v0 + c*c*v1
C = c*s*(v0+v1)
V = [[A, C*cos(phi), 0, C*sin(phi)],
     [C*cos(phi), B, C*sin(phi), 0],
     [0, C*sin(phi), A, -C*cos(phi)],
     [C*sin(phi), 0, -C*cos(phi), B]]
d = sqrt(2*hbar) * [0.08, -0.05, 0.03, 0.06]
```

These follow from a two-mode squeezing symplectic transformation of independent
thermal modes, rotation of mode zero, and displacement. The symplectic
eigenvalues are `hbar*(nj+1/2)`. The physical purities are one and `2500/3021`.
The transformation and thermal conventions follow Weedbrook et al.,
[arXiv:1110.3234v1](https://arxiv.org/pdf/1110.3234v1), equations 24–25, 27,
31–35 and 44, converted from its vacuum-variance-one/interleaved convention.

The implementation constructs inputs from an explicit symplectic product and
checks them against independent scalar formulas evaluated with Decimal at 80
and 120 digits before engine execution. It uses no simulator decomposition or
Gaussian-to-Fock conversion to generate the expected moments. The sine/cosine
Taylor remainder and rounding budget reserve `1e-12` dimensionless reference
uncertainty; the precision cross-check must agree within `1e-60`.

The pure covariance alone is invariant under whole-mode exchange. Its unequal
means detect a whole-state exchange; covariance-only mode-swap controls use
the mixed target. Signed cross correlations, units and mean/covariance shapes
have separate negative controls.

## Profiles and actual storage

At cutoff 20, ordinary execution covers three backends, two physical targets,
both requested initial storage flags, and `hbar` 1 and 1.7: 24 cases. Each
TensorFlow batch profile adds four cases, for a union of 32 identities. The
batch supplies one scalar covariance and displacement, replicated into two
members. Every returned member is checked; this is neither native batched
covariance-input support nor distinct-input batch sensitivity.

General Gaussian preparation emits per-mode vacuum or thermal preparations.
Subset replacement promotes Fock/TF storage to a density matrix even for a
physically pure target with a requested initially pure engine. The tests require
actual density storage in every finite-representation case. Requested initial
flags remain separate metadata; these cases do not claim ket-output coverage.
Gaussian representation uses native moments, without a Fock cutoff.

The compiled path must contain the elementary preparation, squeezing, passive
and both displacement directions, with no remaining composite Gaussian
operation. Tests do not require a particular nonunique factorization or an
exact total gate count.

## Accuracy and historical cutoff study

The componentwise dimensionless mean and covariance limit is `1e-4`, including
the reserved reference uncertainty, with zero relative tolerance. This is an
empirical accuracy requirement, stricter than the unchanged legacy `0.001`
tolerance. The smallest nonzero covariance signal exceeds `0.0369`.

Raw finite density diagnostics require trace error at most `1e-6`, imaginary
trace at most `1e-7`, entrywise Hermiticity error at most `1e-7`, and minimum
eigenvalue of the Hermitian part at least `-1e-6`. Independently reconstructed
complex first and symmetrized second moments must have dimensionless imaginary
residues at most `1e-6` before real parts are used. No density or moment is
renormalized, repaired or clamped. Quadrature products are formed before
cropping to preserve the boundary contribution.

The initial local study at implementation commit
`a19995dc6cb37a205f481922e5c0347f1b0b8959` used cutoffs 8, 12, 16 and 20 for
all 24 finite-representation configurations, plus eight Gaussian controls:
104 executions. Both cutoffs 16 and 20 had to meet the final criteria; their
mean and covariance changes had to be at most `2e-5` independently per member.
Lower-cutoff outputs were retained, without a monotonicity requirement.

On Linux CPU, Python 3.12.3 and the pinned NumPy 1.26.4/TensorFlow 2.21.0
environment, all 104 executions passed. Maximum dimensionless final errors
were `4.15e-8` for means and `1.41e-7` for covariance (rounded upward).
Maximum 16-to-20 changes were `1.12e-13` and `4.05e-12`; raw trace error was
at most `2.39e-7`. These are observed residuals, not universal error bounds.
Each profile had one numerical thread, a 14 GiB virtual-memory ceiling and a
180-second timeout with 30-second termination grace. The largest observed
process peak was 806,984 KiB; each profile finished within its limit.

The ideal final state's generating function gives a conservative trace-tail
bound below `4*3**(-D)` for these inputs. This does not bound covariance error
by the same number, or guarantee accuracy of truncated intermediate gates.
Normalization alone is insufficient; see Provazník et al.,
[arXiv:2202.07332v2](https://arxiv.org/pdf/2202.07332v2), equation 3 and
sections 3–4. Independent target agreement and the fixed convergence criteria
provide the bounded empirical evidence here.

The maintained regression checks cutoff 20 only. The multi-cutoff study is
historical evidence for the named source/environment, not a future CI
convergence guarantee. Material changes to decomposition, precision or backend
require renewed affected assessment. The exact-case guard rejects missing,
skipped, duplicated, renamed or incorrectly configured required rows.

First and second moments identify the intended Gaussian target, but matching
them does not establish fidelity or exact Gaussianity of arbitrary truncated
output. Arbitrary energies/phases, other decompositions, sampling, gradients,
hardware, security-baseline clearance and broader support are not established.
