# Single-mode Gaussian terminal detectors

This correction starts from base `3f8b60e369fac6fdc271a8b8fb38ec39353cc4ce`.
It is separate from PR #49. It changes measurements selecting **one mode**,
including a marginal of an entangled Gaussian state. It does not certify
multimode sampling or update the conditional non-Gaussian state.

The pinned Walrus 0.21.0 PNR sampler defaults to cutoff 5, normalizes truncated
conditional probabilities, and rejects the last output mode at the cutoff.
For one squeezed vacuum mode with tanh(r)=1/2, it returns only 0, 2 and 4.
The missing probability above 4 is `1 - 147 sqrt(3)/256`, approximately
0.00542395034. Its threshold sampler uses finite fanout 10 and local cutoff 1;
that approximation is distinct from the PNR truncation.

## Correction and accuracy boundary

The backend uses internal hbar=2, vacuum covariance I and mean vector
`mu=(2 Re(alpha), 2 Im(alpha))`. For a one-mode covariance V, define

```
A(z) = V + I - z(V - I)
G(z) = 2 exp[(z-1) mu^T A(z)^(-1) mu / 2] / sqrt(det(A(z)))
```

G is the photon-number probability generating function. For z>1 with A positive
definite, Markov's inequality gives `P(N >= d) <= G(z)/z**d`. Four fixed choices
of z select a sufficient cutoff for a `1e-10` omitted-mass target. This target
is independent of the validation suite's tolerances. The maximum cutoff 512
is a resource limit: exceeding it raises `ValueError` before allocating the
density matrix. There is no rejection loop or fallback to the old sampler.

Walrus's unnormalized density-matrix Hermite recurrence provides the finite
diagonal, independently of its sampling implementation. The normalized finite
law has total variation distance equal to its omitted mass in exact arithmetic.
Finiteness, imaginary parts, negative entries and mass are checked before
normalization. Small negative roundoff is clipped only within `1e-9`.
These checks and analytic tests are **not a rigorous floating-point error
certificate**. Neither a tail bound nor total variation bounds all unbounded
observables. The allocation is at most a 512 by 512 complex density matrix,
plus the requested output and recurrence workspace.

For threshold measurement, the vacuum overlap is computed directly:

```
P(no click) = 2 exp[-mu^T (V+I)^(-1) mu / 2] / sqrt(det(V+I))
```

A Bernoulli draw removes finite fanout error for this one-mode path. The formula
is exact in real arithmetic; floating-point and PRNG limitations still apply.

The Gaussian overlap follows from Marian and Marian,
[arXiv:1111.7067v2, equations 11, 14 and 15](https://arxiv.org/html/1111.7067v2).
Their hbar=1 covariance and mean are V/2 and mu/sqrt(2). Substituting the thermal
state identity `z^N = rho_thermal(z/(1-z))/(1-z)` first for 0<=z<1 derives G;
analytic continuation gives the stated positive moment-generating domain.
The [Walrus 0.21.0 source](https://github.com/XanaduAI/thewalrus/blob/v0.21.0/thewalrus/samples.py)
documents the inherited sampling defaults.

## Compatibility and validation

Return shape remains `(shots, 1)`, with integer counts/clicks. The existing
NumPy global random seed controls draws, but seeded outputs change. Very energetic
states that previously returned a censored law can now raise an explicit resource
error. Simultaneous measurements of multiple modes retain the inherited Walrus
path and its approximation limitations. Postselection and conditional-state
limitations are unchanged. No dependency or CI workflow is changed.

The unit regressions compare independent Poisson, geometric and binomial-loss
squeezed laws, including complex phase, tail mass, terminal direct/Engine sampling,
selected entangled marginal and explicit numerical/resource failures.
These tests do not replace the frozen 68-case detector study. The original base
result remains 64 PASS, 2 FAIL and 2 statistically inconclusive; a corrected
baseline requires separate complete evidence with the same frozen criteria.
