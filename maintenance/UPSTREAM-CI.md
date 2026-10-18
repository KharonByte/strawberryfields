# Full upstream CPU CI

The `Tests` workflow runs five CPU configurations on each of Ubuntu 24.04
(x86-64), Windows Server 2025 (x86-64) and macOS 15 (ARM64), using Python 3.12.3.
Each environment installs `maintenance/requirements-core.txt` and the selected
pinned cloud/TensorFlow add-ons. macOS uses `tensorflow`; Linux and Windows use
`tensorflow-cpu`, at the same version. CI checks their presence or absence
before collecting tests.

| Configuration | Selection | Batched |
| --- | --- | --- |
| All backends, frontend, applications, API and integration | Entire `tests` directory, no marker filter | No |
| TensorFlow pure | `tf and pure` | Yes |
| TensorFlow mixed | `tf and mixed` | Yes |
| Without TensorFlow installed | `not tf` | No |
| Without TensorFlow or the cloud client installed | `not tf` | No |

All 15 jobs run without fail-fast cancellation. The `core-tests` check requires
every configuration to succeed. JUnit results, branch coverage and complete
logs are saved in `upstream-<os>-<configuration>` artifacts. GPU execution is
not included. Remote API tests use mocks; no cloud credentials or QPU are needed.

`maintenance/check_required_tests.py` also rejects missing, skipped or failed
supported backend/pure-state combinations for the restored beam-splitter,
two-mode squeezing, heterodyne reset, measurement exception, Mach-Zehnder,
mean-photon, quadrature and Gaussian initial-state checks. The cutoff fixture
now honours explicit indirect parameters, covered by its own regression test.

The guard also requires all 148 restored unbatched cases: TensorFlow polynomial
quadratures, direct two-mode squeezing and dark counts; Gaussian squeezed-state
parity; and Gaussian/Bosonic state observables. It rejects duplicate test
identities and checks parameter counts and pass status. TensorFlow polynomial
quadratures retain their unsupported batching exclusions.

A further 516 previously skipped case/configuration combinations are covered:
428 Gaussian/Bosonic density-matrix comparisons, 16 Fock/TensorFlow displacement
unit checks, 16 shared state/engine assertions, and 56 TensorFlow batching cases.
The density comparisons include complex off-diagonal entries and preserve the
existing Fock/TensorFlow amplitude checks. Batch assertions verify shapes and
every member, with additional distinct-input controls. These are parametrized
combinations, not 516 distinct features; 56 belong to separate batch runs.
The X/Z displacement checks exercise distinct batch members and verify their
means, variances and shapes; polynomial-quadrature batching remains excluded.

The `hbar` fixture honours explicit parameters before environment/default values
and shares the test's monkeypatch undo stack, restoring previous units even
when a test changes them again or fails. State-reduction tests use this same
fixture for all three unit settings (1.0, 1.7 and 2.0) and check the returned
state's units. Homodyne variance is checked
per batch member using a two-sided chi-square interval with false-rejection
probability 1e-6 for Gaussian samples, replacing the absolute tolerance that
accepted a constant sampler. Regression tests reject zero/incorrectly scaled
samples, lost coherences, missing configurations and mislabeled unit values.
The CI guard requires the expanded counts and setup configuration: `setup_hbar`,
cutoff and batch size, with per-unit counts and agreement between requested and
setup units. These properties are captured before the test body; they do not
describe later unit changes. State-reduction and displacement tests also check
the returned state's units directly. Reporting regressions exercise multiple
state-unit values within one test, including restored units and failure paths.
Non-vacuum TensorFlow fidelity remains a separate numerical defect.

Another 704 previously skipped configurations are enabled: 689 unbatched and
15 TensorFlow batched. Gaussian/Bosonic normalization uses finite-cutoff density
tensors with explicit ket/bra ordering and the existing tolerance. Bosonic
squeezed-state parity retains cutoff 10 and permits only cancellation roundoff
(`100 * float64.eps`); other backends retain exact-zero assertions. TensorFlow
probability, density-matrix and unitary-reference comparisons check complete
batch shapes and distinct inputs. Channel extraction still uses Fock internally;
the added coverage compares its output with independently executed TF circuits.
The coverage guard requires these configurations, including legacy storage-first
parameter IDs, and rejects conflicting backend/state labels. See
[the remaining-skip audit](SKIP-AUDIT.md) for unresolved coverage and software issues.

An additional 42 configurations are enabled: 22 unbatched and ten in each TF
batch profile. These cover Gaussian mode addition, Gaussian/Bosonic utility
states in the Fock representation, cutoff-aware vacuum density matrices and
batched channel-reference comparisons. The latter retain all three channel
representations, both two-mode tensor layouts and distinct positive batch
inputs. Regression tests reject missing batch axes, duplicated batch outputs
and missing coverage requirements. The earlier unbatched channel inputs and
numerical tolerances remain covered.

The 15 TensorFlow batch gradient configurations (eight pure, seven mixed) now
compare full Jacobians to analytic derivatives. Distinct parameters exercise
every batch member and zero cross-member derivatives. Mixed two-mode tests use
density-matrix preparation and verify their storage. The guard also requires
the 15 corresponding unbatched cases; only the mixed-storage ket accessor stays
excluded. Regression tests reject disconnected/summed gradients, wrong tensor
axes, duplicated members and cross-member coupling, and require all 15 cases.

Backend capability exclusions remain: Gaussian backends cannot directly prepare
Fock states, direct two-mode squeezing is implemented only by Fock/TensorFlow,
and TensorFlow Wigner evaluation does not support batching. Existing numerical
expected failures and other unsupported combinations are not reclassified or
hidden by this test-only change. Every remaining exclusion is recorded with
its reason in the JUnit report.

Run the full unbatched suite locally after installing the core, cloud and
TensorFlow requirement files:

```bash
BATCHED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q --tb=short -p no:warnings --randomly-seed=42 \
  --junitxml=tests.xml
python maintenance/check_required_tests.py tests.xml all-backends
```

The first complete [hosted run](https://github.com/KharonByte/strawberryfields/actions/runs/38256686910)
passed all seven configurations and the `core-tests` gate at commit `d3cd5fc`
on 12 October 2026. Latest-commit results are recorded in setup PR #2.

| Configuration | Passed | Skipped | Expected failures |
| --- | ---: | ---: | ---: |
| Frontend, applications, API, Gaussian | 2,967 | 1,746 | 8 |
| Fock | 1,959 | 507 | 0 |
| TensorFlow | 1,520 | 946 | 0 |
| TensorFlow pure, batched | 705 | 528 | 0 |
| TensorFlow mixed, batched | 700 | 533 | 0 |
| Bosonic | 2,378 | 1,667 | 0 |
| Without TensorFlow installed | 7,143 | 3,684 | 8 |

Selections overlap; these counts must not be summed as unique test cases.
The original test-level skips and expected failures are retained. No workflow
job was skipped, and there were no unexpected test failures.

## Dependency compatibility

The first broader run exposed a `GraphEmbed` failure with SciPy 1.13.1:
`sqrtm` propagated a `complex256` result to NumPy's matrix inversion, which
does not support that dtype. The same three `TestGraphEmbed` cases pass with
SciPy 1.17.1 and NumPy 1.26.4, without a simulator or test change. The shared
environment now pins that compatible SciPy version. SciPy 1.18.1 requires
NumPy 2 and is not selected for this NumPy 1.26 environment.

## Earlier import blocker

On 12 October 2026, a fresh Python 3.12.3 environment resolved the historical CI
requirements to setuptools 84.0.0. The project installed, but importing the
actual checkout failed along this path:

```text
strawberryfields/__init__.py -> apps -> apps.data -> apps.data.sample
strawberryfields/apps/data/sample.py:20: import pkg_resources
ModuleNotFoundError: No module named 'pkg_resources'
```

The same failure stopped test collection in Dependabot PRs #4, #5 and #6.
It occurred before tests ran, so those failures do not establish a regression
caused by any of the proposed dependency bumps. The restored environment pins
setuptools 80.9.0, which still provides the required API. The long-term fix is
to migrate dataset resource access and validate it in a separate code change.

Reproduce with the historical requirements in a fresh environment, from the
repository root (the current requirements now pin the compatible environment):

```bash
git show 24edbc9:requirements-ci.txt > /tmp/strawberryfields-legacy-requirements.txt
python -m pip install setuptools==84.0.0 wheel
python -m pip install --no-build-isolation -r /tmp/strawberryfields-legacy-requirements.txt .
python -c "import strawberryfields"
```

The SciPy integration rename is already handled by this source and is not
reported as a blocker.

## What the broader probe established

The narrow test environment initially could not collect the full suite because
Plotly was absent. With Plotly added, a bounded non-TensorFlow run passed 511
cases and skipped 5 before being deliberately interrupted after 157 seconds.
That was partial diagnostic evidence. The subsequent complete hosted run above
supersedes it for TensorFlow, Fock, application and API coverage on Linux CPU.

## Legacy documentation and dependency lists

The old Sphinx action ran Python 3.8 and attempted to install NumPy 1.26.4, for
which that interpreter has no compatible distribution. The restored docs job
uses Python 3.12.3, a separate pinned environment and the full Sphinx build with
warnings fatal. The docs job includes TensorFlow CPU for documentation imports;
the separate full test matrix verifies TensorFlow numerical execution.

The old root/docs requirement lists are replaced by includes of the supported
maintenance files. These already use requests 2.34.2, urllib3 2.8.0 and, for
documentation, tensorflow-cpu 2.21.0. This supersedes the isolated legacy bumps
in PRs #4-#6; their old requirements are not silently left outside CI.
