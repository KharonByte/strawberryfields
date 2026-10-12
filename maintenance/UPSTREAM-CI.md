# Full upstream CPU CI

The `Tests` workflow now runs the complete historical matrix on Ubuntu 24.04
with Python 3.12.3. It uses pinned `requirements-ci.txt` dependencies and installs
the separate TensorFlow CPU add-on in the six configurations that require it.

| Configuration | Selection | Batched |
| --- | --- | --- |
| Frontend, applications, API, Gaussian | `frontend or apps or api or gaussian` | No |
| Fock | `fock` | No |
| TensorFlow | `tf` | No |
| TensorFlow pure | `tf and pure` | Yes |
| TensorFlow mixed | `tf and mixed` | Yes |
| Bosonic | `bosonic` | No |
| Without TensorFlow installed | `not tf` | No |

All seven jobs run without fail-fast cancellation. The `core-tests` check fails
if any configuration fails or does not complete successfully. Failures are not
converted to successful checks. JUnit results, coverage and complete logs are
saved in the `upstream-<configuration>` artifacts. GPU execution is not included.

Run one configuration locally after installing `requirements-ci.txt` and,
unless testing the absence of TensorFlow, `maintenance/requirements-tensorflow.txt`:

```bash
BATCHED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests -q --tb=short -p no:warnings --randomly-seed=42 -m "bosonic"
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
