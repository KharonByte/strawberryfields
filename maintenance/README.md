# Scoped maintenance environment

Repository setup provides a pinned CPU test environment for selected bosonic
and Gaussian measurement code. It does not change simulator behavior or test
expectations. Numerical fixes and their demonstrations are reviewed separately.

## Run the checks

From this checkout on Linux x86-64 with Python 3.12.3:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r maintenance/requirements-python312.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  .venv/bin/python -m pytest -q --tb=short --disable-warnings \
  -m "bosonic or gaussian" \
  tests/bosonic_files \
  tests/backend/test_heterodyne.py \
  tests/backend/test_postselection.py \
  tests/backend/test_threshold_measurement.py \
  tests/integration/test_measurement_integration.py \
  --junitxml=maintenance/artifacts/related.xml
```

Tests import this source checkout directly; do not install Strawberry Fields
from PyPI for this check. Ubuntu may require the `python3.12-venv` OS package.
The measurement test environment excludes TensorFlow and GPU support.
The root `requirements.txt` includes this same pinned environment.

## Continuous integration

The `Scoped maintenance checks` workflow runs the same test selection on
Ubuntu 24.04 (x86-64), Windows 2025 (x86-64) and macOS 15 (arm64), with Python
3.12.3. The Linux job retains the `heterodyne` name for branch protection.
Each platform uploads its JUnit report as `scoped-validation-<OS>`.

Black 24.10.0 checks every Python file under `strawberryfields/`, `tests/` and
`maintenance/`, using the existing 100-character line length.

Sphinx builds the complete documentation with warnings treated as errors:

```bash
python3.12 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r doc/requirements.txt
.venv-docs/bin/python maintenance/build_docs.py
```

Graphviz must also be installed. This Linux-only docs environment includes CPU
TensorFlow for API imports; it does not validate TensorFlow numerical results.
The builder copies tracked working files into a temporary directory so generated
API pages cannot contaminate later builds. HTML and diagnostics go under
`maintenance/artifacts/docs/` and are uploaded as the `documentation` artifact.

CodeQL scans Python and Actions workflows. Dependency review rejects newly
introduced high/critical vulnerabilities. Dependabot proposes weekly changes to
action pins and the dependency files in `maintenance/`; nothing auto-merges.
Its configuration and the weekly CodeQL schedule activate from `master` after
the setup PR is merged. Repository secret scanning, push protection and
Dependabot security updates are configured through repository settings.

All active actions use immutable Node.js 24 release pins. Jobs run on pull
requests and pushes to `master`, with duplicate in-progress runs cancelled.
The `Tests` workflow also runs all seven historical upstream selections on
Linux CPU, including three TensorFlow configurations and a separate environment
without TensorFlow. The required `core-tests` result succeeds only when every
selection succeeds. Each uploads its full test log, JUnit results and coverage.
See [the matrix and its results](UPSTREAM-CI.md) for details. Package publishing
remains upstream-only. Generated files under `maintenance/artifacts/` are ignored
by Git.

See [the support scope](../SUPPORT.md) for the maintenance boundary.
