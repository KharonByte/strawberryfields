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
The pinned environment excludes TensorFlow and GPU support.

## Continuous integration

The `Scoped maintenance checks` workflow runs the same test selection on
Ubuntu 24.04 with Python 3.12.3. Its actions use immutable commit pins and the
Node.js 24 runtime. The `heterodyne` job name is retained for the required
branch-protection check. The `scoped-validation` artifact contains the JUnit report.

The inherited broad-test, formatting, documentation and publishing workflows
are restricted to the upstream repository. This fork does not publish packages
or claim that its limited test selection replaces the full upstream suite.
Generated files under `maintenance/artifacts/` are ignored by Git.

See [the support scope](../SUPPORT.md) for the maintenance boundary.
