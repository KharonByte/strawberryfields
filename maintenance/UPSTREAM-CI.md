# Full upstream CI: current limits

The historical matrix is still restricted to upstream. It uses a floating
`requirements-ci.txt` and includes frontend, applications, API, Gaussian, Fock,
bosonic and three TensorFlow configurations. The scoped matrix does not replace
that coverage.

## Reproduced import blocker

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
caused by any of the proposed dependency bumps. The scoped environment pins
setuptools 80.9.0, which still provides the required API. The long-term fix is
to migrate dataset resource access and validate it in a separate code change.

Reproduce in a fresh environment, from the repository root:

```bash
python -m pip install setuptools==84.0.0 wheel
python -m pip install --no-build-isolation -r requirements-ci.txt .
python -c "import strawberryfields"
```

The SciPy integration rename is already handled by this source and is not
reported as a blocker.

## What the broader probe established

The narrow test environment initially could not collect the full suite because
Plotly was absent. With Plotly added, a bounded non-TensorFlow run passed 511
cases and skipped 5 before being deliberately interrupted after 157 seconds.
That is partial diagnostic evidence, not a full-suite pass. The complete
TensorFlow, Fock, application and API matrix remains unverified.

## Legacy documentation and dependency lists

The old Sphinx action ran Python 3.8 and attempted to install NumPy 1.26.4, for
which that interpreter has no compatible distribution. The restored docs job
uses Python 3.12.3, a separate pinned environment and the full Sphinx build with
warnings fatal. It includes TensorFlow CPU for documentation imports only.

The old root/docs requirement lists are replaced by includes of the supported
maintenance files. These already use requests 2.34.2, urllib3 2.8.0 and, for
documentation, tensorflow-cpu 2.21.0. This supersedes the isolated legacy bumps
in PRs #4-#6; their old requirements are not silently left outside CI.
