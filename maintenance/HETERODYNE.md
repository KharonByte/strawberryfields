# Bosonic heterodyne postselection: before and after

`MeasureHeterodyne(select=alpha)` previously reported the requested value while
conditioning the bosonic simulation on half that amplitude. The fix converts
the coherent amplitude to the backend's internal quadratures before conditioning.
One executable production line changes.

The demo uses a two-mode squeezed vacuum with `r=0.7`, `phi=0.3` and
`alpha=0.25-0.15j`. An exact calculation predicts the remaining coherent amplitude:

```text
expected = exp(1j * phi) * tanh(r) * conjugate(alpha)
```

It checks that amplitude, fidelity of the whole conditional state (including
vacuum reset of the measured mode), and the unchanged reported measurement.
The Gaussian backend provides a separate control. No hardware measurement or
customer circuit is involved.

Code that previously doubled `select` to compensate for the defect must remove
that workaround. Supply the intended coherent amplitude directly.

## Reproduce

From this checkout, in Linux with Git and Python 3.12.3:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r maintenance/requirements-python312.txt
.venv/bin/python maintenance/heterodyne_demo.py
.venv/bin/python maintenance/check.py --related
```

The commands import this checkout directly. No Strawberry Fields installation
from PyPI is needed, and `pip install strawberryfields` does not include this fix.
On Ubuntu, creating the virtual environment may require the matching
`python3.12-venv` OS package.

`check.py` exports upstream commit
`162125d8ad5fd5488322a30c520253189596adf5` into a temporary directory and copies
the same two regression test files into it. It runs the demo and tests against
that original source, then against the patched checkout. It never reverts the
working checkout. The expected failures must be reproduced before success can
be reported. Omit `--related` for only the short demo and focused regression run.

Generated logs, JUnit reports and `validation.json` go to `maintenance/artifacts/`
(Git-ignored). [Recorded local results](validation.json) include source/test hashes
and environment versions. [Support scope](../SUPPORT.md) defines the limits.

## Reviewing the patch

The regression uses an analytic quantum-state reference, not the value returned
by another implementation of the changed conversion. The existing low-level test
previously treated a coherent amplitude as a quadrature; its expectation is corrected
and extended to zero and complex selections.

The public module/version names are retained for this source-based demonstration.
There is no KharonByte package release yet. Legacy upstream documentation below
the fork notice in the root README is historical, including installation and
hardware-service claims. This work does not establish customer adoption or
independent external review.
