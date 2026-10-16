# Strawberry Fields - KharonByte maintenance fork

Strawberry Fields is a Python library for simulating photonic quantum circuits.
This is independent maintenance of the archived
[Xanadu project](https://github.com/XanaduAI/strawberryfields), with the original
authorship and Apache-2.0 license retained.

## Current support

The package's Python compatibility range and exact CI interpreter are listed
below. CI uses the pinned dependencies shown here.

| Setting | Current value |
| --- | --- |
| Python requirement | `>=3.12,<3.13` |
| CI Python | `3.12.3` |
| CPU CI runners | Ubuntu 24.04 x86-64, Windows 2025 x86-64, macOS 15 arm64 |
| Dependencies | [Pinned maintenance environment](maintenance/requirements-python312.txt) |

Maintenance covers selected Gaussian and bosonic measurement paths and reviewed
numerical corrections. All three CI platforms also run the full upstream test
suite, including Fock and TensorFlow CPU configurations; that regression coverage does
not expand the [support commitment](SUPPORT.md). GPU execution and quantum
hardware or cloud-service access are outside the scope.

See [the test environment and CI checks](maintenance/README.md) for commands and
limits. Passing CI does not certify every simulator result.

## Install this fork

There is no KharonByte package release on PyPI yet. Install this source checkout
in a fresh virtual environment, using the CI Python version above. The package
and import names remain `StrawberryFields` and `strawberryfields`.

```console
git clone https://github.com/KharonByte/strawberryfields.git
cd strawberryfields
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install the pinned environment
and this checkout:

```console
python -m pip install -r maintenance/requirements-python312.txt wheel==0.48.0
python -m pip install --no-deps --no-build-isolation .
python -m pip check
```

### Optional cloud client

Local simulation does not require Xanadu's cloud client. Blackbird 0.5.0 and
XIR 0.2.2 remain pinned upstream dependencies. For the legacy `RemoteEngine`
API, install this checkout with its optional extra:

```console
python -m pip install -c maintenance/requirements-python312.txt -c maintenance/requirements-cloud.txt ".[cloud]"
```

This preserves the client interface; it does not establish access to a working
cloud service or quantum hardware.

## Try a local circuit

This one-mode Gaussian simulation needs only a CPU. A coherent displacement of
0.5 has a mean photon number of 0.25:

```python
import strawberryfields as sf
from strawberryfields.ops import Dgate

program = sf.Program(1)
with program.context as q:
    Dgate(0.5) | q[0]

state = sf.Engine("gaussian").run(program).state
print(state.mean_photon(0)[0])  # 0.25, up to floating-point rounding
```

The [archived upstream guides](https://strawberryfields.readthedocs.io/en/stable/introduction/introduction.html)
remain useful API references. Their installation and hardware-service guidance
is historical; use this fork's installation instructions and support scope.

## Report a problem

Use the [KharonByte issue tracker](https://github.com/KharonByte/strawberryfields/issues).
Include a minimal circuit, an expected result or independent reference, the
actual result, and environment versions. Check [SUPPORT.md](SUPPORT.md) before
relying on this fork for a new environment or workload.

## Authors and citation

Strawberry Fields was developed by Xanadu and
[its contributors](https://github.com/XanaduAI/strawberryfields/graphs/contributors).
KharonByte maintains this independent fork. Research using Strawberry Fields
should cite the original papers:

> Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and
> Christian Weedbrook. "Strawberry Fields: A Software Platform for Photonic Quantum Computing",
> [Quantum, 3, 129](https://quantum-journal.org/papers/q-2019-03-11-129/) (2019).

> Thomas R. Bromley, Juan Miguel Arrazola, Soran Jahangiri, Josh Izaac, Nicolás Quesada,
> Alain Delgado Gran, Maria Schuld, Jeremy Swinarton, Zeid Zabaneh, and Nathan Killoran.
> "Applications of Near-Term Photonic Quantum Computers: Software and Algorithms",
> [Quantum Sci. Technol. 5 034010](https://iopscience.iop.org/article/10.1088/2058-9565/ab8504/meta) (2020).

## License

[Apache License, Version 2.0](LICENSE).
