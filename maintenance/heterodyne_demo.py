"""Check heterodyne postselection against an exact two-mode analytic result.

Run against this checkout by default, or select an unmodified checkout with
--repo. Exit 0 means both backends agree with the analytic state and preserve
the requested measurement value; exit 1 means a numerical check failed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / "strawberryfields" / "__init__.py").is_file():
        parser.error("--repo must contain a Strawberry Fields source checkout")

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.path.insert(0, str(repo))
    import numpy as np
    import strawberryfields as sf
    from strawberryfields import ops

    if not Path(sf.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Loaded Strawberry Fields from outside the requested checkout")

    sf.hbar = 2
    r, phi = 0.7, 0.3
    selected = 0.25 - 0.15j
    # <selected| S2(r, phi)|00> leaves this coherent amplitude in mode 0.
    expected = np.exp(1j * phi) * np.tanh(r) * selected.conjugate()
    rows = {}
    for backend_name in ("gaussian", "bosonic"):
        program = sf.Program(2)
        with program.context as q:
            ops.S2gate(r, phi) | q
            ops.MeasureHeterodyne(select=selected) | q[1]
        engine = sf.Engine(backend_name)
        result = engine.run(program)
        state = engine.backend.state(modes=[0])
        means = state.means()
        if backend_name == "bosonic":
            means = state.weights() @ means
        actual = complex((means[0] + 1j * means[1]) / 2)
        fidelity = complex(result.state.fidelity_coherent([expected, 0j]))
        reported = complex(np.asarray(q[1].val).reshape(-1)[0])
        preserved = bool(
            abs(reported - selected) < 1e-12
            and np.allclose(result.samples, [[selected]], atol=1e-12, rtol=0)
        )
        rows[backend_name] = {
            "expected_amplitude": [float(expected.real), float(expected.imag)],
            "actual_amplitude": [actual.real, actual.imag],
            "absolute_error": float(abs(actual - expected)),
            "fidelity_to_expected_state": [fidelity.real, fidelity.imag],
            "measurement_value_preserved": preserved,
            "passed": bool(
                abs(actual - expected) < 1e-12 and abs(fidelity - 1) < 1e-10 and preserved
            ),
        }

    source = repo / "strawberryfields/backends/bosonicbackend/bosoniccircuit.py"
    report = {
        "selected_alpha": [selected.real, selected.imag],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "backends": rows,
        "passed": all(row["passed"] for row in rows.values()),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Expected remaining-mode amplitude: {expected:.9f}")
        print("Backend    Actual amplitude           Absolute error  State fidelity  Result")
        for name, row in rows.items():
            actual = complex(*row["actual_amplitude"])
            fidelity = row["fidelity_to_expected_state"][0]
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"{name:10} {actual: .9f}  {row['absolute_error']:14.3e}"
                f"  {fidelity:.9f}     {status}"
            )
        print("The reported measurement must remain 0.25-0.15j in both backends.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
