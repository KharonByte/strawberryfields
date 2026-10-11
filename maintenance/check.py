"""Reproduce the original failure and verify the patch using identical tests.

The baseline is exported into a temporary directory; the working checkout is
never reverted or patched. Requires Python 3.12, Git and the pinned test environment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "162125d8ad5fd5488322a30c520253189596adf5"
SOURCE = "strawberryfields/backends/bosonicbackend/bosoniccircuit.py"
TESTS = [
    "tests/integration/test_measurement_integration.py::TestPostselection::test_heterodyne_conditional_state",
    "tests/bosonic_files/test_backend_bosoniccircuit.py::TestBosonicModes::test_post_select_heterodyne",
]
RELATED = [
    "tests/bosonic_files",
    "tests/backend/test_heterodyne.py",
    "tests/backend/test_postselection.py",
    "tests/backend/test_threshold_measurement.py",
    "tests/integration/test_measurement_integration.py",
]


def junit(path):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    totals = {
        key: sum(int(s.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    totals["passed"] = totals["tests"] - sum(totals[k] for k in ("failures", "errors", "skipped"))
    cases = [
        (case.get("classname"), case.get("name"))
        for suite in suites
        for case in suite.findall("testcase")
    ]
    skipped = [
        (case.get("classname"), case.get("name"))
        for suite in suites
        for case in suite.findall("testcase")
        if case.find("skipped") is not None
    ]
    return totals, sorted(cases), sorted(skipped)


def run(command, cwd, output, name, env):
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    (output / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/artifacts")
    parser.add_argument("--related", action="store_true", help="Also run the broader related suite")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONNOUSERSITE="1",
    )
    env.pop("PYTHONPATH", None)
    report = {
        "baseline_commit": BASELINE,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "thewalrus", "numba", "pytest", "pytest-mock")
            },
        },
        "sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                SOURCE,
                *(test.split("::")[0] for test in TESTS),
                "maintenance/heterodyne_demo.py",
                "maintenance/check.py",
                "maintenance/requirements-python312.txt",
            ]
        },
        "demo": {},
        "regressions": {},
    }
    statuses, collections, skipped = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="sf-heterodyne-") as temporary:
        baseline = Path(temporary) / "baseline"
        baseline.mkdir()
        archive = Path(temporary) / "baseline.tar"
        with archive.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", BASELINE, "strawberryfields", "tests"],
                cwd=ROOT,
                stdout=handle,
                check=True,
            )
        with tarfile.open(archive) as handle:
            handle.extractall(baseline, filter="data")
        for test in TESTS:
            relative = test.split("::")[0]
            shutil.copyfile(ROOT / relative, baseline / relative)

        for label, repo in (("before", baseline), ("after", ROOT)):
            demo = run(
                [
                    sys.executable,
                    str(ROOT / "maintenance/heterodyne_demo.py"),
                    "--repo",
                    str(repo),
                    "--json",
                ],
                repo,
                output,
                f"demo-{label}",
                env,
            )
            if demo.returncode not in (0, 1):
                raise RuntimeError(f"{label} demo crashed; inspect {output}")
            report["demo"][label] = json.loads(demo.stdout)
            xml = output / f"regressions-{label}.xml"
            result = run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=short",
                    "--disable-warnings",
                    *TESTS,
                    f"--junitxml={xml}",
                ],
                repo,
                output,
                f"regressions-{label}",
                env,
            )
            totals, collections[label], skipped[label] = junit(xml)
            report["regressions"][label] = totals
            statuses[label] = {"demo": demo.returncode, "tests": result.returncode}
            print(
                f"{label}: {totals['passed']} passed, {totals['failures']} failed, "
                f"{totals['skipped']} skipped, {totals['errors']} errors",
                flush=True,
            )

    before, after = report["regressions"]["before"], report["regressions"]["after"]
    old_bosonic = report["demo"]["before"]["backends"]["bosonic"]
    half_error = math.dist(
        old_bosonic["actual_amplitude"], [value / 2 for value in old_bosonic["expected_amplitude"]]
    )
    valid = (
        statuses["before"] == {"demo": 1, "tests": 1}
        and statuses["after"] == {"demo": 0, "tests": 0}
        and before["failures"] > 0
        and before["errors"] == 0
        and after["failures"] == after["errors"] == 0
        and collections["before"] == collections["after"]
        and skipped["before"] == skipped["after"]
        and report["demo"]["before"]["backends"]["gaussian"]["passed"]
        and not old_bosonic["passed"]
        and half_error < 1e-12
        and old_bosonic["measurement_value_preserved"]
        and report["demo"]["after"]["passed"]
        and report["demo"]["after"]["source_sha256"] == report["sha256"][SOURCE]
    )
    if args.related:
        print("Running the broader related measurement suite...", flush=True)
        xml = output / "related-after.xml"
        result = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--disable-warnings",
                "-m",
                "bosonic or gaussian",
                *RELATED,
                f"--junitxml={xml}",
            ],
            ROOT,
            output,
            "related-after",
            env,
        )
        totals, _, _ = junit(xml)
        report["related_after"] = totals
        valid = valid and result.returncode == 0 and totals["failures"] == totals["errors"] == 0
        print(
            f"related: {totals['passed']} passed, {totals['failures']} failed, "
            f"{totals['skipped']} skipped",
            flush=True,
        )

    report["passed"] = bool(valid)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if valid else 'FAIL'}: results in {output}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
