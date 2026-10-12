"""Build the complete documentation from a clean copy of tracked source files."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "maintenance/artifacts/docs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    with tempfile.TemporaryDirectory(prefix="sf-docs-") as temporary:
        source = Path(temporary)
        for relative in filter(None, tracked):
            original = ROOT / relative
            if original.is_file():
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
        env = dict(os.environ, PYTHONPATH=str(source))
        with (output / "build.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-b",
                    "html",
                    "-W",
                    "--keep-going",
                    "-T",
                    "-E",
                    "-d",
                    str(output / "doctrees"),
                    "-w",
                    str(output / "warnings.log"),
                    ".",
                    str(output / "html"),
                ],
                cwd=source / "doc",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
    print("\n".join((output / "build.log").read_text(encoding="utf-8").splitlines()[-40:]))
    print(f"Documentation build exit code: {result.returncode}; evidence: {output}")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
