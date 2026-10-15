# Copyright 2026 KharonByte contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Check built package metadata against the documented maintenance environment."""

import argparse
from email.parser import Parser
from pathlib import Path
import re

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
import yaml

REPOSITORY = "https://github.com/KharonByte/strawberryfields"


def readme_setting(readme, name):
    """Read one explicit compatibility setting, rejecting missing or duplicate rows."""
    values = re.findall(rf"^\| {re.escape(name)} \| `([^`]+)` \|$", readme, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"README must contain exactly one '{name}' setting")
    return values[0]


def check_metadata(root, metadata):
    """Return inconsistencies; metadata must come from a built distribution."""
    errors = []
    readme = (root / "README.md").read_text(encoding="utf-8")
    try:
        documented = SpecifierSet(readme_setting(readme, "Python requirement"))
        ci_python = readme_setting(readme, "CI Python")
        if not re.fullmatch(r"\d+\.\d+\.\d+", ci_python):
            raise ValueError("README CI Python must pin an exact patch version")
        version = Version(ci_python)
        declared = SpecifierSet(metadata.get("Requires-Python", ""))
    except (ValueError, InvalidSpecifier) as exc:
        return [str(exc)]

    # The present support contract covers one minor line, with one exact CI pin.
    # Expanding it requires updating this policy and its validation matrix together.
    minor = f"{version.major}.{version.minor}"
    expected = SpecifierSet(f">={minor},<{version.major}.{version.minor + 1}")
    if not metadata.get("Requires-Python") or declared != expected:
        errors.append(f"Requires-Python must match the CI minor line: {expected}")
    if documented != declared:
        errors.append("README Python requirement differs from Requires-Python")
    classifiers = {
        value
        for value in metadata.get_all("Classifier", [])
        if re.fullmatch(r"Programming Language :: Python :: \d+\.\d+", value)
    }
    if classifiers != {f"Programming Language :: Python :: {minor}"}:
        errors.append("Python classifiers differ from the CI minor line")

    for field, expected_value in {
        "Name": "StrawberryFields",
        "Author": "Xanadu Inc.",
        "Maintainer": "KharonByte",
        "Home-page": REPOSITORY,
    }.items():
        if metadata.get(field) != expected_value:
            errors.append(f"{field} must be {expected_value!r}")
    expected_urls = {
        f"Source, {REPOSITORY}",
        f"Issues, {REPOSITORY}/issues",
        f"Support, {REPOSITORY}/blob/master/SUPPORT.md",
    }
    if not expected_urls.issubset(metadata.get_all("Project-URL", [])):
        errors.append("Project-URL must identify the fork's source, issues and support scope")
    if metadata.get_payload().replace("\r\n", "\n").strip() != readme.strip():
        errors.append("Built long description differs from README.md")

    required_commands = (
        f"git clone {REPOSITORY}.git",
        f"python -m pip install -r maintenance/requirements-python{version.major}{version.minor}.txt wheel==0.48.0",
        "python -m pip install --no-deps --no-build-isolation .",
        "python -m pip check",
    )
    if any(command not in readme.splitlines() for command in required_commands):
        errors.append("README must install the fork checkout with the pinned environment")
    if re.search(r"(?im)^\s*(?:python -m )?pip install strawberryfields\s*$", readme):
        errors.append("README must not install the upstream PyPI package")

    support = (root / "SUPPORT.md").read_text(encoding="utf-8")
    if f"| Environment | Python {ci_python} and " not in support:
        errors.append("SUPPORT.md environment differs from README CI Python")

    python_steps = 0
    workflows = root / ".github" / "workflows"
    for path in sorted([*workflows.glob("*.yml"), *workflows.glob("*.yaml")]):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
        for job in workflow.get("jobs", {}).values():
            # This inherited publishing job cannot run in the maintenance fork.
            if path.name == "upload.yml" and job.get("if") == (
                "github.repository == 'XanaduAI/strawberryfields'"
            ):
                continue
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/setup-python@"):
                    python_steps += 1
                    actual = str(step.get("with", {}).get("python-version", ""))
                    if actual != ci_python:
                        errors.append(f"{path.name}: CI Python {actual!r} differs from {ci_python}")
    if not python_steps:
        errors.append("No CI Python configuration found")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path, help="METADATA file from the built wheel")
    args = parser.parse_args()
    package_metadata = Parser().parsestr(args.metadata.read_text(encoding="utf-8"))
    problems = check_metadata(Path(__file__).resolve().parents[1], package_metadata)
    if problems:
        parser.exit(1, "\n".join(problems) + "\n")
    print("README, package metadata and CI Python versions agree.")
