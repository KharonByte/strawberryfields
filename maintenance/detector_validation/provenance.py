"""Read-only source/runtime identities for frozen detector validation."""

import hashlib
import ctypes
import os
import signal
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import resource
import sys

ROOT = Path(__file__).resolve().parents[2]
THREADS = {
    name: "1"
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}
CANDIDATE = "0d586c80b9dbed9761f4f044e7b8c33856aec325"


def sanitized_environment(executable=None, source_root=None):
    """Exact allowlist shared by production launch and no-sampling bootstrap."""
    executable = sys.executable if executable is None else executable
    source_root = ROOT if source_root is None else source_root
    return {
        "PATH": str(Path(executable).parent) + ":/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": "/dev/null",
        "TMPDIR": "/dev/null",
        "TEST_TMPDIR": "/dev/null/detector-tests-disabled",
        "KERAS_HOME": "/dev/null/detector-keras-disabled",
        "PYTHONPATH": str(source_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/dev/null/unused-pycache",
        "CUDA_VISIBLE_DEVICES": "-1",
        "NUMBA_ENABLE_PROFILING": "0",
        "NUMBA_DEBUGINFO": "0",
        **THREADS,
    }


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def parse_json(data):
    if len(data) > 4 * 1024**2:
        raise ValueError("JSON document exceeds bounded input size")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON token: " + value)

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)

    def finite(value, depth=0):
        if depth > 32:
            raise ValueError("JSON nesting exceeds bound")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite JSON number")
        if isinstance(value, dict):
            if len(value) > 10000:
                raise ValueError("JSON object exceeds bound")
            for child in value.values():
                finite(child, depth + 1)
        if isinstance(value, list):
            if len(value) > 10000:
                raise ValueError("JSON array exceeds bound")
            for child in value:
                finite(child, depth + 1)

    finite(result)
    return result


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("JSON document exceeds bounded input size")
    return parse_json(path.read_bytes())


def _git_cpu_limit():
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Git helper parent-death signal")
    if os.getppid() != parent:
        os._exit(125)
    resource.setrlimit(resource.RLIMIT_CPU, (1, 1))


def _git(args, text=False):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=text, timeout=5, preexec_fn=_git_cpu_limit
    )


def source_identity():
    head = _git(["rev-parse", "HEAD"], text=True).strip()
    tracked = _git(["ls-files", "-z"]).decode().split("\0")
    names = set(filter(None, tracked))
    names.update(str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py"))
    names.update(
        str(p.relative_to(ROOT))
        for p in (ROOT / "maintenance/detector_validation").iterdir()
        if p.is_file()
    )
    files = {}
    for name in sorted(names):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("nonregular source member: " + name)
        files[name] = digest(path)
    return {"simulator_commit": head, "files": files}


def runtime_identity():
    if sys.version_info[:3] != (3, 12, 3):
        raise ValueError("Python 3.12.3 required")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("reviewed platform is Linux x86_64 only")
    from packaging.requirements import Requirement

    pins = {}
    requested = {}
    for filename in (
        "requirements-python312.txt",
        "requirements-core.txt",
        "requirements-cloud.txt",
        "requirements-tensorflow.txt",
    ):
        for line in (ROOT / "maintenance" / filename).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r", "-c")):
                continue
            requirement = Requirement(line)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            specs = list(requirement.specifier)
            if len(specs) != 1 or specs[0].operator != "==":
                raise ValueError("exact applicable maintenance pin required")
            requested[requirement.name] = specs[0].version
    for name, version in sorted(requested.items()):
        dist = importlib.metadata.distribution(name)
        if dist.version != version:
            raise ValueError("pin mismatch: " + name)
        # Hash the actual installed distribution files, not just version labels.
        h = hashlib.sha256()
        count = 0
        for entry in sorted(dist.files or [], key=str):
            if str(entry).endswith((".pyc", ".pyo")) or "__pycache__" in str(entry):
                continue
            path = Path(dist.locate_file(entry))
            if not path.is_file():
                raise ValueError("missing installed distribution file: " + name)
            h.update(str(entry).encode() + b"\0")
            h.update(bytes.fromhex(digest(path)))
            count += 1
        if count == 0:
            raise ValueError("empty installed distribution manifest: " + name)
        pins[name] = {"version": version, "file_count": count, "sha256": h.hexdigest()}
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "invocation_executable": sys.executable,
        "executable_sha256": digest(Path(sys.executable).resolve()),
        "platform": platform.system(),
        "machine": platform.machine(),
        "pins": pins,
        "threads": THREADS,
        "gpu": "disabled",
    }
