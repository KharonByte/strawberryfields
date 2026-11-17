"""Single fresh-process detector chunk capture; invoke only through approved supervisor.

This worker does not grant collection approval or enforce global resource budgets.
The supervisor must freeze/review code, cases, dependencies and acceptance, enforce
limits, and reject any missing/failed chunk. No scientific acceptance occurs here.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import resource
import warnings
import sys
import time
import traceback
from pathlib import Path

from .stream import EvidenceStream

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "vacuum",
    "coherent_product",
    "thermal",
    "squeezed_vacuum",
    "two_mode_squeezed",
    "coherent_interference",
}
PATHS = {"gaussian_direct", "local_engine_gaussian", "apps_sample"}
DENIED_OPERATIONS = []


def digest(path):
    """Hash an explicit local file, never search or collect ambient secrets."""
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def preflight():
    """Verify actual local caps; approval remains a supervisor/reviewer decision."""
    caps = {}
    for name, identifier, maximum in (
        ("address_space", resource.RLIMIT_AS, 14 * 1024**3),
        ("cpu_seconds", resource.RLIMIT_CPU, 120),
    ):
        soft, hard = resource.getrlimit(identifier)
        if soft == resource.RLIM_INFINITY or not 0 < soft <= maximum:
            raise RuntimeError("Missing or excessive " + name + " resource limit")
        if hard == resource.RLIM_INFINITY or not 0 < hard <= maximum:
            raise RuntimeError("Missing or excessive hard " + name + " resource limit")
        caps[name] = {"soft": soft, "hard": hard}
    for name, identifier in (
        ("file_size", resource.RLIMIT_FSIZE),
        ("core_size", resource.RLIMIT_CORE),
    ):
        soft, hard = resource.getrlimit(identifier)
        if soft != 0 or hard != 0:
            raise RuntimeError("Expected zero " + name + " resource limit")
        caps[name] = {"soft": soft, "hard": hard}
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if (
        soft == resource.RLIM_INFINITY
        or hard == resource.RLIM_INFINITY
        or not 0 < soft <= 64
        or not 0 < hard <= 64
    ):
        raise RuntimeError("Missing or excessive open_files resource limit")
    caps["open_files"] = {"soft": soft, "hard": hard}
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise RuntimeError("Missing single-thread setting: " + name)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("GPU-disabled environment missing")
    return caps


def metadata_bytes(value):
    encoded = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(encoded) > 262144:
        raise ValueError("Metadata exceeds 262144-byte bound")
    return encoded


def array_bytes(np, array, maximum):
    if array.nbytes + 128 > maximum:
        raise ValueError("Array exceeds fixed evidence bound")
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer, np.ascontiguousarray(array), version=(1, 0), allow_pickle=False
    )
    payload = buffer.getvalue()
    if len(payload) > maximum:
        raise ValueError("Encoded array exceeds evidence bound")
    return payload


def deny_operation(event, args, reason):
    """Retain bounded denied-operation evidence, with no locals or environment dump."""
    if len(DENIED_OPERATIONS) >= 100:
        raise RuntimeError("Denied-operation record bound exceeded")
    record = {
        "event": event,
        "reason": reason,
        "stack": [
            {"file": frame.filename, "function": frame.name, "line": frame.lineno}
            for frame in traceback.extract_stack(limit=20)
        ],
    }
    if event == "open":
        path, mode, flags = args
        record.update(
            path=path if isinstance(path, (str, int)) else "<non-string path>",
            mode=mode,
            flags=flags,
        )
    if event == "os.mkdir":
        path, mode, dir_fd = args
        record.update(
            path=path if isinstance(path, (str, int)) else "<non-string path>",
            mode=mode,
            dir_fd=dir_fd,
        )
    DENIED_OPERATIONS.append(record)
    raise PermissionError(reason)


KERAS_DISABLED_HOME = "/dev/null/detector-keras-disabled"
KERAS_CONFIG_SOURCES = (
    (
        "tensorflow-cpu",
        "2.21.0",
        "tensorflow/python/keras/backend.py",
        "9853e5675b1bd30adb181b720bc371fae3e484cc00bf2928a2c62d3351cedecc",
        6256,
        6270,
    ),
    (
        "keras",
        "3.15.1",
        "keras/src/backend/config.py",
        "1bc60421f5357e0903c36fff39544181d1ed1acb8431918f7b61c8cb5d5117be",
        333,
        347,
    ),
)


def recognized_keras_denial(record):
    """Recognize only denied, pinned-source optional config initialization."""
    event = record.get("event")
    if os.environ.get("KERAS_HOME") != KERAS_DISABLED_HOME:
        return False
    if event == "os.mkdir":
        arguments_match = (
            record.get("path") == KERAS_DISABLED_HOME
            and type(record.get("mode")) is int
            and record["mode"] == 0o777
            and type(record.get("dir_fd")) is int
            and record["dir_fd"] == -1
        )
        line_index = 4
    elif event == "open":
        arguments_match = (
            record.get("path") == KERAS_DISABLED_HOME + "/keras.json"
            and record.get("mode") == "w"
            and type(record.get("flags")) is int
            and record["flags"] == os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC
        )
        line_index = 5
    else:
        return False
    if not arguments_match:
        return False
    for spec in KERAS_CONFIG_SOURCES:
        package, version, relative, expected_hash, _, _ = spec
        distribution = importlib.metadata.distribution(package)
        source = Path(distribution.locate_file(relative)).resolve(strict=True)
        origin_match = any(
            frame.get("file") == str(source)
            and frame.get("function") == "<module>"
            and frame.get("line") == spec[line_index]
            for frame in record["stack"]
        )
        if origin_match:
            return (
                distribution.version == version
                and hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash
            )
    return False


def validate_denied_operations(records=None):
    """Accept only reviewed *denied* discovery attempts; grant no operation access."""
    records = DENIED_OPERATIONS if records is None else records
    if len(records) > 100:
        raise RuntimeError("Denied-operation record bound exceeded")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("stack"), list):
            raise RuntimeError("Malformed denied-operation record")
        stack = record["stack"]
        if any(not isinstance(frame, dict) for frame in stack):
            raise RuntimeError("Malformed denied-operation stack")
        numpy_probe = record.get("event") == "subprocess.Popen" and any(
            frame.get("function") == "check_support_sve"
            and frame.get("file", "").endswith("/numpy/testing/_private/utils.py")
            for frame in stack
        )
        flags = record.get("flags")
        devnull_probe = (
            record.get("event") == "open"
            and record.get("path") == "/dev/null"
            and isinstance(flags, int)
            and flags & os.O_ACCMODE == os.O_RDWR
            and flags & ~(os.O_RDWR | os.O_CLOEXEC) == 0
            and any(
                frame.get("function") == "from_subprocess"
                and frame.get("file", "").endswith("/platform.py")
                for frame in stack
            )
        )
        if not (numpy_probe or devnull_probe or recognized_keras_denial(record)):
            raise RuntimeError("Unreviewed denied operation invalidates capture")
    return {"validated": True, "count": len(records), "write_or_spawn_access_granted": False}


def reject_python_writes(event, args):
    """Cooperative Python guard, NOT native-code filesystem containment."""
    if event in {"os.fork", "os.forkpty", "os.posix_spawn", "subprocess.Popen"}:
        deny_operation(event, args, "Python subprocess creation rejected")
    if event == "open":
        _, mode, flags = args
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int)
            and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        ):
            deny_operation(event, args, "Python filesystem write rejected")
    if event in {
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.link",
        "os.symlink",
        "os.chmod",
        "os.chown",
        "os.truncate",
    }:
        deny_operation(event, args, "Python filesystem mutation rejected")


def validate_case(case, chunk):
    """Reject unsupported settings before importing the candidate."""
    if case["path"] not in PATHS or case["family"] not in FAMILIES:
        raise ValueError("Unsupported path/family")
    if case["detector"] not in {"pnr", "threshold"} or case["required"] is not True:
        raise ValueError("Unsupported detector or nonrequired case")
    modes = case["modes"]
    measured = case["measured_modes"]
    if type(modes) is not int or modes not in {1, 2, 3}:
        raise ValueError("Unsupported mode count")
    if (
        not measured
        or len(measured) != len(set(measured))
        or any(type(m) is not int or not 0 <= m < modes for m in measured)
    ):
        raise ValueError("Invalid measured mode map")
    expected_order = measured if case["path"] == "gaussian_direct" else sorted(measured)
    if case["output_mode_order"] != expected_order:
        raise ValueError("Output mode map disagrees with actual path")
    if len(case["eta"]) != modes or any(
        type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 1 for x in case["eta"]
    ):
        raise ValueError("Invalid efficiencies")
    if (case["shots"], case["chunk_rows"], case["chunk_count"]) != (100000, 10000, 10):
        raise ValueError("Unfrozen sampling budget")
    seeds = case["chunk_seeds"]
    if (
        len(seeds) != 10
        or len(set(seeds)) != 10
        or not 0 <= chunk < 10
        or any(type(x) is not int or not 0 <= x < 2**32 for x in seeds)
    ):
        raise ValueError("Invalid chunk/seeds")
    if case["hbar_backend"] != 2 or case["hbar_physical"] != (
        2 if case["path"] == "apps_sample" else 8
    ):
        raise ValueError("Unsupported units")
    if case["path"] == "apps_sample":
        p = case["parameters"]
        if (
            case["family"] != "two_mode_squeezed"
            or modes != 2
            or measured != [0, 1]
            or case["eta"][0] != case["eta"][1]
            or case["app_loss"] != 1 - case["eta"][0]
            or p["graph"] != "[[0,w],[w,0]]"
            or p["w_expression"] not in {"1", "exp(0.7i)"}
            or p["total_mean_photons"] != "2/9"
        ):
            raise ValueError("Unsupported apps graph/loss configuration")


def moments(np, case):
    """Independent pre-loss physical moments, with no SF/Walrus helpers."""
    n = case["modes"]
    h = case["hbar_physical"]
    family = case["family"]
    p = case["parameters"]
    cov = (h / 2) * np.eye(2 * n)
    mean = np.zeros(2 * n)
    if family in {"coherent_product", "coherent_interference"}:
        alpha = np.asarray(p["alpha_re_im"], dtype=float)
        if alpha.shape != (n, 2):
            raise ValueError("Wrong coherent preparation shape")
        mean = math.sqrt(2 * h) * np.concatenate((alpha[:, 0], alpha[:, 1]))
    elif family == "thermal":
        if n != 1:
            raise ValueError("Thermal case must be single mode")
        cov *= 2 * p["mean_photons"] + 1
    elif family == "squeezed_vacuum":
        if n != 1:
            raise ValueError("Squeezed case must be single mode")
        r = math.atanh(math.sqrt(p["tanh_r_squared"]))
        phi = p["phi"]
        c, s = math.cosh(2 * r), math.sinh(2 * r)
        cov = (h / 2) * np.array(
            [
                [c - math.cos(phi) * s, -math.sin(phi) * s],
                [-math.sin(phi) * s, c + math.cos(phi) * s],
            ]
        )
    elif family == "two_mode_squeezed":
        if n != 2:
            raise ValueError("TMS case must have two modes")
        population = p["q"] / (1 - p["q"])
        amplitude = math.sqrt(population * (population + 1))
        m = np.array(
            [
                [0, amplitude * complex(math.cos(p["phi"]), math.sin(p["phi"]))],
                [amplitude * complex(math.cos(p["phi"]), math.sin(p["phi"])), 0],
            ]
        )
        diagonal = (h / 2) * (2 * population + 1) * np.eye(n)
        cov = np.block([[diagonal + h * m.real, h * m.imag], [h * m.imag, diagonal - h * m.real]])
    elif family != "vacuum":
        raise ValueError("Unknown preparation")
    omega = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError("Nonfinite preparation")
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-12):
        raise ValueError("Non-symmetric preparation")
    minimum = float(np.linalg.eigvalsh(cov + 1j * h * omega / 2).min())
    if minimum < -1e-12:
        raise ValueError("Unphysical preparation within bounded arithmetic guard")
    return mean, cov, minimum


def numeric(np, value):
    """Serialize only bounded numeric operation parameters, never repr arbitrary objects."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "biufc" or arr.size > 256 or not np.isfinite(arr).all():
        raise ValueError("Non-numeric/oversized/unbound compiled parameter")
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "real": arr.real.tolist(),
        "imag": arr.imag.tolist() if np.iscomplexobj(arr) else None,
    }


def circuit_record(np, program):
    if len(program.circuit) > 100:
        raise ValueError("Unexpectedly large circuit")
    return [
        {
            "operation": type(cmd.op).__name__,
            "modes": [r.ind for r in cmd.reg],
            "parameters": [numeric(np, p) for p in cmd.op.p],
        }
        for cmd in program.circuit
    ]


def counts(np, raw, case):
    arr = np.asarray(raw)
    if arr.shape != (case["chunk_rows"], len(case["output_mode_order"])):
        raise ValueError("Returned sample shape mismatch")
    if arr.dtype.kind not in "iuf" or not np.isfinite(arr).all():
        raise ValueError("Returned counts must be finite real numbers")
    # Strict upper bound avoids float64 representation of int64 max rounding to 2**63.
    if np.any(arr < 0) or np.any(arr >= 2**63) or np.any(arr != np.floor(arr)):
        raise ValueError("Count conversion would not be lossless")
    if case["detector"] == "threshold" and np.any(arr > 1):
        raise ValueError("Nonbinary threshold return")
    converted = arr.astype("<i8")
    if not np.array_equal(converted, arr):
        raise ValueError("Count conversion changed values")
    return converted, str(arr.dtype)


def capture(case, chunk, evidence, metadata):
    # Imports are deliberately lazy: reading/importing worker does not import a simulator.
    sys.path.insert(0, str(ROOT))
    import numpy as np

    # Installed Numba source must be frozen/reviewed by the supervisor.
    # No JIT/target/precision option changes: retain the default in-memory cache.
    import numba
    from numba.core import dispatcher as numba_dispatcher
    from numba.core.caching import NullCache

    cache_records = []
    cache_dispatchers = []
    original_enable = numba_dispatcher.Dispatcher.enable_caching

    def memory_only_walrus_cache(dispatcher):
        module = dispatcher.py_func.__module__
        if not module.startswith("thewalrus."):
            raise RuntimeError("Unexpected non-Walrus file-cache request")
        if type(dispatcher._cache) is not NullCache:
            raise RuntimeError("Unexpected Numba initial cache")
        if len(cache_records) >= 256:
            raise RuntimeError("Cache-policy record bound exceeded")
        options = dict(dispatcher.targetoptions)
        cache_dispatchers.append((dispatcher, options))
        cache_records.append(
            {
                "module": module,
                "function": dispatcher.py_func.__qualname__,
                "targetoptions": options,
                "initial_cache": type(dispatcher._cache).__name__,
            }
        )

    metadata["cache_policy"] = {
        "change": "child-local Dispatcher.enable_caching retains NullCache for Walrus only",
        "jit_disabled": bool(numba.config.DISABLE_JIT),
        "compiler_target_options_changed": None,
        "source": digest(numba_dispatcher.__file__),
        "caching_source": digest(Path(numba_dispatcher.__file__).with_name("caching.py")),
        "numba_version": numba.__version__,
        "functions": cache_records,
        "original_method_name": original_enable.__qualname__,
        "native_filesystem_containment": False,
    }

    def validate_runtime_policy(stage):
        audit_status = validate_denied_operations()
        disabled = bool(numba.config.DISABLE_JIT)
        changed = any(
            dict(dispatcher.targetoptions) != options for dispatcher, options in cache_dispatchers
        )
        null_caches = all(
            type(dispatcher._cache) is NullCache for dispatcher, _ in cache_dispatchers
        )
        metadata["cache_policy"].update(
            jit_disabled=disabled,
            compiler_target_options_changed=changed,
            all_intercepted_caches_null=null_caches,
            intercepted_dispatchers=len(cache_dispatchers),
        )
        if disabled or changed or not null_caches:
            raise RuntimeError("Runtime JIT/cache policy mismatch")
        metadata.setdefault("runtime_policy_checks", {})[stage] = audit_status

    numba_dispatcher.Dispatcher.enable_caching = memory_only_walrus_cache
    import strawberryfields as sf
    import thewalrus
    from strawberryfields.apps import sample as apps_sample
    from strawberryfields.backends.gaussianbackend.backend import GaussianBackend
    from strawberryfields.engine import LocalEngine
    import thewalrus.samples as walrus_samples

    if Path(sf.__file__).resolve().parent != ROOT / "strawberryfields":
        raise RuntimeError("Candidate import is not from reviewed source root")
    imported_paths = [module.__file__ for module in (np, sf, thewalrus, walrus_samples)]
    imported_paths += [
        str(ROOT / name)
        for name in (
            "strawberryfields/engine.py",
            "strawberryfields/ops.py",
            "strawberryfields/apps/sample.py",
            "strawberryfields/decompositions.py",
            "strawberryfields/backends/gaussianbackend/backend.py",
            "strawberryfields/backends/gaussianbackend/gaussiancircuit.py",
            "strawberryfields/compilers/gaussian.py",
        )
    ]
    from .provenance import SINGLE_MODE_SOURCE, sampler_configuration

    single_mode_available = (ROOT / SINGLE_MODE_SOURCE).is_file()
    if single_mode_available:
        module = sys.modules.get("strawberryfields.backends.gaussianbackend.sampling")
        if module is None or Path(module.__file__).resolve() != ROOT / SINGLE_MODE_SOURCE:
            raise RuntimeError("single-mode sampler import identity missing")
        imported_paths.append(str(ROOT / SINGLE_MODE_SOURCE))
    metadata["imports"] = [digest(path) for path in imported_paths]
    metadata["versions"] = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "thewalrus", "numba", "llvmlite")
    }
    metadata["sampler_defaults"] = sampler_configuration(case, single_mode_available)
    metadata["measurement_execution"] = {"layer": case["path"], "status": "planned"}
    old_hbar = sf.hbar
    sf.hbar = case["hbar_physical"]
    np.random.seed(case["chunk_seeds"][chunk])

    def rng_digest():
        state = np.random.get_state()
        return hashlib.sha256(
            state[0].encode() + state[1].tobytes() + json.dumps(state[2:]).encode()
        ).hexdigest()

    metadata["rng"] = {
        "api": "numpy.random legacy RandomState",
        "before_sha256": rng_digest(),
    }
    original, engine, result = None, None, None
    try:
        if case["path"] == "gaussian_direct":
            mean, cov, minimum = moments(np, case)
            metadata["pre_loss_physical_moments"] = {
                "mean": mean.tolist(),
                "covariance": cov.tolist(),
                "uncertainty_min_eigenvalue": minimum,
                "guard_tolerance": 1e-12,
            }
            backend = GaussianBackend()
            backend.begin_circuit(case["modes"])
            backend.prepare_gaussian_state(
                mean * math.sqrt(2 / sf.hbar),
                cov * 2 / sf.hbar,
                list(range(case["modes"])),
            )
            if case["family"] == "coherent_interference":
                p = case["parameters"]
                backend.beamsplitter(p["beamsplitter_theta"], p["beamsplitter_phi"], 0, 1)
            for mode, eta in enumerate(case["eta"]):
                backend.loss(eta, mode)
            indices = case["measured_modes"] + [m + case["modes"] for m in case["measured_modes"]]
            metadata["actual_pre_detector_reduced_moments"] = {
                "mean": backend.circuit.smeanxp()[indices].tolist(),
                "covariance": backend.circuit.scovmatxp()[np.ix_(indices, indices)].tolist(),
                "hbar": 2,
                "ordering": "xxpp_requested_modes",
            }
            metadata["direct_call_sequence"] = [
                {"method": "begin_circuit", "modes": case["modes"]},
                {
                    "method": "prepare_gaussian_state",
                    "mean_internal": (mean * math.sqrt(2 / sf.hbar)).tolist(),
                    "covariance_internal": (cov * 2 / sf.hbar).tolist(),
                    "modes": list(range(case["modes"])),
                },
            ]
            if case["family"] == "coherent_interference":
                metadata["direct_call_sequence"].append(
                    {
                        "method": "beamsplitter",
                        "modes": [0, 1],
                        "theta": case["parameters"]["beamsplitter_theta"],
                        "phi": case["parameters"]["beamsplitter_phi"],
                    }
                )
            metadata["direct_call_sequence"] += [
                {"method": "loss", "mode": m, "eta": eta} for m, eta in enumerate(case["eta"])
            ]
            metadata["direct_call_sequence"].append(
                {
                    "execution_status": "planned",
                    "method": (
                        "measure_fock" if case["detector"] == "pnr" else "measure_threshold"
                    ),
                    "modes": case["measured_modes"],
                    "shots": case["chunk_rows"],
                }
            )
            measure = (
                backend.measure_fock if case["detector"] == "pnr" else backend.measure_threshold
            )
            validate_runtime_policy("before_capture")
            metadata["direct_call_sequence"][-1]["execution_status"] = "api_invocation_entered"
            metadata["measurement_execution"] = {
                "layer": "gaussian_backend_api",
                "status": "entered",
            }
            raw = measure(case["measured_modes"], shots=case["chunk_rows"])
            metadata["direct_call_sequence"][-1]["execution_status"] = "api_invocation_returned"
            metadata["measurement_execution"]["status"] = "returned"
        elif case["path"] == "local_engine_gaussian":
            p = case["parameters"]
            original = sf.Program(case["modes"])
            with original.context as q:
                family = case["family"]
                if family == "coherent_product":
                    mean, cov, minimum = moments(np, case)
                    sf.ops.Gaussian(cov, mean, decomp=False) | q
                    metadata["uncertainty_min_eigenvalue"] = minimum
                elif family == "vacuum":
                    sf.ops.Vacuum() | q[0]
                elif family == "thermal":
                    sf.ops.Thermal(p["mean_photons"]) | q[0]
                elif family == "squeezed_vacuum":
                    sf.ops.Squeezed(math.atanh(math.sqrt(p["tanh_r_squared"])), p["phi"]) | q[0]
                elif family == "two_mode_squeezed":
                    sf.ops.S2gate(math.atanh(math.sqrt(p["q"])), p["phi"]) | q
                elif family == "coherent_interference":
                    for mode, pair in enumerate(p["alpha_re_im"]):
                        sf.ops.Coherent(math.hypot(*pair), math.atan2(pair[1], pair[0])) | q[mode]
                    sf.ops.BSgate(p["beamsplitter_theta"], p["beamsplitter_phi"]) | q
                else:
                    raise ValueError("Unsupported LocalEngine preparation")
                for mode, eta in enumerate(case["eta"]):
                    sf.ops.LossChannel(eta) | q[mode]
                detector = (
                    sf.ops.MeasureFock if case["detector"] == "pnr" else sf.ops.MeasureThreshold
                )
                detector() | [q[mode] for mode in case["measured_modes"]]
            metadata["original_circuit"] = circuit_record(np, original)
            engine = sf.LocalEngine("gaussian")
            validate_runtime_policy("before_capture")
            metadata["measurement_execution"] = {
                "layer": "engine_run",
                "status": "entered",
                "limit": "Engine invocation is not proof of backend sampler entry",
            }
            result = engine.run(original, shots=case["chunk_rows"])
            metadata["measurement_execution"]["status"] = "returned"
            raw = result.samples
        else:
            # Narrow profile hook: exact known function/caller codes; never enumerate locals.
            captured = []
            target_code = LocalEngine.run.__code__
            caller_code = apps_sample.sample.__code__
            old_profile = sys.getprofile()
            if old_profile is not None:
                raise RuntimeError("Existing profiler prevents safe exact-call capture")

            def profiler(frame, event, arg):
                if (
                    event == "call"
                    and frame.f_code is target_code
                    and frame.f_back is not None
                    and frame.f_back.f_code is caller_code
                ):
                    if captured:
                        raise RuntimeError("Unexpected repeated apps engine call")
                    captured.append((frame.f_locals["self"], frame.f_locals["program"]))
                    metadata["original_circuit"] = circuit_record(np, captured[0][1])

            p = case["parameters"]
            phase = 0 if p["w_expression"] == "1" else 0.7
            w = complex(math.cos(phase), math.sin(phase))
            adjacency = np.array([[0, w], [w, 0]])
            if phase == 0:
                adjacency = adjacency.real
            validate_runtime_policy("before_capture")
            sys.setprofile(profiler)
            try:
                metadata["measurement_execution"] = {
                    "layer": "apps_sample",
                    "status": "entered",
                    "limit": "Application invocation is not proof of backend sampler entry",
                }
                raw = apps_sample.sample(
                    adjacency,
                    2 / 9,
                    case["chunk_rows"],
                    case["detector"] == "threshold",
                    case["app_loss"],
                )
                metadata["measurement_execution"]["status"] = "returned"
            finally:
                sys.setprofile(old_profile)
            if len(captured) != 1:
                raise RuntimeError("Actual apps engine invocation not captured")
            engine, original = captured[0]
            metadata["apps_return_container"] = "list"
        try:
            values, dtype = counts(np, raw, case)
        except (ValueError, TypeError, OverflowError):
            # Preserve bounded numeric returned values in their original dtype/shape.
            # Object/ragged/oversized payloads must never be serialized with pickle.
            returned = np.asarray(raw)
            if (
                returned.dtype.kind in "biufc"
                and returned.dtype.itemsize <= 16
                and returned.ndim == 2
                and returned.shape[0] <= 10000
                and returned.shape[1] <= 3
            ):
                invalid_receipt = evidence.emit(4, array_bytes(np, returned, 480128))
                metadata["invalid_return"] = dict(
                    invalid_receipt,
                    original_dtype=str(returned.dtype),
                    original_shape=list(returned.shape),
                )
            else:
                metadata["invalid_return"] = {
                    "preserved": False,
                    "reason": "not a bounded numeric 2-D array",
                }
            raise
        raw_receipt = evidence.emit(2, array_bytes(np, values, 160128))
        metadata.update(original_dtype=dtype, shape=list(values.shape), raw_file=raw_receipt)
        if engine is not None:
            if len(engine.run_progs) != 1:
                raise RuntimeError("Unexpected compiled program count")
            metadata["compiled_circuit"] = circuit_record(np, engine.run_progs[0])
            records = engine.samples_dict
            if set(records) != set(case["measured_modes"]) or any(
                len(v) != 1 for v in records.values()
            ):
                raise RuntimeError("Unexpected samples_dict mode/history shape")
            requested = np.column_stack([records[m][0] for m in case["measured_modes"]])
            requested_case = dict(case, output_mode_order=case["measured_modes"])
            copied, _ = counts(np, requested, requested_case)
            indices = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
            if not np.array_equal(copied[:, indices], values):
                raise RuntimeError("samples_dict disagrees with mapped final return")
            if case["path"] == "local_engine_gaussian":
                metadata["samples_dict_file"] = evidence.emit(3, array_bytes(np, copied, 160128))
            metadata["samples_dict_consistency"] = "elementwise matched in declared mode order"
    finally:
        metadata["rng"]["after_sha256"] = rng_digest()
        metadata["imports_after"] = [digest(path) for path in imported_paths]
        sf.hbar = old_hbar
    validate_runtime_policy("after_capture")
    metadata["guard_policy_validated"] = True
    metadata["write_or_spawn_access_granted"] = False
    if metadata["imports_after"] != metadata["imports"]:
        raise RuntimeError("Imported source changed during capture")


def started_metadata(chunk, started):
    metadata = {
        "schema": "terminal-detector-chunk-v1",
        "output_type": "candidate_raw_chunk",
        "status": "started",
        "chunk_index": chunk,
        "worker": digest(__file__),
        "evidence_type": "candidate_terminal_samples",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source_root": str(ROOT),
        "started_unix": started,
        "denied_python_operations": DENIED_OPERATIONS,
        "python_write_or_spawn_access_granted": False,
        "native_filesystem_containment": False,
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-sha256", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    args = parser.parse_args()
    evidence = EvidenceStream(int(os.environ["DETECTOR_EVIDENCE_FD"]))
    if not sys.dont_write_bytecode:
        raise RuntimeError("Worker requires Python -B before imports")
    sys.addaudithook(reject_python_writes)
    started = time.time()
    metadata = started_metadata(args.chunk, started)
    evidence.emit(1, metadata_bytes(metadata))
    try:
        metadata["verified_resource_limits"] = preflight()
        case_bytes = sys.stdin.buffer.read(32769)
        if len(case_bytes) > 32768:
            raise ValueError("Case document exceeds size limit")
        case_sha256 = hashlib.sha256(case_bytes).hexdigest()
        if case_sha256 != args.case_sha256:
            raise ValueError("Case bytes do not match supervisor-provided SHA-256")
        case = json.loads(case_bytes)
        validate_case(case, args.chunk)
        metadata.update(
            case=case,
            case_id=case["id"],
            case_file={
                "input": "stdin",
                "bytes": len(case_bytes),
                "sha256": case_sha256,
            },
            bound_args={},
            compile_options="backend default gaussian",
            requested_mode_order=case["measured_modes"],
            seed=case["chunk_seeds"][args.chunk],
            rows=case["chunk_rows"],
            output_mode_order=case["output_mode_order"],
        )
        # Only the known Gaussian conditional-state warning is emitted by these paths;
        # bound retained warning records without changing filter behavior of apps.sample.
        warning_records = []
        previous_showwarning = warnings.showwarning

        def record_warning(message, category, filename, lineno, file=None, line=None):
            if len(warning_records) >= 100:
                raise RuntimeError("Warning record limit exceeded")
            warning_records.append(
                {
                    "category": category.__name__,
                    "message": str(message)[:2048],
                    "filename": filename,
                    "line": lineno,
                }
            )

        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = record_warning
            try:
                capture(case, args.chunk, evidence, metadata)
            finally:
                metadata["warnings"] = warning_records
                warnings.showwarning = previous_showwarning
        metadata["status"] = "captured_not_accepted"
    except BaseException as error:
        # Keep partial arrays and earlier receipts. No traceback or ambient local-variable dump.
        metadata.update(
            status="error",
            error_type=type(error).__name__,
            error_message=str(error)[:2048],
        )
        raise
    finally:
        metadata["elapsed_seconds"] = time.time() - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        metadata["resource_usage"] = {
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "max_rss_kib_linux": usage.ru_maxrss,
        }
        evidence.emit(5, metadata_bytes(metadata))


if __name__ == "__main__":
    main()
