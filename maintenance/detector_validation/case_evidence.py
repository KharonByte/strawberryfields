"""Small shared raw-case adapters; admission and failure semantics stay in callers."""

from collections import Counter

from .storage import encoded, sha


def worker_command(executable, case, chunk, *, module="fresh_worker", cpu_timer=False):
    command = [
        executable,
        "-B",
        "-m",
        "maintenance.detector_validation." + module,
        "--case-sha256",
        sha(encoded(case)),
        "--chunk",
        str(chunk),
    ]
    if cpu_timer:
        command.append("--cpu-timer")
    return command


def check_worker_command(actual, executable, case, chunk, **options):
    if actual != worker_command(executable, case, chunk, **options):
        raise ValueError("worker command/case binding mismatch")


def read_chunk_samples(read_array, prefix, case, *, check_domain=False):
    """Read frozen primary/Engine arrays using the caller's bounded NPY reader.

    ``prefix`` includes its trailing slash. Readers retain their existing shape
    and size checks; legacy physical-support rejection remains outside this helper.
    """
    block = read_array(prefix + "samples.npy", case["chunk_rows"], len(case["output_mode_order"]))
    if check_domain and any(
        any(v < 0 or (case["detector"] == "threshold" and v > 1) for v in row) for row in block
    ):
        raise ValueError("raw detector domain mismatch")
    if case["path"] == "local_engine_gaussian":
        secondary = read_array(
            prefix + "secondary.npy", case["chunk_rows"], len(case["measured_modes"])
        )
        mapping = [case["measured_modes"].index(m) for m in case["output_mode_order"]]
        if any(tuple(row[j] for j in mapping) != primary for row, primary in zip(secondary, block)):
            raise ValueError("Engine samples_dict mapping mismatch")
    return block


def evaluate_rows(case, rows, statistics, evaluator):
    """Count identical raw indicators and call the unchanged scientific evaluator."""
    from .reanalyse import raw_indicators

    counts = Counter(key for row in rows for key in raw_indicators(case, row))
    return evaluator(case, counts, len(rows), statistics)
