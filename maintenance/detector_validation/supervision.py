"""Observable process-local supervision; configured limits are not exact CPU guarantees.

Linux pidfds provide exit readiness. wait4 is the sole reaper for the tracked
child. Aggregate waited-descendant usage is labeled separately from own CPU.
No scientific module is imported and no security setting is changed.
"""

import os
import sys
import ctypes
import math
import select
import resource
import selectors
import signal
import time
from pathlib import Path


def open_pidfd(pid):
    """Use an exported pidfd API; unsupported/denied errors are never bypassed."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0), "os.pidfd_open"
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = library.pidfd_open
    except AttributeError as error:
        raise RuntimeError("No exported pidfd_open binding") from error
    function.argtypes = (ctypes.c_int, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return descriptor, "libc.pidfd_open exported symbol, flags=0"


def snapshot():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "monotonic_seconds": time.monotonic(),
        "process_cpu_seconds": time.process_time(),
        "self_user_cpu_seconds": own.ru_utime,
        "self_system_cpu_seconds": own.ru_stime,
        "aggregate_reaped_children_user_seconds": children.ru_utime,
        "aggregate_reaped_children_system_seconds": children.ru_stime,
        "configured_cpu_rlimit": list(resource.getrlimit(resource.RLIMIT_CPU)),
        "configured_as_rlimit": list(resource.getrlimit(resource.RLIMIT_AS)),
        "process_clock_resolution_seconds": time.get_clock_info("process_time").resolution,
        "strict_cpu_enforcement_verified": False,
    }


def delta(before, after):
    return {
        key: after[key] - before[key]
        for key in (
            "monotonic_seconds",
            "process_cpu_seconds",
            "self_user_cpu_seconds",
            "self_system_cpu_seconds",
            "aggregate_reaped_children_user_seconds",
            "aggregate_reaped_children_system_seconds",
        )
    }


def resource_violations(observed, *, process_cpu_ceiling, wall_ceiling):
    """Postflight acceptance check, never presented as preventive enforcement."""
    result = []
    if observed["process_cpu_seconds"] > process_cpu_ceiling:
        result.append("observed_process_cpu_exceeded")
    if observed["monotonic_seconds"] > wall_ceiling:
        result.append("observed_wall_exceeded")
    return result


def arm_diagnostic_cpu_timer(total_cpu_ceiling, headroom):
    """Process-own CPU timer diagnostic only, with explicitly unproven lateness bound."""
    if (
        type(headroom) not in (int, float)
        or type(total_cpu_ceiling) not in (int, float)
        or not math.isfinite(headroom)
        or not math.isfinite(total_cpu_ceiling)
        or not 0 < headroom < total_cpu_ceiling
    ):
        raise ValueError("Explicit positive timer headroom required")
    if signal.getsignal(signal.SIGPROF) != signal.SIG_DFL:
        raise RuntimeError("SIGPROF disposition not default")
    if signal.SIGPROF in signal.pthread_sigmask(signal.SIG_BLOCK, []):
        raise RuntimeError("SIGPROF blocked")
    if signal.getitimer(signal.ITIMER_PROF) != (0.0, 0.0):
        raise RuntimeError("Existing process CPU timer")
    before = snapshot()
    remaining = total_cpu_ceiling - headroom - before["process_cpu_seconds"]
    if remaining <= 0:
        raise RuntimeError("Diagnostic CPU allocation already consumed")
    signal.setitimer(signal.ITIMER_PROF, remaining)
    timer_readback = signal.getitimer(signal.ITIMER_PROF)
    if not 0 < timer_readback[0] or timer_readback[1] != 0:
        signal.setitimer(signal.ITIMER_PROF, 0)
        raise RuntimeError("CPU timer setup/readback unsupported")
    return {
        "before": before,
        "requested_total_cpu_ceiling": total_cpu_ceiling,
        "reserved_headroom_seconds": headroom,
        "requested_timer_seconds": remaining,
        "observed_timer_seconds": list(timer_readback),
        "signal": "SIGPROF default termination",
        "scope": "own process all threads",
        "fork_inheritance": "timer not inherited",
        "exec_inheritance": "timer preserved",
        "strict_lateness_bound_established": False,
    }


class _Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [("interval", _Timespec), ("value", _Timespec)]


_WORKER_CPU_TIMER = None


def start_worker_cpu_timer(budget=170):
    """Arm once after exec and before native imports, at an absolute process CPU deadline.

    Restricted to the reviewed Linux LP64 libc ABI. A NULL sigevent selects the
    POSIX default SIGALRM notification. This timer is not inherited across exec.
    """
    global _WORKER_CPU_TIMER
    if _WORKER_CPU_TIMER is not None:
        raise RuntimeError("Worker CPU timer already armed")
    if (
        not sys.platform.startswith("linux")
        or ctypes.sizeof(ctypes.c_void_p) != 8
        or ctypes.sizeof(ctypes.c_long) != 8
    ):
        raise RuntimeError("Worker CPU timer requires reviewed Linux LP64 ABI")
    if signal.getsignal(
        signal.SIGALRM
    ) != signal.SIG_DFL or signal.SIGALRM in signal.pthread_sigmask(signal.SIG_BLOCK, []):
        raise RuntimeError("Worker SIGALRM unavailable")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0) or time.process_time() >= budget:
        raise RuntimeError("Existing wall timer or CPU allocation exhausted")
    libc = ctypes.CDLL(None, use_errno=True)
    create, arm, read = libc.timer_create, libc.timer_settime, libc.timer_gettime
    create.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    arm.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_Itimerspec), ctypes.c_void_p]
    read.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Itimerspec)]
    create.restype = arm.restype = read.restype = ctypes.c_int
    identifier = ctypes.c_void_p()
    if create(time.CLOCK_PROCESS_CPUTIME_ID, None, ctypes.byref(identifier)) != 0:
        raise OSError(ctypes.get_errno(), "process CPU timer_create failed")
    allocation = _Itimerspec(_Timespec(0, 0), _Timespec(int(budget), int((budget % 1) * 1e9)))
    if arm(identifier, 1, ctypes.byref(allocation), None) != 0:  # TIMER_ABSTIME
        raise OSError(ctypes.get_errno(), "absolute process CPU timer_settime failed")
    _WORKER_CPU_TIMER = (libc, identifier, budget)
    return worker_cpu_timer_state(budget)


class WorkerTimerObservationError(RuntimeError):
    """A rejected single observation, with only explicitly selected timer facts."""

    def __init__(self, observation):
        self.observation = observation
        super().__init__(
            "Worker absolute CPU timer absent, changed or blocked: "
            + ", ".join(observation["failed_checks"])
        )


def worker_cpu_timer_state(budget=170):
    """Read the same timer at most three times; only a wide bracket may be re-read."""
    rejected = []
    for attempt in range(3):
        try:
            state = _worker_cpu_timer_observation(budget)
        except WorkerTimerObservationError as error:
            if rejected and (
                error.observation["cpu_before_seconds"] < rejected[-1]["cpu_after_seconds"]
                or error.observation["remaining_seconds"] > rejected[-1]["remaining_seconds"]
            ):
                error.observation["failed_checks"].append("observation_trajectory")
            rejected.append(error.observation)
            if error.observation["failed_checks"] != ["observation_width"] or attempt == 2:
                error.observation_history = rejected
                raise
        except BaseException as error:
            # Preserve earlier reads even if the next native read fails outright.
            error.cpu_timer_observation_history = rejected
            raise
        else:
            if rejected and (
                state["cpu_before_seconds"] < rejected[-1]["cpu_after_seconds"]
                or state["remaining_seconds"] > rejected[-1]["remaining_seconds"]
            ):
                failure = dict(state, failed_checks=["observation_trajectory"])
                error = WorkerTimerObservationError(failure)
                error.observation_history = rejected + [failure]
                raise error
            state["observation_count"] = len(rejected) + 1
            state["observation_policy"] = "three-reads-width-only-v1"
            state["rejected_observations"] = rejected
            return state


def _worker_cpu_timer_observation(budget=170):
    """Bracket timer_gettime in its own process CPU clock and verify native signal state."""
    if _WORKER_CPU_TIMER is None or _WORKER_CPU_TIMER[2] != budget:
        raise RuntimeError("Worker CPU timer is not bound")
    libc, identifier, _ = _WORKER_CPU_TIMER
    fields = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    bit = 1 << (signal.SIGALRM - 1)
    default = not (int(fields["SigIgn"], 16) | int(fields["SigCgt"], 16)) & bit
    unblocked = not int(fields["SigBlk"], 16) & bit
    state = _Itimerspec()
    before = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    if libc.timer_gettime(identifier, ctypes.byref(state)) != 0:
        raise OSError(ctypes.get_errno(), "process CPU timer_gettime failed")
    after = time.clock_gettime(time.CLOCK_PROCESS_CPUTIME_ID)
    remaining = state.value.seconds + state.value.nanoseconds / 1e9
    interval = state.interval.seconds + state.interval.nanoseconds / 1e9
    python_default = signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
    checks = {
        "python_signal_default": python_default,
        "native_signal_default": default,
        "native_signal_unblocked": unblocked,
        "one_shot": interval == 0,
        "remaining_positive": 0 < remaining,
        "clock_order": 0 <= before <= after,
        "observation_width": after - before <= 0.01,
        "deadline_upper": before + remaining <= budget + 0.01,
        "deadline_lower": after + remaining >= budget - 0.01,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise WorkerTimerObservationError(
            {
                "schema": "worker-cpu-timer-rejection-v1",
                "failed_checks": failed,
                "budget_seconds": budget,
                "cpu_before_seconds": before,
                "cpu_after_seconds": after,
                "remaining_seconds": remaining,
                "interval_seconds": interval,
                "python_signal_default": python_default,
                "native_sigign": int(fields["SigIgn"], 16),
                "native_sigcgt": int(fields["SigCgt"], 16),
                "native_sigblk": int(fields["SigBlk"], 16),
                "observation_tolerance_seconds": 0.01,
            }
        )
    return {
        "timer": "CLOCK_PROCESS_CPUTIME_ID",
        "signal": "SIGALRM",
        "budget_seconds": budget,
        "remaining_seconds": remaining,
        "interval_seconds": interval,
        "observation_tolerance_seconds": 0.01,
        "native_default": bool(default),
        "native_unblocked": bool(unblocked),
        "cpu_before_seconds": before,
        "cpu_after_seconds": after,
        "scope": "absolute process user plus system CPU across threads; not descendants",
        "strict_lateness_bound_established": False,
    }


def terminate_group(proc, sig):
    # A reaped leader no longer reserves the numeric process-group identifier.
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def process_stat(pid):
    """Kernel process-own CPU counters before reaping; tick resolution explicit."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rindex(") ") + 2 :].split()
        ticks = os.sysconf("SC_CLK_TCK")
        own_user, own_system, waited_user, waited_system = map(int, fields[11:15])
        return {
            "state": fields[0],
            "own_user_ticks": own_user,
            "own_system_ticks": own_system,
            "waited_children_user_ticks": waited_user,
            "waited_children_system_ticks": waited_system,
            "clock_ticks_per_second": ticks,
            "own_cpu_seconds_tick_resolution": (own_user + own_system) / ticks,
            "resolution_seconds": 1 / ticks,
            "scope": "proc stat own process; NOT wait4 aggregate",
        }
    except (OSError, ValueError, IndexError) as error:
        return {"unavailable": type(error).__name__}


def cleanup_reap(proc, deadline):
    """Bounded wait4-only exceptional cleanup; normal supervision uses pidfd readiness."""
    terminate_group(proc, signal.SIGKILL)
    polls = 0
    while proc.returncode is None:
        polls += 1
        try:
            own_stat = process_stat(proc.pid)
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        except ChildProcessError:
            return {
                "reaped": False,
                "reason": "already_reaped_usage_missing",
                "polls": polls,
            }
        if pid:
            proc.returncode = os.waitstatus_to_exitcode(status)
            return {
                "reaped": True,
                "polls": polls,
                "usage": {
                    "user_cpu_seconds": usage.ru_utime,
                    "system_cpu_seconds": usage.ru_stime,
                    "cpu_seconds": usage.ru_utime + usage.ru_stime,
                    "max_rss_kib_linux": usage.ru_maxrss,
                    "scope": "wait4 tracked child including its waited descendants",
                    "own_process_stat_before_reap": own_stat,
                },
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"reaped": False, "reason": "cleanup_deadline", "polls": polls}
        select.select([], [], [], min(0.01, remaining))
    return {"reaped": True, "reason": "already_accounted", "polls": polls}


def supervise(
    proc,
    callbacks,
    *,
    started,
    wall_ceiling,
    cleanup_grace,
    own_cpu_start=None,
    own_cpu_tripwire=None,
    tripwire_reason="controller_chunk_cpu_reserve",
    child_cpu_tripwire=None,
    child_poll_seconds=0.02,
    child_gap_limit=1.0,
):
    """Wait for pipe/exit events or deadlines, retaining counters and wait4 usage.

    callbacks maps fd to (event mask, callback). callback returns False on EOF,
    True to keep watching, or a string failure reason. Callbacks must be bounded.
    Requested wall ceiling includes TERM/KILL cleanup. A delayed uninterruptible
    process may remain unreaped: report that gap instead of asserting cleanup.
    """
    if wall_ceiling <= 0 or cleanup_grace < 0:
        raise RuntimeError("Supported pidfd/deadline supervision required")
    if child_cpu_tripwire is not None and (
        not math.isfinite(child_cpu_tripwire)
        or child_cpu_tripwire <= 0
        or not 0 < child_poll_seconds <= 0.02
    ):
        raise ValueError("Invalid preventive child monitor")
    before = snapshot()
    origin_cpu = before["process_cpu_seconds"] if own_cpu_start is None else own_cpu_start
    hard_deadline = started + wall_ceiling
    grace = min(cleanup_grace, wall_ceiling / 4)
    soft_deadline = hard_deadline - grace
    selector = selectors.DefaultSelector()
    pidfd = None
    pidfd_backend = None
    reason = None
    trigger = None
    child_usage = None
    kill_at = None
    counters = {
        "select_calls": 0,
        "ready_events": 0,
        "callback_calls": 0,
        "empty_deadline_wakeups": 0,
        "term_sent": 0,
        "kill_sent": 0,
    }
    remaining = dict(callbacks)

    child_monitor = {
        "tripwire_cpu_seconds": child_cpu_tripwire,
        "poll_seconds": child_poll_seconds,
        "gap_limit_seconds": child_gap_limit,
        "polls": 0,
        "max_poll_gap_seconds": 0.0,
        "last": None,
        "strict_lateness_bound_established": False,
    }
    last_poll = time.monotonic()

    def stop(why, immediate=False):
        nonlocal reason, trigger, kill_at
        if reason is None:
            reason, trigger = why, snapshot()
            terminate_group(proc, signal.SIGKILL if immediate else signal.SIGTERM)
            counters["kill_sent" if immediate else "term_sent"] += 1
            kill_at = (
                None if immediate else min(hard_deadline - grace / 2, time.monotonic() + grace / 2)
            )
            # Keep bounded read callbacks active to preserve already emitted output.
            for fd, (mask, _) in list(remaining.items()):
                if mask & selectors.EVENT_WRITE:
                    selector.unregister(fd)
                    del remaining[fd]

    def reap():
        nonlocal child_usage, kill_at
        if kill_at is not None:
            terminate_group(proc, signal.SIGKILL)
            counters["kill_sent"] += 1
            kill_at = None
        own_stat = process_stat(proc.pid)
        # No poll()/wait() is permitted before this exclusive wait4.
        pid, status, usage = os.wait4(proc.pid, 0)
        if pid != proc.pid:
            raise RuntimeError("Unexpected reaped PID")
        proc.returncode = os.waitstatus_to_exitcode(status)
        child_usage = {
            "user_cpu_seconds": usage.ru_utime,
            "system_cpu_seconds": usage.ru_stime,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "max_rss_kib_linux": usage.ru_maxrss,
            "scope": "wait4 tracked child including its waited descendants",
            "own_process_stat_before_reap": own_stat,
        }

    try:
        pidfd, pidfd_backend = open_pidfd(proc.pid)
        selector.register(pidfd, selectors.EVENT_READ, "exit")
        for fd, (mask, _) in remaining.items():
            selector.register(fd, mask, "pipe")
        while child_usage is None or remaining or kill_at is not None:
            now = time.monotonic()
            if reason is None and child_usage is None and child_cpu_tripwire is not None:
                observed = process_stat(proc.pid)
                child_monitor["polls"] += 1
                child_monitor["max_poll_gap_seconds"] = max(
                    child_monitor["max_poll_gap_seconds"], now - last_poll
                )
                last_poll = now
                child_monitor["last"] = observed
                child_monitor["affinity_count"] = len(os.sched_getaffinity(proc.pid))
                if (
                    child_gap_limit is not None
                    and child_monitor["max_poll_gap_seconds"] > child_gap_limit
                ):
                    stop("child_cpu_monitor_gap", immediate=True)
                elif child_monitor["affinity_count"] > 5:
                    stop("child_cpu_affinity_exceeds_model", immediate=True)
                elif "unavailable" in observed:
                    stop("child_cpu_monitor_unavailable", immediate=True)
                elif observed["own_cpu_seconds_tick_resolution"] >= child_cpu_tripwire:
                    stop("child_cpu_headroom_tripwire", immediate=True)
            if reason is None and now >= soft_deadline:
                stop("wall_cleanup_reserve")
            if (
                reason is None
                and own_cpu_tripwire is not None
                and time.process_time() - origin_cpu >= own_cpu_tripwire
            ):
                stop(tripwire_reason)
            if child_usage is not None and not remaining:
                break
            if kill_at is not None and now >= kill_at:
                terminate_group(proc, signal.SIGKILL)
                counters["kill_sent"] += 1
                kill_at = None
            if now >= hard_deadline:
                terminate_group(proc, signal.SIGKILL)
                counters["kill_sent"] += 1
                if child_usage is not None:
                    break
                # No unbounded wait beyond the requested wall ceiling.
                events = selector.select(0)
                if not any(key.data == "exit" for key, _ in events):
                    reason = reason or "unreaped_at_wall_ceiling"
                    break
            else:
                deadline = min(
                    soft_deadline if reason is None else hard_deadline,
                    kill_at if kill_at is not None else hard_deadline,
                )
                counters["select_calls"] += 1
                wait = max(0.0, deadline - now)
                if child_cpu_tripwire is not None and child_usage is None:
                    wait = min(wait, child_poll_seconds)
                events = selector.select(wait)
                if not events:
                    counters["empty_deadline_wakeups"] += 1
            for key, mask in events:
                counters["ready_events"] += 1
                if key.data == "exit":
                    reap()
                    selector.unregister(pidfd)
                    if not remaining and kill_at is None:
                        break
                elif key.fd in remaining:
                    counters["callback_calls"] += 1
                    outcome = remaining[key.fd][1](mask)
                    if outcome is False:
                        selector.unregister(key.fd)
                        del remaining[key.fd]
                    elif isinstance(outcome, str):
                        stop(outcome)
    except BaseException as error:
        reason = "supervision:" + type(error).__name__ + ":" + str(error)[:512]
        terminate_group(proc, signal.SIGKILL)
        counters["kill_sent"] += 1
        if child_usage is None:
            try:
                cleaned = cleanup_reap(proc, hard_deadline)
                counters["exceptional_cleanup_polls"] = cleaned["polls"]
                child_usage = cleaned.get("usage")
                if not cleaned["reaped"]:
                    reason += ":" + cleaned["reason"]
            except BaseException as cleanup_error:
                reason += ":cleanup:" + type(cleanup_error).__name__
    finally:
        selector.close()
        if pidfd is not None:
            os.close(pidfd)
    after = snapshot()
    return {
        "exit_code": proc.returncode,
        "limit_failure": reason,
        "requested_wall_ceiling_seconds": wall_ceiling,
        "requested_cleanup_reserve_seconds": grace,
        "requested_own_cpu_tripwire_seconds": own_cpu_tripwire,
        "controller_before": before,
        "controller_trigger": trigger,
        "controller_after": after,
        "controller_delta": delta(before, after),
        "observed_since_launch_wall_seconds": after["monotonic_seconds"] - started,
        "tracked_child_wait4": child_usage,
        "cleanup_reaped": child_usage is not None,
        "cleanup_scope": "tracked child only; group signals do not establish descendant reaping",
        "descendant_termination_verified": False,
        "pidfd_backend": pidfd_backend,
        "counters": counters,
        "child_cpu_monitor": child_monitor,
        "strict_cpu_enforcement_verified": False,
    }
