"""Observable process-local supervision; configured limits are not exact CPU guarantees.

Linux pidfds provide exit readiness. wait4 is the sole reaper for the tracked
child. Aggregate waited-descendant usage is labeled separately from own CPU.
No scientific module is imported and no security setting is changed.
"""

import os
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


def terminate_group(proc, sig):
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
            return {"reaped": False, "reason": "already_reaped_usage_missing", "polls": polls}
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
):
    """Wait for pipe/exit events or deadlines, retaining counters and wait4 usage.

    callbacks maps fd to (event mask, callback). callback returns False on EOF,
    True to keep watching, or a string failure reason. Callbacks must be bounded.
    Requested wall ceiling includes TERM/KILL cleanup. A delayed uninterruptible
    process may remain unreaped: report that gap instead of asserting cleanup.
    """
    if wall_ceiling <= 0 or cleanup_grace < 0:
        raise RuntimeError("Supported pidfd/deadline supervision required")
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

    def stop(why):
        nonlocal reason, trigger, kill_at
        if reason is None:
            reason, trigger = why, snapshot()
            terminate_group(proc, signal.SIGTERM)
            counters["term_sent"] += 1
            kill_at = min(hard_deadline - grace / 2, time.monotonic() + grace / 2)
            for fd in list(remaining):
                selector.unregister(fd)
                del remaining[fd]

    def reap():
        nonlocal child_usage
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
            if reason is None and now >= soft_deadline:
                stop("wall_cleanup_reserve")
            if (
                reason is None
                and own_cpu_tripwire is not None
                and time.process_time() - origin_cpu >= own_cpu_tripwire
            ):
                stop(tripwire_reason)
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
                events = selector.select(max(0.0, deadline - now))
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
        "strict_cpu_enforcement_verified": False,
    }
