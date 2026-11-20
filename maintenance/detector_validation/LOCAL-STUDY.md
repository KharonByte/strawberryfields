# Prospective local full study resource policy

The `fresh` command's `scientific-v2` profile preserves every case, row count,
chunk, seed and statistical criterion from `scientific`: 68 cases, 680 chunks,
6.8 million rows and 1,022 criteria. It changes operational allowances only.
The old `scientific` and `operational` profiles and hosted release policy remain
available with their original resources. Historical reproduction requires its
original source/runtime release; this profile never reclassifies old failures.

This is a supported local path, separate from hosted orchestration. Hosted studies
still require the owner-merged reviewed release described in [CI.md](CI.md).
A local release must bind its exact source, runtime, accounting and three actual
independent reviews before collection. This document is not a collection release.

## Resource basis and enforcement

Retained corrected-detector cold workers reached 149.260006 CPU seconds and
149.569951 wall seconds; 32 of the 141 recovery workers exceeded 120 CPU seconds.
The earlier run also retained a 128.015766-second worker under a configured
120-second limit. Configuration therefore does not establish exact enforcement.
A new profile provides the following bounded prospective allowances:

| Scope | Allowance / action |
| --- | --- |
| Worker | 180 CPU seconds, 240 wall seconds including cleanup |
| Child CPU monitor | Poll process-own kernel counters at most every 20 ms; kill at 170 CPU seconds |
| Monitor admission | At most five exposed CPUs; missing counters or observed poll gap over one second abort |
| Controller per chunk | Trip at 18 CPU seconds; reject observed use above 20 |
| Whole collection | 43,200 family CPU seconds, 57,600 wall seconds, 768 MiB evidence |
| Controller process hard CPU limit | 13,680 seconds, still charged to family use |
| Finalization reserve | 60 CPU seconds and 60 wall seconds |
| Separate report command | 120 CPU seconds and 180 wall seconds |

Before each worker, require at least 260 remaining family CPU seconds, 300 wall
seconds and 80 controller CPU seconds. Never shrink a worker's reservation to fit.
The controller hard limit covers 680 times 20 seconds plus 80 seconds of overhead;
the family budget still limits actual combined consumption. These are empirical
headrooms, not guaranteed worst-case completion bounds. Polling and signal
scheduling have no proven strict lateness bound; actual overruns invalidate the
attempt even if exit status is zero. The monitor is process-own; final wait4 usage
also includes any waited descendants. Group signaling does not prove arbitrary
descendant cleanup.

The wall alarm fires with the finalization reserve remaining and then arms a
second default-termination alarm. Each completed chunk is retained and its source,
seed, runtime, returned measurement and monitor metadata checked before continuing.
Any failure stops the activity. Keep all partial bytes, exit records and prior
checkpoints; do not resume, reroll seeds, replace failed output or pool old rows.
After failure, verify source/runtime and child termination where possible; abrupt
termination may leave completion unknown and requires external inspection.

The terminal collection receipt explicitly excludes its own final write. A full
operational verdict therefore also needs an enclosing process observation through
exit, including wait usage, wall duration, final storage inventory and cleanup.
The report command has a separate measured allowance; it is not free collection
CPU. Freeze native numerical environment details externally (CPU exposure,
native libraries and JIT/dispatch configuration), and compare them before/after.
Source and Python package hashes alone do not establish native runtime equivalence.

## Evidence custody and acceptance

Use a new evidence directory and never overwrite or delete historical attempts.
Inventory preserved bytes before release and after completion. Preserve the full
raw container, source snapshot, package/native identities, release and reviews,
commands, logs, process receipts, reports and an SHA-256 inventory. Create a separate
archive, verify all entries against that inventory and retain the originals.

A concrete no-upload handoff is an owner-downloadable archive with its checksum.
The owner names the custodian and retention period, downloads it to retained
storage, and verifies its checksum before local deletion is considered. Alternatively,
the owner can authorize a specific repository-controlled storage destination.
Until verified handoff, workspace loss remains a risk: local hashes, file permissions
and copies on the same filesystem are not durable external custody. No external
upload or storage authorization follows from this document.

Before launch, independent reviewers must accept the exact operational protocol,
source/runtime bindings, measured resource feasibility, accounting and custody plan.
After collection, independently audit all 680 unique coordinates, all primary and
required secondary arrays, all 1,022 criteria and operational completion. Only a
complete scientifically passing and operationally valid new attempt can establish
this baseline. Fixed seeds provide reproducibility, not independence from historical
runs; no historical rows count toward the new sample size. Synthetic CI and the
previous recovered accuracy-only pass do not certify the new head.

## Worker CPU timer profile (`scientific-v3`)

The subsequent `scientific-v3` profile retains all v2 scientific and resource
allowances. It adds a worker-owned CPU timer so delayed scheduling of the parent
monitor is no longer itself a terminal failure. The failed v2 attempt remains
incomplete: it stopped after 123 complete chunks when a 1.416-second parent polling
gap exceeded its frozen one-second rule. The origin of that delay is unproved.
This change does not reinterpret or resume that attempt.

After exec and before native imports, arm one POSIX timer at the absolute
`CLOCK_PROCESS_CPUTIME_ID` deadline of 170 CPU seconds, with default, unblocked
`SIGALRM`. CPU already consumed during startup counts toward that deadline.
The timer is not claimed to protect startup before it is armed; the existing
RLIMIT and parent monitor cover that interval. POSIX timers do not survive exec.
The implementation requires the reviewed Linux LP64 libc ABI and fails closed if
its timer functions are unavailable.

The initially considered `ITIMER_PROF` design was rejected before any scientific
collection: a nominal five-second timer permitted 6.5–6.7 measured CPU seconds in
this environment. In contrast, independent native controls for the POSIX process
CPU clock terminated two-second single/four-thread workloads at approximately
2.002/2.015 measured CPU seconds despite a delayed parent. These are empirical
controls, not strict latency guarantees. See the Linux
[POSIX timer manual](https://man7.org/linux/man-pages/man2/timer_create.2.html),
[absolute timer semantics](https://man7.org/linux/man-pages/man2/timer_settime.2.html)
and [signal semantics](https://man7.org/linux/man-pages/man7/signal.7.html).

Native signal masks/dispositions and the timer are checked at worker entry,
after imports/preparation immediately before measurement, after returned-output
processing, and at capture completion. Each remaining-time query is bracketed by reads of the
same process CPU clock. The fixed 170-second deadline must lie in the resulting
interval within a 0.01-second observation allowance. The analyser requires all
four records, monotone CPU observations and remaining timer, zero recurrence, and
the exact policy binding. Arming twice is rejected. This does not prove continuous
immunity to native timer changes or cover forked descendants.

Parent CPU monitoring at 170 seconds, the 180-second RLIMIT and observed-usage
rejection, the wall deadline and cleanup checks remain. Poll gaps remain measured
telemetry in v3; v2 still rejects gaps over one second. No existing record can
qualify for v3 merely by changing its profile label. All scientific criteria,
cases, seeds and sample counts remain unchanged.

A new exact-head, independently reviewed release is required for one full fresh
study. Do not pool old rows or substitute results from the failed attempt. The
native and delayed-parent controls support this mechanism; they do not guarantee
future completion or scientific accuracy.
