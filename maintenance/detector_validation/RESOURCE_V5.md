# Prospective scientific-v5 preservation allowance

`scientific-v5` inherits the complete `scientific-v4` design and execution policy,
with a cumulative preserved-evidence allowance of **6 GiB** instead of 5 GiB.
All cases, seeds, sample counts, references, statistical criteria, worker timers,
CPU/wall limits, store capacity and report limits remain unchanged. Historical
profiles and releases retain their original definitions and outcomes.

The measured enclosing retention obligation was 4,773,911,025 bytes. Adding the
805,306,368-byte full store and 16,777,216-byte reserve requires **5,595,994,609
bytes**, exceeding 5 GiB by 227,285,489 bytes. The 6 GiB allowance leaves
846,456,335 bytes of headroom at that snapshot. Re-inventory before release:
additional artifacts and audit records consume that headroom. This allocation is
not a guarantee of free disk space or durable archival storage.

Use `fresh init --profile scientific-v5` only with a new experiment directory,
honest cumulative accounting and the existing independent review/freeze process
described in [FRESH.md](FRESH.md). Both accounting scopes enforce the same
preservation check, including the full store allocation and reserve. A new
directory or budget label does not remove retained evidence from the obligation.
Do not edit a historical manifest, relabel an old study or delete prior evidence
to fit this allowance. This profile does not itself approve or launch collection.

The local supervisor and analyser retain the v4 monitoring, metadata and runtime
checks. The separate hosted study policy is unchanged. This prerequisite permits
future work to retain its application-only diff after integration; it does not
establish that a future candidate passes the scientific study.
