# Operational execution history and cold-start requirements

Operational success means the declared public runtime pipeline returned and retained
its required rows and passed integrity checks. It never means detector probability
accuracy. Scientific acceptance is a separate full-profile assessment. No default,
API, supported domain or simulator implementation is changed by these diagnostics.

## Preserved first portable operational attempt

The v7 experiment `15308cbc-6f1c-41f1-b6a1-e7102219f696` used release SHA-256
`2086b92bc97fbc09a95f237a204ba3af534f628ffcb876d733052f7c155234d3`.
Its fixed profile requested four vacuum PNR and four asymmetric coherent threshold
rows. The whole collection target was60CPU/120wall seconds. Initialization,
freeze, run and report together observed65.010141CPU/65.128840wall seconds;
read-only review/hash costs outside those windows were labelled separately.

Collection itself observed56.144116CPU/56.187299wall seconds and was INCOMPLETE.
The PNR worker returned four rows, using38.99229 waited CPU seconds. Allocation
logic then granted the threshold worker only11CPU seconds. That worker recorded
12.228354 waited CPU seconds, exited with status-9, and emitted only its initial
frame: no raw rows or final metadata. Both the truncated stream and observed
worker-CPU overrun were retained. The signal sender and threshold failure stage
are unknown; missing output is not a detector failure. There was no reroll or
replacement of this attempt. Accuracy remains UNTESTED.

The evidence demonstrates inadequate remaining allocation for establishing both
paths, not a measured decomposition into import, JIT and sampler time. The old
receipts did not capture enough phase markers to attribute that time. Cold imports
and compilation are plausible contributors, not established causes of the exit.

## One prospectively reviewed follow-up

The v8 diagnostic keeps the same eight requested rows, inputs and seeds. It uses
one new experiment and exact independent review/freeze, explicitly referring to
the preserved first experiment without pooling rows or resetting historical costs.
The target is300CPU/600wall seconds overall,120CPU/240wall per worker,14GiB address
space per process, one numerical thread and8MiB collection evidence. Worker
admission must reserve the full allocation before launch; an undersized remaining
allowance causes an incomplete stop instead of silently launching another worker
with too little budget. Original full scientific settings and criteria are unchanged.

The separate enclosing phase allowance accounts setup, controls and independent
analysis. Known prior command usage is retained; unknown older totals remain
explicitly unknown. Budget declarations require external authorization and review;
changing an experiment UUID does not replenish an enclosing budget. Exactly one
follow-up collection was authorized. That authorization has been consumed; it does
not authorize another collection.

Workers emit bounded diagnostic phase markers for entry, preflight and capture
entry/return, with process CPU and monotonic timing. Capture can include imports,
JIT, state preparation and measurement; these markers do not isolate JIT cost or
establish internal sampling progress. There is no prewarm, JIT-disable change or
seed-selection loop. Requested limits and observed overshoots remain distinct;
watchdogs are practical controls, not strict OS quota promises.

Run the public `init`, external three-role review, `freeze`, `run`, and `report`
commands in FRESH.md. Use a new directory, preserve every exit and partial result,
and do not resume or reroll failed sampling automatically. A full scientific
profile requires its own reviewed release and cannot inherit this operational
approval. All68 original scientific cases remain untested by the operational
profile, regardless of its outcome. Historical scientific FAIL/INCOMPLETE evidence
remains unchanged.

The follow-up returned all eight requested rows and its independent report recorded
operational PASS. Collection observed 85.177143014 CPU seconds and 85.169019443
wall seconds. All 68 scientific accuracy cases remain UNTESTED by this operational
profile. The earlier partial attempt remains preserved; its rows were not pooled
with this attempt.
