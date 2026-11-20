# Prospective scientific-v4 operational allocation

Scientific-v4 copies scientific-v3's cases, seeds, rows, references, tolerances
and multiplicity unchanged. Earlier profiles and evidence retain their original
limits. It is a new bounded allocation, not recovery or continuation of a failed
release.

| Limit | v3 | v4 |
| --- | ---: | ---: |
| Worker CPU hard ceiling | 180 s | 300 s |
| Absolute CPU timer / parent tripwire | 170 s | 290 s |
| Worker wall ceiling | 240 s | 360 s |
| Whole collection family CPU | 43,200 s | 72,000 s |
| Whole collection wall | 57,600 s | 86,400 s |
| Cumulative preserved evidence allowance | 4 GiB | 5 GiB |

The current-host completed chunks showed heterogeneous slowdowns relative to the
previous host: matched-case median ratios approximately 0.936, 1.543 and 1.988,
with maximum observed ratio 2.185. Applying the latter to the former largest
chunk and whole run gives planning scenarios of approximately 255 and 66,301 CPU
seconds. These are not statistical upper bounds or evidence of uniform scaling.
The new limits provide finite engineering headroom, not guaranteed completion.
Preserving the historical studies plus another full store exceeds the old 4 GiB
allowance; admission must measure retained bytes honestly, without deleting old
attempts or counting them as new samples.

The 18 s controller tripwire / 20 s per-chunk ceiling remains unchanged. At the
unchanged 0.02 s polling period, a 360 s worker permits approximately 18,000
scheduled polls. The supervisor still enforces its own CPU ceiling and reports
observed cost; scheduler delays provide no exact polling guarantee. Whole
controller CPU remains 13,680 s: 680 allocations of 20 s plus 80 s for setup and
finalization. Admission reserves 300 + 20 + 60 CPU seconds and 360 + 60 wall
seconds before each chunk. Finalization retains 60 CPU/wall seconds; reporting
has its separate 120 CPU / 180 wall allowance. The external execution envelope
must permit the new inherited limits and account through final process exit.

Only the approved 170/180/240 and 290/300/360 timer/CPU/wall combinations are
accepted for monitored timer workers. The extended timer is explicit in the
command, worker preflight, checkpoint observations and independent report checks.
A ledger-only change cannot enable it. Each new release still requires independent
source/runtime/resource/custody review and current-host enforcement controls.
Any failure stops that release; there is no automatic retry, resume or pooling.
