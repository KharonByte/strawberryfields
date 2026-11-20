# Bounded CPU-timer observations

The absolute process CPU deadline, signal, scientific cases and statistical
criteria are unchanged. Each timer checkpoint now takes at most three readings
of the same timer. It re-reads only when every check except the 10 ms observation
bracket passes. It never rearms the timer, accesses the RNG, redraws samples or
retries a detector chunk. All other failures stop immediately; three wide readings
also stop. Every rejected reading is retained, including when a later read fails.

Current-source successful records require an explicit observation policy, count
and rejection list (empty when the first reading passes). Reporting independently
checks the retained operands, width-only eligibility and the complete clock and
remaining-time trajectory. Removing history is not a legacy-format fallback.
Historical releases retain their frozen analyser and unchanged evidence.

A controlled observer delay reproduced rejection of a healthy, unchanged timer
solely due to bracket width. Bounded re-observation addresses that demonstrated
condition. It does not establish which predicate caused the historical failed
study: that receipt lacked individual predicate values. The failed study remains
incomplete. Future failures retain selected timer operands and named predicates,
not ambient local variables.

A new study requires a separate reviewed source/runtime release, unchanged fixed
scientific design and all original operational ceilings. Earlier rows cannot fill
its coverage. Reusing fixed seeds is reproducibility, not independent replication;
no conditional-on-eventual-completion confidence guarantee is implied.
