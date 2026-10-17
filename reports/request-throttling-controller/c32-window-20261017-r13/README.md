# Historical capped-client C32 reverse-order window experiment

Retained separate scenario; not the current offline uncapped batch. Same seed11/96 compatible programs as r12, OFF then ON,180s per side. OFF/ON engine output16.479066/32.276535 tokens/s; preemptions247/0; KV100%/75%. Independent verifier PASS15 overlapping completed rounds. Zero request/token/history errors. Experimental APC/512MiB profile, not original matrix qualification or complete-batch makespan. The current default is all96 programs arriving at time0 with no client cap; see offline-burst r15/r16 reports. No new capped-client runs are planned.

Run python verify.py here. Private token payloads, credentials and access endpoints excluded.
