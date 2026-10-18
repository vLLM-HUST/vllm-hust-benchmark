# FrontierScience Controller complete quality checkpoint

Controller completed all160 answers and grades: Olympiad58/100 (58.00%),2 unresolved; Research
score>=7:4/60 (6.67%),1 unresolved. Full denominators retained. Solver request failures0, output
truncations27/9; judge truncations6/10, including4/9 with recoverable complete verdict under the
frozen team policy. All audit flags retained. Solver2442.096s; judge930.191s.

Matched Native:60/100 and5/60, respectively. This single-trial quality result is lower by2.00
and1.67 percentage points. It is not evidence of a performance benefit or proof of the cause of
score variation. Both arms use the same Native Qwen substitute judge, Appendix B prompts, frozen
extraction policy and same new runtime. Substitute judging and one solver pass limit interpretation.
Independent performance is pending.

Profile: Qwen3.5-35B-A3B BF16 Ascend910B2 TP2; Core/Ascend0.23.0, APC off/full automatic
KV/context65536/engine sequence limit16/quality client16. This is separate from historical matrix
runtime and offline Agent replay. No program IDs or scheduling mechanism benefit are manufactured
for ordinary single-turn tasks.

Private raw-and-plan archive SHA is in checkpoint.json. It contains prompts/answers/internal
references and is not public. Independent verify.py --archive passed all160 solver and judge record
hashes, exact requests, original responses, token accounting, judge-to-solver binding, grade
reparsing with the pinned client and byte-for-byte summary/per-case recomputation. The included
receipt records that private check. Public-only mode checks sanitized proof/statistics, not private
response content or scientific correctness of the judge. No running service, scoring budget or input
was changed.
