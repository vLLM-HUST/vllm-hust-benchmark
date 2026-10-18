# FrontierScience Native quality checkpoint

Native completed all160 tasks and grading. Olympiad:60/100 (60.00%),2 unresolved; Research
score>=7:5/60 (8.33%),1 unresolved. Unresolved grades remain in the full denominator. Solver
transport failures:0. Solver output truncations:30 Olympiad and11 Research, retained and graded.
Judge output truncations:5 and3; frozen first-complete VERDICT policy recovered3 and2 of those,
respectively. Audit flags and all negative cases remain in the summary.

Solver2598.665s; judge1010.146s; report0.817s. Single full quality trial, same Qwen substitute judge
and team Appendix B/scoring client; this is not official leaderboard replication. Controller quality
and independent performance are pending, so no optimization gain or paired comparison is claimed.

Both arms use the new matched Qwen3.5-35B-A3B BF16 Ascend910B2 TP2 profile: Core/Ascend0.23.0, APC
off, full automatic KV, context65536, engine sequence limit16, fixed quality client16. This differs
from the historical matrix runtime and from the offline Agent replay profile.

The private archive contains the frozen source plan, solver/judge records, exact request/response
bodies and report. It is not public because it contains prompts, answers and internal addresses. Its
SHA256 is recorded in checkpoint.json. verify.py --archive validates all record hashes,
source/dataset hashes, exact solver and judge requests, raw responses and token accounting,
solver-to-judge binding, reparses all grades with the pinned client and recomputes both summary.json
and per_case.csv byte-for-byte. It makes no service calls. The private archive was actually
verified; private-verification.json is that receipt. Public-only mode checks sanitized statistics
and hashes; it cannot inspect private evidence or establish the judge's scientific correctness.

This completed-stage checkpoint was taken while the unchanged Controller stage continued. The whole
dataset queue remains running; no running model, workload or scoring policy was altered. Website
comparison waits for the matched Controller result and corresponding performance evidence.
