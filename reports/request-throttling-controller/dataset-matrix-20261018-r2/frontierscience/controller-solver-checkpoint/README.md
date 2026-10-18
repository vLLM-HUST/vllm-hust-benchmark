# FrontierScience Controller solver checkpoint

The Controller arm completed all160 tasks in2442.096132 seconds (40min42.096s), with160 successful
responses and0 request failures. Olympiad100:73 stop,27 length; Research60:51 stop,9 length. Actual
output usage794954/376110 tokens. Truncated answers remain and are graded by the same Native judge.
Controller grading is still pending; no quality score or paired optimization claim is made.

Same frozen new-runtime matched profile as the Native checkpoint: Qwen3.5-35B-A3B BF16 Ascend910B2
TP2, Core/Ascend0.23.0, APC off/full automatic KV/context65536/engine sequence limit16/fixed quality
client16. Ordinary single-turn requests do not create program IDs; differences are not evidence of
request throttling benefit. Solver wall times are full answer-generation times with different actual
output lengths, not the separate repeated short-output performance benchmark.

The private archive was collected after the solver stage succeeded while judging continued, with
cluster/Pod UID checks. Its hash is recorded in checkpoint.json. It contains raw prompts/answers and
internal references and is not public. verify.py --archive actually passed all160 frozen
request/response/hash/finish-reason/usage checks; receipt included. Public-only verification checks
aggregate/hash consistency, not private raw bytes or task correctness. See the sibling Native
checkpoints for the matching baseline. No running workload, model or queue configuration was
changed.
