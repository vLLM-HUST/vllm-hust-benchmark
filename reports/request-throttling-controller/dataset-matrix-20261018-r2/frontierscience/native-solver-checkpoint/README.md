# FrontierScience Native solver checkpoint

The frozen full160-task Native solver stage completed in2598.665018 seconds (43min18.665s).
Olympiad100 and Research60 all returned successfully;30 and11 respectively reached the16384
output-token limit. Length-ended answers remain in the dataset and are sent to the fixed judge. This
is **completed answer generation only**: grading and the Controller pair are still pending. No task
score or optimization gain is reported here.

Both arms use a new matched single-TP2 Qwen3.5-35B-A3B BF16 Ascend910B2 runtime (Core/Ascend0.23.0,
APC off, full automatic KV, context65536, engine sequence limit16). It is not the historical
matrix0.28 profile. The fixed quality-client setting16 is distinct from the no-client-cap Agent
replay scenario; no program IDs are manufactured.

The private raw-and-plan archive was collected only after the stage succeeded, with live cluster/Pod
identity checks, while the subsequent judge stage continued. It contains the unchanged source
plan/client hashes, every request/response, run/index and solver log. Its checksum is published; the
archive itself contains raw prompts and internal service references and is retained privately.
Running verify.py with --archive checks all160 request-record hashes, the exact frozen
prompts/generation parameters, original response content/finish reasons, usage token accounting and
per-track aggregate counts. Public-only verification checks the sanitized checkpoint and file
checksums; it does not rerun inference or establish answer correctness.

No changes were made to the running model, input, judge or queue. Full quality and matched
performance evidence will be added when completed; this checkpoint must not be rendered as a final
accuracy row.
