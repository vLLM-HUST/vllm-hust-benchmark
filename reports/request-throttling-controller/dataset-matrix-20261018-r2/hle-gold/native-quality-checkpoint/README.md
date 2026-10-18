# HLE Gold complete Native quality checkpoint

Native quality: 164/668 = 24.550898%; 25 unresolved grades remain in the full denominator. All 668 text/image tasks were submitted. Solver request failures: 0; judge transport failures: 0. 372 solver responses and 25 judge responses reached the frozen output limit; all retained. Strict JSON policy is unchanged, and no truncated judge rescue is applied. Text145/575; image19/93.

Solver13356.851448s, judge2000.331800s. Controller quality and independent performance are pending; this is not an optimization comparison or official leaderboard replication. Both arms use the same Native Qwen substitute judge under the matching profile.

Private archive SHA256: 268772cdf3e7b2b4c09ebac066531fb4022763746dd8ac608803b0d00f465a32. Independent verification passed all 668 solver/judge bindings, input/source hashes, exact text/image request bodies, raw responses/usage, pinned-client grade reparse, and byte-for-byte summary/per-case reconstruction. Raw prompts, answers, images and internal endpoints remain private.

`python verify.py` checks public proof consistency; `--archive` additionally checks the private archive.
