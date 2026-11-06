# Independent code review of b276a4e and 87c277c (2026-11-06)

Reviewer: separate agent with no access to the author's conclusions, read-only; CPU suite 956 passed.
Verdict: no blocker. The per-row kernel arguments match native full_graph_fia for a single request (TND,
sparse_mode 3, same mask, pre/next tokens SWA_INT_MAX, actual_seq_lengths [1], LSE size 1); graph bookkeeping
(params/handles/events) stays aligned across layers and sizes; skipped padding tasks still record their event;
draft-model paths stay native; the default 'tnd' path installs nothing.

| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | major | FULL_DECODE_ONLY pads with [1..B] rows whose seq_lens are 0, not a dummy TND request; row treated them as real rows (KV-length-0 M=1 calls, host update every replay). Real rows were unaffected. | Fixed in 9b0641f, regression tests added |
| 2 | minor | Block-table rebinding can point at a per-step temporary when the dummy padding request overflows the table (FULL mode, full batch). Same pattern as native. | Open, documented; not reachable in the shipped FULL_DECODE_ONLY preset |
| 3 | minor | KV-sharing layers: replay looks up `layer_name`, native uses `kv_sharing_target_layer_name or layer_name`. Qwen3.5 unaffected. | Open, out of validated scope |
| 4 | minor | One shared row workspace sized from the first captured layer; models with mixed attention shapes could be undersized. | Open, out of validated scope |
| 5 | minor | Rejected requests (87c277c) not reported in finished_requests for data-parallel clients. | Fixed in 9b0641f, regression test added |
| 6 | nit | Other add_request ValueErrors (multimodal, too-long prompt) still kill EngineCore; structured-output grammar compiled then discarded. | Open (pre-existing) |
| 7 | nit | A non-default layout silently never engages if the size uses paged attention or head_size 512 (fails closed; visible only in STATS). | Open, documented |

Test adequacy: row tests cover plumbing well; expected arguments are hand-written, not derived from native code;
no multi-layer interleaving; bit-identity is device-only. BSH eager/capture/update paths have no tests (BSH is a
diagnostic layout, not recommended).

Device check of the fix (tools/row_recheck.sh, quality/rowfix-q1, MuxPD 9b0641f, shipped row preset, TP2, c16):
text == native-solo200 on 200/200 and == row-q1 on 200/200; GATE-rule 154 correct, 0 regressions. Not a gate run.
