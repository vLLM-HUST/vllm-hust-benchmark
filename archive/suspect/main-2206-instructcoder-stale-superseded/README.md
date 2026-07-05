# Superseded main#2206 instructcoder-online submission

`submissions/backfill-instructcoder-online-20260604-2206f1f7` was removed from the active public leaderboard inputs because it produced a stale low `instructcoder-online` point for `main#2206`:

- workload: `instructcoder-online`
- commit: `2206f1f7b7212801187bc001c5f6cb86b2289214`
- old submitted_at: `2026-06-04T01:49:37Z`
- old throughput: `125.3294783026658 tok/s`
- old runtime parameters included `gpu_memory_utilization=0.6`, port `8010`, and a Hugging Face cache model path.

It is superseded by `submissions/historical-pr-main-2206-bf298-instructcoder-clean-gapfill-2206f1f7b7-bf2984e34a`, rerun on NPU0 through the isolated backfill dev-hub path with graph mode, `gpu_memory_utilization=0.90`, `max_model_len=32768`, `max_num_seqs=16`, no enforce-eager, no prefix caching, and no chunked prefill. The rerun completed 2048/2048 requests with 0 failures and `168.18954390116278 tok/s`, which matches the neighboring single-card `instructcoder-online` trend.
