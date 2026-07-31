# Invalid CI publication `30554037879`

The public snapshot update from Ascend benchmark run `30554037879` was reverted because the run did
not complete the official-target admission gate. Its submission directory had no `STATUS` file and
used a retired public configuration.

The original manifest and result remain available in repository history at commit `903d5847`. They
are retained there for incident analysis and must not be restored to `submissions/` or included in
public leaderboard aggregation.

Recovery is tracked in
[vLLM-HUST/vllm-hust-benchmark#122](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/122).
