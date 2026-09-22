# Compare official datasets on a new deployment

Enter when bringing a new model, chip count or serving optimization onto the leaderboard. Dataset
identity, deployment configuration, fixed-target admission and measurement kind are different
contracts; resolve which must be equal before reusing old results or running hardware.

For BetterScale TP2, Fletcher explicitly chose on 2026-09-22: use official workload datasets,
graph-enabled serving on both sides, actual model/hardware and performance-oriented configuration.
Other settings may differ if disclosed. This does not authorize calling custom SWE traces ShareGPT,
faking an old fixed contract, or trimming data to hide a development-time context admission bound.

## Avoid the paid discovery traps

- `official_scenarios.json` gives generic dataset defaults; the versioned specs in
  `docs/official-baselines/` give fixed-target client/server parameters. Preserve both the inherited
  spec identity and explicit measurement overrides.
- `same_spec.resolve_server_parameters` injects `enforce_eager` when the key is absent. A graph
  deployment must explicitly set false and verify rendered argv; merely removing the key does not
  select graph execution through that helper.
- Client resolution translates random lengths and splits prefix repetition's total4096 into3840
  shared +256 varying tokens with10 prefixes. Reuse that translation instead of guessing CLI flags.
  Host/port overrides belong only to online clients, not offline latency/throughput commands.
- Current strict same-spec hash includes graph/worker settings. Differently optimized deployments
  may be dataset-matched but not hash-equal. Never copy a hash from one arm onto another or conceal
  effective arm settings to obtain a compare card.
- Historical public filtering is not a generic hardware/model ranking policy: website aggregator
  checks official workload names on910B against old baseline versions; registry classification also
  restricts the old public-model set. Adding a model string to an export alone is not publication
  admission.
- The website schema's accountable model band was restricted to7B-13B in the inspected September22
  snapshot. It is not evidence that a27B model belongs to that band. Expand truthful classification,
  never relabel the model.

## Dataset preparation and metric fidelity

Use upstream dataset samplers and pinned source. A local lossless InstructCoder Parquet directory
needs `hf_name=likaixin/InstructCoder` so the upstream dispatcher selects the right prompt
formatter. Record resolved dataset revision and split; train108,391 at
revision6a778a720284d6520b56bd03d5c3070930d41071 is not the total train+test count quoted in the
dataset description.

Count rendered chat tokens by tokenizing the rendered string. With the observed Transformers5.14.1
interface, `len(apply_chat_template(...))` can count a mapping's keys rather than tokens. The
Qwen27/EvoScientist32-record check gave a real maximum input+output of8237; one row exceeds8192. No
truncation is justified by that fact.

Upstream ShareGPT already filters prompt/total lengths under its documented sampling policy.
Distinguish that unchanged producer behavior from new filtering introduced to accommodate a
candidate. Do not substitute online HTTP throughput for a scenario declared offline
throughput/latency. Null unmeasured metrics are preferable to invented zeros, especially TTFT on
nonstreaming/offline paths.

Timing records must retain all repeats, errors, actual output counts, cache policy, client
concurrency/rate and per-arm effective argv. Existing step/profile speedups do not fill HTTP
throughput fields. Dataset-identical measurements are not automatically old-target-equivalent or
model-quality evaluations.

## Version-specific defaults caught before formal export

The pinned vLLM0.25.1 CustomDataset CLI defaults `custom_output_len=256`, overriding each JSONL
row's `output_tokens`. Use `--custom-output-len -1` to honor the32 original EvoScientist lengths
(78,633 outputs total; maximum8,099), and audit actual output-length arrays rather than trusting
dataset file identity. For its `openai-chat` backend use `--skip-chat-template`: the server applies
the chat template, so the client should send the original prompt instead of a previously rendered
template inside another user message.

The same runtime hides `/reset_prefix_cache` unless `VLLM_SERVER_DEV_MODE=1`. Enable it for both
loopback-only benchmark servers; do not patch the donor or silently retain prior-workload prefix
cache after a404.

The website now separates entry normalization/public admission into
`scripts/leaderboard_entry_policy.py`. Explicit old official target IDs retain old checks; a TP2
deployment does not become a single-card fixed target merely by using910B and an official dataset.
The standard producer still owns exports. New entries marked
`metadata.measurement_scope=dataset-matched` with an unregistered, non-official spec ID retain
`verified=false` and are labeled `outside-fixed-target`, not invented historical backfills. This
status is a scope declaration, not validation of performance or dataset equality.

Upstream metric translation now lives in `benchmark_metrics.py`: offline `avg_latency` becomes
`batch_latency_ms`, never TTFT; request/s cannot fill `throughput_tps`; missing peak memory and
error observations remain null. Existing published artifacts are not retroactively rewritten. The
current catalog aliases `tpot_ms` to `tbt_ms`; this field is a per-request average TPOT, not the raw
ITL distribution. Keep that definition explicit in new comparison methods.

The v0.25.1 offline throughput JSON reports total input+output tokens/s, while its stdout separately
reports actual output-token count. If presenting output tokens/s consistently with HTTP results,
derive it from that count and measured elapsed seconds, retain both raw files, and disclose the
basis. Never substitute requests/s. Use existing
`aggregate_results.aggregate_entries(method="mean", outlier_handling="none")` for two-repeat
producer aggregates and retain both raw entries; a latest/best front-end selection is not a repeat
mean.
