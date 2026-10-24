# G6 Counterfactual Spec — minimal rank-one template (2026-08-11)

> Gate: **G6 — counterfactual** (experiment_plan.md step 7). This is a
> **development-level** counterfactual template and run; it is **NOT
> blind-scored** and must not be relabeled as M0 blind-accuracy evidence
> (blind scoring requires the separately frozen Team-A custody/reveal
> protocol).

## 1. Rank-1 mechanism (from causal ranking evidence)

The only mechanism with a reproducible, observable phase delta in the
G4/G5 real-online runs is the **KV-recovery state machine, specifically the
H2D restore path** (`preempt -> restore_start -> restore_done ->
scheduler_wakeup -> admission -> first_prefill_or_decode`). Under the 8 GiB
pressure workload, the request that is preempted and restored pays a recovery
tail while the other requests finish faster (CPU-tier offload reduces their
wait).

## 2. Matched pair (only the rank-1 mechanism variable changes)

| | base | head |
| --- | --- | --- |
| mode | `tiering_enabled` | `tiering_disabled` |
| connector/spec | OffloadingConnector + `TieringOffloadingSpec` | OffloadingConnector + `CPUOffloadingSpec` |
| recovery/restore path | present (H2D restore + tier promotion) | absent (CPU offload only, no restore) |
| D2H offload | present | present (kept) |
| device KV | 8 GiB (`--kv-cache-memory-bytes 8589934592`) | 8 GiB |
| model/hardware | Qwen2.5-14B-Instruct, NPU6, FP16, 32768, util 0.6 | same |
| workload/seed | 4×15000→8192, seed 20260809 | same |
| profiler | on (scope active only for TieringOffloadingSpec) | on (scope inactive for CPUOffloadingSpec) |

Only the recovery/restore mechanism differs; D2H offload, capacity, model,
hardware, workload, and seed are identical.

## 3. Predicted phase delta

- base: the recovered request's end-to-end time has a **recovery tail**;
  empirically ~40 s (default cudagraph) over the non-recovered median.
- head: no restore path, so **no tail** — all requests uniform.

## 4. Pass / fail rule

Define per-lifecycle `tail_s = total_s[recovered] - median(total_s[other])`
(in base the recovered request is identifiable as the slowest; in head, the
max-minus-median over the 4 requests). Aggregate over ≥3 lifecycles:

- **PASS** if `median(tail_s)_base > 10 s` (tail present) **and**
  `median(tail_s)_head < 5 s` (tail absent), with base/head lifecycles ≥3.
- **FAIL** otherwise (mechanism not causal for the delta, or pair not matched).

## 5. Reuse / template

- Server scripts: `g4_server.sh {enabled|disabled}` (8 GiB, default cudagraph).
- Client: `g4_pressure_client.py` (4×15000→8192, seed 20260809).
- Artifacts per run: `server.log`, `pressure_client.out`, trace shards.
- To reuse: change the single mechanism variable only (mode), keep capacity,
  model, hardware, workload, seed fixed; run ≥3 lifecycles each side.
