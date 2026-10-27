# Research upgrade: shared causal measurement plane

## Primary opportunity

This repository directly owns B1 and C2. It also supplies measurement inputs
for A1 serving-roofline calibration and C3 statistical gates, but it must not
claim either full contribution until their models and validation exist.

## Seven-step boundary

1. **Problem:** request metrics cannot distinguish queue, prefill, decode,
   communication, KV, transfer, graph, or client causes.
2. **Importance:** optimizing a correlated symptom wastes accelerator time and
   can worsen the true bottleneck.
3. **Gap:** current dominant-span attribution is deterministic localization,
   not causal diagnosis.
4. **Idea:** expose a stable low-overhead event schema and validate attribution
   through controlled stage interventions and counterfactual replay.
5. **Feasibility:** the runtime hooks and matched hook-on/off carrier already
   exist; extend them only for one fault class at a time.
6. **Evaluation:** blinded known/false regressions, trace-off overhead,
   controlled interventions, attribution precision/recall, localization time,
   and request-level effects.
7. **Takeaway gate:** causal serving diagnosis requires intervention-linked
   cross-layer state, not the longest observed span alone.

## Shared-interface contract

- Export versioned phase spans, resource counters, graph identity, workload
  arrival semantics, and correlation IDs.
- Keep project-specific policy decisions outside this repository.
- Feed A1/C3 models through stable artifacts; do not absorb their papers merely
  because they consume profiler events.

The immediate gate remains controlled live attribution. More trace fields alone
do not constitute progress.
