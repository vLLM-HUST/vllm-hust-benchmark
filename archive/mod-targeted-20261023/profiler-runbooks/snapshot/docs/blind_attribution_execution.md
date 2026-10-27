# Blind-attribution local execution

This slice implements only the public, diagnosis-side work that can be
completed before NPU calibration and opaque-case access. It does not construct
or inspect an opaque case and it does not make an M0 accuracy claim.

The closed case schema is
`docs/blind_attribution_case_bundle.schema.json`. Runtime validation additionally
rejects answer-bearing keys, semantically revealing case IDs, duplicate IDs,
unmatched configuration/clock models, non-alternating pairs, invalid intervals,
unsupported candidates, and unknown fields. Pair metadata makes workload,
request order, token counts, concurrency, and batch shape explicit. The shared
bundle is deterministically rendered as the five
comparison views named in `docs/blind_attribution_contract.md`.
The raw-event arm contains the closed Request Lifecycle event rows, while the
span arm contains derived intervals; the two representations are intentionally
different and neither contains the DAG edges used by the full arm.

`traceloom_adapter.py` consumes TraceLoom's augmented SQLite relations. It
stores request/lifecycle join keys and resolves only the relations applicable
to each link: tree-node occurrence and anchor/event identity, raw-row lineage,
runtime/device relations, synchronization actions, and optional graph rows.
Device-backed links additionally verify the exact occurrence interval and
overlap-safe TraceLoom cost bound. Empty, missing, inconsistent, or ambiguous
membership is never replaced by a timestamp-overlap guess. TraceLoom continues
to own its execution tree and profiler rows; the adapter does not serialize a
second tree or raw-profiler-row model.

The full-DAG scorer uses matched candidate deltas, the declared noise floor,
the 0.60 dominance and 0.40 residual rules, paired bootstrap confidence with
seed `2026082701`, and fail-closed abstention for incomplete identity, event
loss, clock ambiguity, unmatched configuration, unsupported links, or a
top-candidate interval crossing zero. The output remains `NOT_M0_PROVEN`.
Candidate timing is eligible for a decision only when its lifecycle span has a
request-scoped causal/ownership edge or a supported TraceLoom link. The result
records the owners, edge kinds, span count, and TraceLoom-link count used for
the ranking, so a longest stage or complete trace is not silently promoted to
a causal conclusion.

Run the public development fixture with a new output directory:

```bash
python .benchmarks/run_blind_attribution_development.py \
  --output .benchmarks/results/issue1_blind_attribution_development/local-01
```

The command writes all five views and one timed result using exclusive-create
semantics. Reusing an output path fails instead of overwriting a pre-reveal
result. These development files test the pipeline only and must not be counted
as opaque accuracy, NPU calibration, or counterfactual evidence.

The deferred work is unchanged: five matched no-intervention NPU calibration
pairs, custody-provided opaque cases, pre-reveal method outputs, reveal,
rank-one counterfactuals, the final score table, and any justified claim update.

The exact public calibration target, workload order, scorer settings, and role
isolation boundary are recorded in `blind_attribution_preblind_target.json`.
The calibration uses the maximum total positive measurable-candidate duration
delta from five independent no-intervention pairs. The
`insufficient_evidence_or_multi_cause` abstention bucket remains visible in the
pair report but is excluded from that total, matching the scorer's aggregation
domain. The calibration contains no fault or case oracle.

Run the fixed calibration with a local, non-repository output directory:

```bash
python .benchmarks/run_preblind_noise_calibration.py \
  --device "$NPU_DEVICE" \
  --port "$SERVICE_PORT" \
  --output-dir "$LOCAL_OUTPUT_DIR"
```

The completed public calibration summary is
`.benchmarks/results/issue1_preblind_noise_calibration/summary.json`. Raw logs,
process identifiers, device indices, service ports, and host paths remain in
local custody and are not part of the public result.
