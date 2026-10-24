# TP4 validation of four-layer prefill FX groups

The default-off compiler in [PR #10](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/pull/10)
now has real TP4 startup and finite serving evidence. It remains experimental:
mixed-decode token differences require investigation in
[Issue #11](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/11).
**This experiment establishes no additional throughput or Goodput benefit.**
The previously measured streaming preset and its approximately 3.1× scene-specific
Goodput result remain documented separately in [the TP4 report](performance-tp4-goodput.md).

The run occurred on 2026-10-12 in Asia/Shanghai (2026-10-11 16:21–16:32 UTC).
The raw-P control and grouped-P candidate used two groups of four Ascend 910B2
devices. This was a functional diagnostic with trace enabled, not a timing run.
No new TP8 experiment was conducted.

## Matched configuration and execution

Both services used runtime `f09a100ade651936b3efeaf045ec5ce63d0b38fe`,
vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`, and Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989`. The pre-existing local Ascend build-script
edit is recorded and hashed in each manifest. The model was Qwen3.5-35B-A3B BF16,
TP4/EP4, PP1/DP1, maxseq 8, 16384-token context/budget, OMP1, synchronous V1,
prefix cache off and Mamba cache mode none.

Both roles used isolated P/D communication, overlap streams, fixed four-layer
quotas, raw whole-prompt admission, and D FULL_DECODE_ONLY gears 1/2/4/8. Both
explicitly disabled `enable_npugraph_ex` and `enable_static_kernel`, and enabled
native AOT. Only `compile_prefill_groups` differed in execution policy. Ports,
CPU/card assignments, caches and trace paths were isolated. The control used
cards 0–3 and the candidate 4–7; this diagnostic does not exclude card-group or
service-order effects. The backend differs from the established streaming
benchmark, so its old results cannot serve as this compiler's matched control.

After both services passed startup checks, each ran the same 19-request numeric
plan, followed by the same 16-request natural-language suite. Roles ran
sequentially and each phase drained. The candidate then ran ten lifecycle
requests. Server admission logs contain exactly 35 control and 45 candidate
requests, with unique IDs: **79 completed responses and one planned cancellation**.
There was no automatic request retry. Both owned services stopped and all eight
devices were released.

## What passed

| Check | Observation |
| --- | --- |
| Candidate P startup | Ten actual AOT group handles returned on each of four ranks |
| D startup, both roles | Four captured graph entries per rank; candidate D backbone compiled independently after P groups |
| Generated graph text | All 40 P artifacts reference the expected four absolute GDN/attention layer prefixes; four D artifacts cover all 40 layers |
| Numeric transport | 19/19 per role; independent raw SSE/journal replay verifies prompt IDs, token counts, usage, finish, DONE and phase order |
| Natural-language suite | 16/16 text matches across roles; all finite answer checks pass |
| Candidate shape endpoints | P1, P128, P129 and P16383 complete; P16383 generates one token within the 16K context |
| Candidate cancellation | Four active D requests, 12K P cancelled with ACK at 8/40 layers, D drained, fresh P completed |
| Candidate rank receipts | All four ranks return the expected complete group sequence or acknowledged cancellation prefix |

The ten groups have distinct entry identities, cache namespaces and logged AOT
save paths. Their internal `code_hash` values are not all unique: groups 1–9 have
the same code structure. The actual layer prefixes and separate namespaces are
the relevant isolation evidence. P uses FX/AOT in NONE mode, **not P ACL replay**.

The cancellation trace covers one intermediate full-cohort cancellation. It
does not cover first/last-group races or partial-cohort cancellation. Held KV
release is supported by the pinned scheduler's ACK protocol plus successful
reuse; the allocator's held-block count was not directly observed. Group-return
logs prove successful host calls, not kernel completion or physical overlap.

Startup reported 32.86 GiB available KV memory in the control and 32.19 GiB in
the candidate (2,867,200 versus 2,809,424 KV-cache tokens). These are native
startup estimates, not an independent memory-efficiency benchmark. Compilation
and loading were concurrent across services, so their startup durations are not
a clean latency comparison.

## Numerical result and unresolved boundary

Only **9/19** same-ordinal synthetic token sequences match exactly. All five 12K
P sequences match. P128's initial and post-load repeats produce two sequences
within both roles, first differing at zero-based index 4 (`248068` / `74455`).
That instability alone does not establish an explanation for the following
mixed-D observations:

| D lane | Cross-role first difference, both rounds | Control repeat difference | Candidate repeat difference |
| --- | --- | --- | --- |
| 0 | index 3: `223` → `224` | index 74 | index 114 |
| 1 | index 65: `31838` → `10597` | none | index 104 |
| 2 | no difference | none | none |
| 3 | first round matches; second differs at index 26 | index 26 | index 80 |

At lane 0's first divergence, both prefixes are `[220, 220, 131]`. Subsequent
differences compare different autoregressive histories. Existing timestamp
alignment places all four observed lane-0 fourth-token receipts before the
later 12K request was even dispatched. Mapping the fourth token to the synchronous
one-token schedule at `computed_before=130` identifies a D1 step with the other
three initial P128 requests (384 pending P tokens), rather than D4. This mapping
uses the pinned scheduling protocol, not an independent sampler/device timestamp.
Thus the later long P is not necessary for this observed early difference;
initial P state and the earlier short-P/D execution remain possible sources.

Static review found no per-group reset of the MoE index: the partition wrapper
is a no-op with NONE and partitioning disabled, and the same P forward context
is restored across yields. Opaque MoE still resolves layers through its runtime
index; serving index values were not independently recorded. Correct graph text
and natural-language answers do not establish arbitrary-input equivalence.
No cause is assigned to BF16, fusion, communication or KV corruption.

The next bounded diagnostic should bind actual shared admission and batch shape,
repeat the four short D prompts without the 12K P, and exclude card-group effects.
If logits are needed, an observational sampler hook must preserve the native
sampling and synchronization path; this plugin intentionally rejects API
logprobs. A near-tied argmax alone would not prove compiler or state correctness.
PR #10 therefore remains draft; no clean performance run followed this diagnostic.

## Reproduction and evidence

The new [19-request probe](../scripts/probe_prefill_group_outputs.py) defaults to
writing a plan. Execution requires the plan file SHA, explicit endpoint and a
new output path. It records raw SSE bytes and chunk times and continues after
numeric differences, while stopping on transport/control failures. It is a
diagnostic, not a performance benchmark.

```bash
python scripts/probe_prefill_group_outputs.py --output plan.json
sha256sum plan.json
python scripts/probe_prefill_group_outputs.py --execute \
  --plan plan.json --plan-sha256 SHA_FROM_ABOVE \
  --base-url http://127.0.0.1:8002 --label raw --output raw.json
python scripts/probe_prefill_group_outputs.py --execute \
  --plan plan.json --plan-sha256 SHA_FROM_ABOVE \
  --base-url http://127.0.0.1:8004 --label candidate \
  --baseline-report raw.json --output candidate.json
```

[Evidence index (archive member: `evidence/2026-10-12-tp4-group-fx/README.md`)](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/812ef01ae992475d4ce6a3bb6a6ce4bda43b5f12/archive/mod-collections-20261023/muxpd/manifest.json) includes the
independent audits and the complete release archive. Reports retain captured
absolute paths; replay instructions describe how to map them without rewriting
or weakening the original SHA checks. Numeric replay does not import the
producer or its SSE parser.

At the runtime commit, the full local CPU suite passed 679 tests with two skips.
The new numeric probe passed 17 targeted tests; service ownership, lifecycle,
rank-return and independent numeric-auditor checks passed 17, 13, 14 and 9
targeted fixtures respectively. Initial tooling failures are retained in the
archive. These tests complement the finite device observations; they do not
replace the unresolved numerical or device-profiling checks.
