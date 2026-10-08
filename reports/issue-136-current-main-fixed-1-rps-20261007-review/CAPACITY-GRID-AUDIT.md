# Issue #136 Capacity Grid Audit

Date: 2026-10-07 UTC

Scope: CPU-only review of the 12 capacity-pilot grids in
`run-issue136-all-capacity-pilots.sh` and their compatibility with
`build-issue136-scaled-plan.py`. No NPU, lock, port, service, or active
benchmark worktree was touched.

## Verdict

The original extension had adequate TP-specific upper sentinels, but its
intermediate spacing was too coarse to support a bounded near-capacity
selection. In particular, adjacency alone allowed boundaries such as 4 to 6
RPS or 8 to 12 RPS. The grids and validator were tightened before execution.

The revised campaign is ready to run as an experimental capacity-selection
stage. It does not by itself establish a statistically precise capacity
estimate; it selects the formal scaled-load rate under the frozen p99-TTFT
rule, and the subsequent three-repetition formal matrix remains required.

## Grid checks

- All 12 workload/TP combinations are present exactly once.
- Every rate list is strictly ascending, unique, finite in its shell literal
  form, and positive.
- TP1, TP2, and TP4 end at 8, 12, and 16 RPS respectively for every workload.
- The random, ShareGPT, and prefix workloads have 11/13/15 points at
  TP1/TP2/TP4. Agent-research has 13/15/17 points because it retains the
  0.25 and 0.75 RPS low-rate probes.
- Every adjacent interval is at most `max(0.5 RPS, 25% of the lower rate)`.

The 8/12/16 endpoints are reasonable first-run rejected-sentinel candidates:
the known random TP1 4-RPS observation was still eligible, and the upper
bounds grow with TP. They are not assumed to be rejected. If a final endpoint
is still eligible, the existing decision has no higher rejected rate and the
scaled-plan builder fails closed; the affected grid must then be extended.

## Validator and contract compatibility

`build-issue136-scaled-plan.py` still emits the same materializer and formal
rate-matrix schemas. It now additionally rejects a selected/rejected boundary
wider than `max(0.5 RPS, 25% of selected rate)`. This closes the ambiguity
where two widely separated rates could be adjacent in a sparse input list.

The existing requirements remain intact: the selected point must be the last
eligible member of a contiguous prefix, the immediately following point must
be rejected, raw rows and checksums must reproduce the decision, and all 12
workload/TP records must be present. Float rates remain compatible with the
materializer and formal runner contracts.

## Verification

- `bash -n` passed for both shell runners; `py_compile` passed for the
  scaled-plan builder and its test module.
- `shellcheck` passed for both shell runners.
- `pytest -q test_build_issue136_scaled_plan.py`: 8 passed.
- A standalone parser asserted 12 unique workload/TP rows, strictly ascending
  positive unique rates, exact 8/12/16 endpoints, and the bracket-spacing
  invariant for every adjacent pair.
- `sha256sum -c issue136-next-stage-sources.sha256` passed after updating the
  three modified source digests.

## Residual limitations

- Each capacity point uses 40 prompts, so p99 TTFT is a pilot selection signal,
  not a publication-grade latency estimate.
- A fail-closed result at 8/12/16 requires extending only the affected grid;
  it must not be interpreted as a selected capacity.
- A bounded bracket limits grid-resolution error but does not eliminate run
  variance or non-monotonic latency. Formal three-repetition evidence is still
  required before publication.
