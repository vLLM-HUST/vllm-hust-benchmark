# v4 addendum: #23 acceptance result under the redefined standard (2026-11-07)

REPORT-v1.md to REPORT-v3.md are unchanged. Every QUALITY-V2 FAIL recorded there (default raw/group
endpoint 4; opt-in ROW endpoint 4) stays a FAIL under QUALITY-V2. Issue references are to
`vLLM-HUST/vllm-ascend-MuxPD-hust`.

**Verdict under the redefined #23 acceptance: TP2 relies on the existing, accepted consistency
result; TP4 completed token-level verification in this round. Both meet the new standard.**

## Scope change recorded during collection

The plan `data/QUALITY-ROW-SOLO.md` (frozen and hashed before any run) had a TP4 part A and a TP2
token-level part B. After ns4-1 finished and while ns4-2 was running, before any result was compared
with ROW, the maintainer changed the scope (`data/QUALITY-ROW-SOLO-SCOPE.md`, hashed):

- Part B (TP2 runs ns2-1, rw2-1) was cancelled and never ran. The plan file and driver were not
  edited; the run directories were pre-created so the controller exited before any device use.
- TP2 is accepted on the existing evidence: ROW (`qf-row`, MuxPD `43fcaf4`) equals native
  single-request output (`native-solo200`) in **text and finish reason** on 200/200 cases
  (`data/qf-row-check.txt`). That evidence used `max_tokens` 2048 and kept no token arrays, so it is
  **not** a token-level verification and is not reported as one.

## TP4 token-level result (part A)

MuxPD `43fcaf4` (clean), Core `d0f22d2`, Ascend `c182a9e`, devices 0-3, TP4+EP,
`HCCL_DETERMINISTIC=true`, frozen-v2 plan (200 cases, temperature 0, seed 42, 8192-token cap).

- Native single-request: ns4-1, ns4-2. Engine args `data/native-quality-fusion-engine-args.json`
  (the same file as the v1 native runs); driver `data/run_quality_solo.py`, one request in flight
  (asserted every engine step; observed maximum 1). Both completed on first launch, exit 0, within
  the frozen bounds, 0 errors.
- ROW: sv2-w1, sv2-w2 from v2, reused unchanged (same code, environment, devices and plan;
  `data/raw-row-fusion-engine-args.json`).

| Pair             | Token arrays | Text    | Finish reason |
| ---------------- | ------------ | ------- | ------------- |
| sv2-w1 vs ns4-1  | 200/200      | 200/200 | 200/200       |
| sv2-w1 vs ns4-2  | 200/200      | 200/200 | 200/200       |
| sv2-w2 vs ns4-1  | 200/200      | 200/200 | 200/200       |
| sv2-w2 vs ns4-2  | 200/200      | 200/200 | 200/200       |
| ns4-1 vs ns4-2   | 200/200      | 200/200 | 200/200       |
| sv2-w1 vs sv2-w2 | 200/200      | 200/200 | 200/200       |

Every ROW x native single-request pair matches on complete token arrays: **PASS**. The repeated
native runs agree with each other, so no run was selected. Comparison output:
`data/row-solo-tp4-compare.txt` (`data/row_solo_compare.py`); per-case token-array digests:
`data/ns4-*.outcomes.json` and `data/sv2-w*.outcomes.json`.

Known configuration difference: native uses chunked prefill (required by the frozen driver) and
MuxPD does not; every prompt fits in one prefill step, and the outputs are identical.

## Interpretation and limits

- The five QUALITY-V2 ROW "losses" (v2) were differences from **batched** native; ROW reproduces
  native single-request output exactly at TP4, so they are batch-dependent numerics of the batched
  control, not ROW defects.
- Default mode (ROW off) is not required to match batched native per case or per token, and was not
  re-evaluated here. ROW stays default-off. No ROW performance was measured.
- Scope: one model (Qwen3.5-35B-A3B BF16), this 200-case set, TP2 and TP4 only.

## Checksums

`SHA256SUMS-v4` verifies the current tree. As explained for v1 in REPORT-v3.md, earlier manifests
report files changed by later uploads as FAILED: this upload rewrote `README.md`, so it now fails
against `SHA256SUMS-v2` and `SHA256SUMS-v3` as well. Data files of earlier versions are unchanged.
