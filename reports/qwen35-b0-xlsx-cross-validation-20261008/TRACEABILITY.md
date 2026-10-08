# Qwen3.5 B0 run traceability audit

Source workbook: `reports/qwen35-b0-xlsx-cross-validation-20261008/35B-B0-baseline.xlsx`

Source SHA-256: `82e81a8dbe574cf7593c3d6f74eb3a835351c3279dc12e9f3581d6f91157b743`

Verdict: **static result values only; no concrete B0 run can be traced from this workbook.** Later
preparation documents provide D-grade command and time association, but no runtime receipt or
same-run cryptographic binding. The workbook still cannot prove that the values came from that
Qwen3.5-35B run.

The later evidence and separately preserved corrected workbook are recorded in
`HISTORICAL-B0-EVIDENCE-SUPPLEMENT.md`. Local inspection verifies the corrected file's bytes, ZIP
structure, and ten changed cells. It still does not provide same-run binding and therefore does not
promote either workbook to formal B0 evidence.

## Workbook contents

- One worksheet: `a1-a4-metric-dataset-matrix`.
- 21 populated dataset columns and five populated static metrics, for 105 numeric values.
- Shared strings contain dataset names, metric names, formulas, units, and applicability labels
  only.
- The package contains no embedded raw result, command log, startup receipt, environment manifest,
  external evidence link, or second configuration sheet.
- Document properties identify an editor and office build, not an experiment.

## Configuration evidence absent

The workbook does not identify any of the following:

- model repository, model variant, immutable model revision, dtype;
- tokenizer identity or fingerprint, weight manifest or model-file hashes;
- vLLM, vLLM-Ascend, container, driver, CANN, or torch-npu revisions;
- server/client command lines, resolved spec, effective configuration, or startup receipt;
- TP, PP, DP, EP, or DCP topology;
- `disable_custom_all_reduce`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, or
  `max_num_batched_tokens`;
- prefix caching, chunked prefill, speculative decoding, eager/compile mode, cudagraph mode, or
  capture sizes;
- request records, token timestamps, successful/failed counts, correctness, OOM/truncation events,
  three repetitions, logs, release evidence, or an artifact-wide checksum manifest.

Any XML substring matches for short tokens such as `tp`, `pp`, `dp`, or `ep` come from OOXML markup
or unrelated text and are not experiment fields.

## Evidence paths

- Source workbook: `reports/qwen35-b0-xlsx-cross-validation-20261008/35B-B0-baseline.xlsx`
- Full numeric/unit cross-validation:
  `reports/qwen35-b0-xlsx-cross-validation-20261008/assessment.json`
- Human-readable cross-validation: `reports/qwen35-b0-xlsx-cross-validation-20261008/README.md`

No absolute path to a qualifying Qwen3.5-35B raw run exists because no such run artifact was
supplied or embedded. Promotion to formal B0 requires the missing run bundle, not additional
inference from these 105 values.
