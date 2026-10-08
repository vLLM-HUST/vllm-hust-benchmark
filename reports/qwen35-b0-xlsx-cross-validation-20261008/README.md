# Qwen3.5-35B B0 XLSX Cross-Validation

- Source: `reports/qwen35-b0-xlsx-cross-validation-20261008/35B-B0-baseline.xlsx`
- SHA-256: `82e81a8dbe574cf7593c3d6f74eb3a835351c3279dc12e9f3581d6f91157b743`
- Source file was opened read-only and not modified.
- Populated coverage: 21 datasets × 5 static metrics = 105 values.
- Admission result: **static values only; not admissible as formal V4.9 B0 evidence**.

## Supplemental historical evidence

Later documentary evidence supplies a historical server/client command and a different corrected
workbook identity. It does not change the admission result. Both workbooks are preserved: the
original is `82e81a8d...b743`, and the corrected copy is `ddb9d8da...6506`. Local ZIP and cell-level
comparison confirms exactly ten numeric changes without overwriting either file.

See `HISTORICAL-B0-EVIDENCE-SUPPLEMENT.md` and `historical-evidence-supplement.json` for the
D/R/S/U/X evidence grading, historical command contract, exact cell deltas, current Dataset Matrix
status, cross-contract differences, anomalies, and closure checklist.

## Contract result

All five workbook units match the Dataset Matrix registry. The current Qwen3.5 TP2 registry has no
admitted raw measurement for any of these 105 cells, so there are zero same-contract numeric
confirmations. Non-Qwen3.5-35B measurements are excluded from this report's comparison surface and
cannot satisfy or stand in for the 35B contract.

## Internal checks

- 20/21 columns are within 2 tokens/request of a 256-token output pattern. BFCL-V3 is about 232.68;
  without request records this cannot distinguish variable output length, failure, or truncation.
- Total throughput is arithmetically greater than output throughput for every populated column,
  allowing an implied input throughput, but the workbook does not provide the underlying token
  counts or duration.
- Empty dataset columns (23): BBH, BFCL-V4, BURSTGPT-TRACE, GPQA, HOTPOTQA-REACT-100, INFINITEBENCH,
  LIVECODEBENCH-TEST-GENERATION, PAIO-LONGTEXT-4000, PAIO-PREFIX-SHARED-5000, PAIO-REUSE-CONV-5000,
  PAIO-SEMANTIC-SIMILAR-5000, SCHEMASTORE, SHAREGPT-V3, SONNET, STRUCTEVAL,
  SZYN-OPENCODE-SWEBENCH-VERIFIED-500, SYFI-CODING-TRACE, TAU2-BENCH, TOOLBENCH, ULTRACHAT-200K,
  VISIONARENA, WILDCHAT, WILDCHAT-4.8M.

## Missing V4.9 evidence

- success rate with planned/completed/failed request counts
- task correctness or frozen official scorer output where applicable
- OOM event count
- silent truncation count and rate
- three independent repetitions and per-metric aggregation/CV
- config_id or canonical resolved spec
- exact server and client commands
- model/runtime/container/hardware immutable identities
- request-level records and token timestamps
- raw benchmark result JSON for every repetition
- server/client/NPU logs and exit/resource-release evidence
- artifact manifest plus SHA-256 coverage

The full 105-cell workbook values, source hashes, structural checks, and contract gaps are in
`assessment.json` and `workbook-structural-comparison.json`. These values remain quarantined from
the admitted Dataset Matrix until a traceable raw run bundle is supplied.
