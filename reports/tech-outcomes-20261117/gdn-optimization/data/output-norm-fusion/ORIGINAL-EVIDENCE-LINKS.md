# Original evidence (not re-uploaded)

Per issue #664, material already archived under a fixed commit is linked, not
duplicated. All links below pin the evidence-binding PR head.

- output_norm original 4 runs and config: `https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/809d8f71abd4986012767f1e2e3b7573a7960a18/ops/gdn_output_norm_fusion/evidence.json`
  (`serving_end_to_end.runs` holds off_1/off_2/on_1/on_2 with duration,
  gen_rate, total_rate, TTFT, mean/median TPOT — enough to recompute +2.66%)
- split_fusion structured op_statistic aggregates: `https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/809d8f71abd4986012767f1e2e3b7573a7960a18/ops/gdn_split_fusion/evidence.json`
  (`l4_attribution.pure_decode` / `cohort_512` / `cohort_2048` hold the off/on
  target-group times and per-operator counts — the structured content of the
  missing CSV, though the CSV file itself was never committed)
- conv_pack and gate_fp32 originals: `https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/809d8f71abd4986012767f1e2e3b7573a7960a18/ops/gdn_conv_pack/evidence.json`,
  `https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/blob/809d8f71abd4986012767f1e2e3b7573a7960a18/ops/gate_fp32/evidence.json`
