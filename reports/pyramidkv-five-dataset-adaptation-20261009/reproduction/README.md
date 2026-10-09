# Reproduce the named subset

Use the pinned source/image/model identities and installed package versions from
[contract.json](../plan/contract.json). The registry is private: image access/export is a
prerequisite. The exact original server/environment/orchestration scripts are preserved here as
`.txt` files so formatters do not rewrite run provenance. They use the recorded host's paths and
devices; inspect and adapt paths on another host, declaring the resulting environment as a new run
contract. Never start the orchestration against a server belonging to another workload.

1. Download the three fixed URLs in [dataset-source.json](dataset-source.json), verify their SHA-256
   values, and place the two Parquet files under `assets/mmlu-pro/`. Use `pyarrow==19.0.1` and the
   contract's `transformers==5.14.1`; do not replace Torch/NPU packages.
1. Check out `TIGER-AI-Lab/MMLU-Pro` at `f418b116db00b065c2aea046518d8fcf74d39872` for the reference
   prompt/extractor. Check out PyramidKV at `5ee310696c033bd3160a9264c679dc30d7ccc1df` for the
   transport and shared analysis. The benchmark adapter's source hash must match the frozen
   contract.
1. Make `runtime-contract.json` from the exact runtime identity in `plan/contract.json` (`runtime`
   field); actual process argv/environment/config must match it. On a new host, record its real
   identity and freeze a new contract rather than pretending to reproduce this pod's identity.
1. Run preparation before any model outputs. The original command follows (set `PYTHONPATH` to the
   benchmark checkout's `src` directory):

```bash
python -m vllm_hust_benchmark.mmlu_pro_pyramidkv prepare \
  --assets /root/workspace/pyramidkv-runtime/dataset-program/assets/mmlu-pro \
  --model-path /root/workspace/Qwen3.5-35B-A3B \
  --reference /root/workspace/pyramidkv-runtime/dataset-program/mmlu-reference \
  --runtime-contract /root/workspace/pyramidkv-runtime/dataset-program/runtime-contract.json \
  --pyramidkv-repo /root/workspace/vllm-ascend-pyramidkv-hust \
  --output /root/workspace/pyramidkv-runtime/dataset-program/plan
```

5. Verify the regenerated `cases.json`, `tasks.json` and `inventory.jsonl.gz` against the hashes in
   the original contract. Source questions/prompt IDs are reconstructed from the pinned dataset and
   tokenizer, not substituted with synthetic prompts.
1. The recorded orchestration is `run-pair.py.txt`. It stops this workload's existing server, checks
   HBM release, starts B0 and then B1 with `serve.sh.txt` / `env.sh.txt` and `method-config.json`,
   checks health, performs the same startup cache policy and invokes the commands below. B1 remains
   running after completion. Startup and measurement logs are preserved separately.

```bash
python -m vllm_hust_benchmark.mmlu_pro_pyramidkv run \
  --plan /root/workspace/pyramidkv-runtime/dataset-program/plan \
  --pyramidkv-repo /root/workspace/vllm-ascend-pyramidkv-hust \
  --output /root/workspace/pyramidkv-runtime/dataset-program/paired/B0 \
  --server-log /root/workspace/pyramidkv-runtime/dataset-program/paired/B0-server.log \
  --server-pid ACTUAL_B0_PID --arm B0
# Repeat with the newly started B1 PID and B1 paths/arm.
python -m vllm_hust_benchmark.mmlu_pro_pyramidkv analyze \
  --plan /root/workspace/pyramidkv-runtime/dataset-program/plan \
  --pyramidkv-repo /root/workspace/vllm-ascend-pyramidkv-hust \
  --root /root/workspace/pyramidkv-runtime/dataset-program/paired \
  --output /root/workspace/pyramidkv-runtime/dataset-program/paired/comparison.json
```

The API endpoint is localhost:8000, served model `qwen35-pyramidkv`, using token-ID completions. No
judge model, agent scaffold, external tools or answer-driven prompt selection is involved.

The inventory and raw archives are split into ordered binary parts to satisfy this repository's
1,000 KiB per-file limit. Check `SHA256SUMS`, then concatenate the listed parts in order; verify the
reassembled SHA-256 from the corresponding parts index before decompressing. For example, from the
report directory:

```bash
sha256sum -c SHA256SUMS
cat plan/inventory.jsonl.gz.part* > /tmp/pyramidkv-inventory.jsonl.gz
sha256sum /tmp/pyramidkv-inventory.jsonl.gz
# Compare with plan/inventory-parts.json, then:
gzip -dc /tmp/pyramidkv-inventory.jsonl.gz | wc -l
```

The inventory must contain 12,032 rows. `tasks.json` contains 70 selected task identities and gold
choice letters. Raw results include every generated answer, per-task score, failure (if any), SSE,
server logs, sampled metrics/NPU receipts and startup records. The raw archives are not edited to
make model outputs, timing or failures look better.
