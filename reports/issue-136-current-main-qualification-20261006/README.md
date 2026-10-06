# Issue 136 current-main compatibility qualification

This directory preserves the clean one-card Ascend 910B2 service qualification used to unblock the
current-main campaign in [issue #136](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/136).
It is compatibility/readiness evidence only. It is not a benchmark repetition or a publishable
performance point.

## Frozen source and runtime

- vLLM-HUST: `c696cc916ef30c7eb57c3e90a9f278e82e6e0bc7`
- vLLM-Ascend-HUST candidate: `b400fab66eb2d1cf2e69bca32e807f187d382965`
- model: local Qwen2.5-14B-Instruct checkpoint, FP16, TP1, max model length 32768
- hardware: one Ascend 910B2 selected as runtime device 1
- vLLM distribution: `0.23.0+empty` (module version `0.23.0`)
- vLLM-Ascend distribution: `0.23.0.post1`
- torch: `2.10.0+cpu`
- torch-npu: `2.10.0.post4`
- FLA NPU: `26.9.1+deva4a7958`

The candidate backend commit is the head of
[vLLM-Ascend-HUST PR #41](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/41), not a main-branch
performance anchor. Formal matrix cells must freeze the merge commit after that PR passes CI and is
merged.

## Result

The service reached its health endpoint, completed graph compilation, allocated an 8.78 GiB KV
cache for 47,872 tokens, and returned a valid HTTP 200 completion with eight output tokens. The
response is retained in `completion.json`; this qualification checks service execution, not answer
quality. `STATUS` is `OK`, and `npu-after.txt` confirms that the selected device released its
process after shutdown.

The `EngineDeadError` near the end of `server.log` occurs after the successful response, when the
harness deliberately terminates the service process group. It is a shutdown artifact, not an
inference failure.

## Evidence map

- `run-qualification.sh`: exact v7 qualification harness
- `import-provenance.json`: observed Python modules, extension path, and distribution versions
- `core-commit.txt` and `plugin-commit.txt`: checked source revisions
- `extension.sha256` and `fla-wheel.sha256`: binary dependency identities
- `source-evidence.tar.gz`: byte-preserving archive of the original evidence directory
- `server.log`, `health.txt`, and `completion.json`: readable service evidence; text line endings
  and final newlines are repository-normalized
- `npu-before.txt` and `npu-after.txt`: device/process boundaries
- `SHA256SUMS.source-absolute`: immutable original manifest with source-container paths
- `SHA256SUMS`: portable manifest for this archived directory

The NPU snapshots also show independent issue #214 workloads on other devices. This qualification
used only runtime device 1 and makes no performance claim, so those concurrent task-owned workloads
do not enter a published comparison.
