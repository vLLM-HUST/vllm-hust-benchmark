# Qwen3.5-35B historical B0 evidence supplement

This supplement records evidence received after the original XLSX audit. It does not modify the
checked-in workbook, admit its values into the Dataset Matrix, or change any frozen benchmark
contract.

## Evidence grades

| Grade | Meaning in this report                                                       |
| ----- | ---------------------------------------------------------------------------- |
| R     | Reproduced or directly inspected in this container                           |
| D     | Documentary record or externally supplied attestation, not a runtime receipt |
| S     | Current snapshot; not proof of the historical run environment                |
| U     | User-supplied statement                                                      |
| X     | Missing evidence                                                             |

## Workbook variants

The two workbook identities are different and must remain separate.

| Variant            | Availability      |   Size | SHA-256                                                            | Grade |
| ------------------ | ----------------- | -----: | ------------------------------------------------------------------ | ----- |
| Container original | Preserved locally | 13,343 | `82e81a8dbe574cf7593c3d6f74eb3a835351c3279dc12e9f3581d6f91157b743` | R     |
| Engineer corrected | Preserved locally | 12,193 | `ddb9d8dab05dbccb2adb094b0598c812f2c2e69976128337ce23343fb8c16506` | R     |

The container original has one visible worksheet `a1-a4-metric-dataset-matrix`, no formulas, and no
defined names. Its 21 populated dataset columns contain 105 values across request throughput, output
throughput, total throughput, mean TTFT, and mean TPOT. All 105 values match `assessment.json`; the
five BBH cells are empty. These checks are R grade for the local workbook bytes only. They do not
bind the workbook to a historical run.

The corrected workbook is now present. Both ZIP archives pass CRC/integrity checks, contain the same
17 members, expose the same single visible worksheet, use dimension `A1:AW30`, contain no formulas
or defined names, and have identical cell-coordinate sets. The only cell-value differences are the
ten rows below. The corrected copy still contains 105 populated metric values and an empty BBH
column.

The engineer manifest is also preserved locally at 14,203 bytes with SHA-256
`e1de9a4d70e7d4800781df6d18619d1fb411bad421f7a00e6cd70d5fd3ba933c`. Its bytes are R grade; its
documentary configuration and historical-run claims retain their own D/S/U/X grades.

| Dataset         | Cell | Container original | Declared corrected |
| --------------- | ---- | -----------------: | -----------------: |
| JSONSCHEMABENCH | T2   |               1.81 |               1.80 |
| JSONSCHEMABENCH | T4   |             462.85 |             461.04 |
| JSONSCHEMABENCH | T5   |            4488.37 |            4470.80 |
| JSONSCHEMABENCH | T6   |           44070.49 |           44233.91 |
| JSONSCHEMABENCH | T7   |              30.16 |              30.23 |
| LONGBENCH       | X2   |               1.23 |               1.21 |
| LONGBENCH       | X4   |             313.96 |             308.72 |
| LONGBENCH       | X5   |            8049.71 |            7915.46 |
| LONGBENCH       | X6   |           80210.74 |           81211.22 |
| LONGBENCH       | X7   |              45.71 |              46.31 |

No local file was overwritten. The ten file-level differences are R-grade observations; their
attribution to historical result JSON remains D because those JSON files and a same-run immutable
binding are not present in this bundle.

## Historical documented contract

Twenty-two preparation documents record the following server and client configuration. This is
D-grade evidence: the result JSON files do not contain the server parameters, and no same-run
manifest cryptographically binds the commands to the results.

| Area                 | Historical documented value                                                                 | Grade          |
| -------------------- | ------------------------------------------------------------------------------------------- | -------------- |
| Model                | `/models/Qwen3.5-35B-A3B`; immutable revision not recorded                                  | D / X revision |
| Precision            | BF16 compute, `kv-cache-dtype=auto`                                                         | D              |
| Hardware/topology    | devices 0,1; TP2 / PP1 / DP1; expert parallel enabled                                       | D              |
| Capacity             | max model length 32768; max sequences 16; max batched tokens 8192; block size 128; GMU 0.85 | D              |
| Cache/prefill        | prefix cache disabled; chunked prefill enabled                                              | D              |
| Decode/graph         | no speculative setting; non-eager; compile mode 3; `FULL_DECODE_ONLY`; captures 1/2/4/8/16  | D              |
| Scheduling/execution | FCFS; seed 0; multiprocessing; custom all-reduce disabled                                   | D              |
| Client               | OpenAI chat; custom dataset; output length 256; request rate `inf`; temperature 0; seed 0   | D              |
| Request count        | 200 prompts, except HumanEval 164                                                           | D              |

Recorded server command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve /models/Qwen3.5-35B-A3B --served-model-name qwen3.5-35b \
  --host 127.0.0.1 --port 18180 --dtype bfloat16 --kv-cache-dtype auto --block-size 128 \
  --tensor-parallel-size 2 --enable-expert-parallel --pipeline-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-enforce-eager \
  --seed 0 --scheduling-policy fcfs --distributed-executor-backend mp --disable-custom-all-reduce \
  --no-trust-remote-code --load-format auto --no-enable-log-requests --uvicorn-log-level info \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-capture-sizes 1 2 4 8 16
```

Recorded client contract:

```bash
vllm bench serve --backend openai-chat --dataset-name custom \
  --custom-output-len 256 --num-prompts 200 --request-rate inf \
  --temperature 0 --seed 0
```

HumanEval replaces `--num-prompts 200` with 164. DCP and speculative decoding are unset.

The historical load is an unbounded-arrival throughput test. Queueing-heavy TTFT from that run must
not be compared with a fixed-rate, closed-loop, SLO-qualified, or fixed-window campaign as if the
load contracts matched.

## Cross-contract comparison

The current Qwen3.5 Dataset Matrix registry freezes only the model revision, BF16, TP2, and 2x
Ascend 910B2 at scenario level. It has 258 `not_tested` cells, 49 `not_applicable` cells, and one
admitted `baseline_only` cell: SZYN agent resolve rate 46.4%. That agent result is not a
serving-throughput B0 and cannot confirm any of the 105 workbook values.

| Field              | Historical B0                                        | Current Dataset Matrix                      | Unified 35B Frontier        | Classification                 |
| ------------------ | ---------------------------------------------------- | ------------------------------------------- | --------------------------- | ------------------------------ |
| Model revision     | not bound                                            | `712cf743...d1f6d45`                        | `712cf743...d1f6d45`        | historical X                   |
| Hardware           | 2x 910B2 only from current snapshot/time association | 2x 910B2                                    | 2x 910B2                    | S is not historical R          |
| TP / PP / DP       | 2 / 1 / 1                                            | TP2 only; PP/DP unstated                    | 2 / 1 / 1                   | partial documentary match      |
| Expert parallel    | enabled                                              | unstated                                    | disabled                    | different from Frontier        |
| Context            | 32768                                                | unstated                                    | 262144                      | different from Frontier        |
| Prefix caching     | disabled                                             | unstated                                    | enabled                     | different from Frontier        |
| Max batched tokens | 8192                                                 | unstated                                    | 4096                        | different from Frontier        |
| Graph              | FULL_DECODE_ONLY                                     | unstated                                    | FULL_AND_PIECEWISE          | different from Frontier        |
| Speculative decode | unset                                                | unstated                                    | native MTP2                 | different from Frontier        |
| Arrival            | request rate `inf`                                   | per-cell contract required but not admitted | closed-loop C1/C2/C4/C8/C16 | different load                 |
| Request/window     | 200 prompts; HumanEval 164                           | no serving B0 admitted                      | strict 900-second windows   | different measurement boundary |
| Output budget      | fixed 256                                            | no serving B0 admitted                      | source-derived per turn     | different workload             |

The contract differences explain why the workbook is useful as qualification and arithmetic
cross-validation but cannot replace formal repetitions or enter Frontier gain calculations.

## Known anomalies

- JSONSCHEMABENCH reportedly completed 199/200 requests and LONGBENCH-V2 198/200. The corrected
  values are not locally reproducible until their raw JSON arrives.
- HumanEval reportedly completed 164/164 but was stored below a directory named
  `aime-2024-baseline-vllm-hust`.
- InstructCoder reportedly used a separate path under `/root/dataset/instructcoder/`.
- BBH has no result JSON and all five workbook cells are empty.
- The 2026-10-08 runtime and model inspection is S grade. It cannot prove the environment used on
  2026-10-04/05.

## Current snapshot, not historical provenance

The following facts were inspected on 2026-10-08 and are S grade only:

| Area              | Current snapshot                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Accelerator       | Ascend 910B2, 64 GB HBM per visible chip                                                                                  |
| Software          | CANN 9.1.0; Python 3.12.13                                                                                                |
| vLLM              | `0.28.1.post1.dev143+gf18cf803c.empty`; source path is not a Git repository, so the commit is not independently traceable |
| vLLM-Ascend       | `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`; release-tag marker `v0.27.1`                                     |
| Extension/runtime | vllm-hust-ext 0.2.0.dev0; torch 2.13.0+cpu; torch-npu 2.13.0rc1; transformers 5.14.1                                      |
| Model             | `/models/Qwen3.5-35B-A3B`; Qwen3_5MoeForConditionalGeneration; 14 safetensors shards                                      |
| Environment       | Ascend/ATB paths point to CANN 9.1.0; ATB CXX ABI 1; task queue enabled; OMP threads 1                                    |

The environment name and the statement that it was unchanged during the historical run are U grade.
The image RepoDigest and 14 weight-shard hashes are X. Small model metadata files have snapshot
hashes in the external manifest, but those abbreviated hashes are not copied here and do not
substitute for the weight manifest.

## Required closure evidence

Formal promotion still requires all of the following in one immutable evidence bundle:

1. Runtime-effective configuration and complete startup logs.
1. Container image RepoDigest.
1. Traceable vLLM core and vLLM-Ascend/plugin commits.
1. SHA-256 for all 14 model weight shards.
1. Same-run binding among server command, client command, result files, logs, and environment
   manifest.
1. Planned/completed/failed counts, request records, token timestamps, correctness, OOM/truncation
   evidence, exits, and resource release.
1. The frozen formal repetition count and aggregation outputs.
1. The raw result JSON and materialized inputs listed by the manifest for independent local
   revalidation of all 105 corrected values.

Until those items arrive, the historical values remain qualification and cross-validation evidence
only.
