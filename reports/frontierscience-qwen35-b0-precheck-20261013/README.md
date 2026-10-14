# FrontierScience Qwen3.5-35B-A3B — B0 development evaluation

Full gold split, measured on 2026-10-13 UTC (finished 2026-10-14 01:11 China time). The solver
completed 160 tasks, the judge attempted all 160, and the independent load test completed 160/160
requests. Six judge outputs remained invalid. These are development evaluation results, not official
FrontierScience leaderboard replication or V5.5 acceptance.

| Track                 | Success / total |   Rate | Valid grades | Mean valid score / 10 |
| --------------------- | --------------: | -----: | -----------: | --------------------: |
| Olympiad              |        57 / 100 | 57.00% |           97 |                   N/A |
| Research (score >= 7) |          5 / 60 |  8.33% |           57 |     2.530877192982456 |

| Diagnostic                                | Olympiad | Research |
| ----------------------------------------- | -------: | -------: |
| Solver length-limited                     |       33 |        6 |
| Judge length-limited                      |        4 |        7 |
| Length-limited judge with extracted grade |        2 |        4 |
| Invalid / unresolved judge                |        3 |        3 |
| Cases with heuristic audit flags          |        2 |        5 |
| Solver / judge transport failures         |    0 / 0 |    0 / 0 |

Research score distribution and the six track/subject aggregates are in [report.json](report.json).
[per-case-outcomes.json](per-case-outcomes.json) contains all 160 cases with numeric grades,
termination flags and SHA-256 hashes of the original solver/judge record files, without problem or
answer content.

## Protocol and interpretation

- Dataset: [openai/frontierscience](https://huggingface.co/datasets/openai/frontierscience),
  revision 25ed67db7da8f4591484e764008ff585544f5a30; 100 Olympiad questions and 60 Research tasks,
  covering physics, chemistry and biology. Source file hashes are frozen in the contract.
- Uses the official [paper Appendix B prompts](https://arxiv.org/abs/2601.21165v1), with the same
  Qwen3.5-35B-A3B service as solver and judge. This is one trial, not the official judge and
  repeated-trial protocol. API role choice and local extraction are recorded in the frozen client.
- Solver: chat completions, concurrency 1, temperature 0, seed 17, thinking off, max tokens 16384.
  Judge: same request settings, max tokens 8192. No automatic grading retries.
- Local mmlu_extract_first policy: select the first complete standalone explicit VERDICT line,
  including length-ended outputs; no loose-letter/numeric fallback. A final unterminated line at a
  truncation boundary is not accepted. This differs from requiring the official prompt's last line.
- Invalid grades stay in the full denominator as nonsuccess. They are not scientific proof that the
  solver answer is wrong. The Research numeric mean uses only 57 valid grades; missing grades are
  not assigned zero points.
- Heuristic contradictions and extraction flags do not change scores. They are nonexhaustive; human
  review is still required. No claim of domain expert validation or judge independence.
- Smoke previously failed strict-final grading checks. The full run was explicitly authorized using
  the fixed 8192 cap and extraction-first policy; it is not represented as a passed calibration.

## Separate load test

160 custom science prompts; output cap 256, request-rate infinity, temperature 0, seed 0, default
server chat template. The bench phase is independent of solver/judge correctness generation,
including the explicit thinking-off setting used for correctness. No concurrency-1 latency claim is
made for bench.

| Metric                 |                   Measured |
| ---------------------- | -------------------------: |
| Completed / failed     |                    160 / 0 |
| Duration               |        73.71363537106663 s |
| Request throughput     |    2.170561785408859 req/s |
| Output throughput      |  555.6638170646679 token/s |
| Total token throughput | 1404.5027012822786 token/s |
| Mean TTFT              |       33954.69977196772 ms |
| Mean TPOT              |      26.662729527052566 ms |

TTFT includes queue effects at request-rate infinity. No B1, improvement, repeat variance, or
official leaderboard promotion is claimed.

## Actual service and model

The full run reused the continuously running service from fs-qwen35-v02-20261013T081932Z; current
process arguments were rechecked before evaluation. No new startup or cold-start lifecycle is
claimed. The same service handled solver, judge and bench sequentially.

Model directory: /models/Qwen3.5-35B-A3B; served name: qwen3.5-35b. The model inventory records 14
weight shard filenames and sizes, but **does not hash full weights**. Eleven small
configuration/tokenizer/index files were hashed before and after the run. The declared model
revision is not full-weight verified. The container image digest was not supplied.

The saved hardware receipt identifies Ascend 910B2. The logical device selection was
ASCEND_RT_VISIBLE_DEVICES=0,1. Saved package receipts report vLLM
0.28.1.post1.dev143+gf18cf803c.empty, vLLM-Ascend
0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909, Python 3.12.13, torch 2.13.0+cpu and
torch_npu 2.13.0rc1. These are not the pinned public v0.18 baseline.

All 18 required runtime fields were verified against saved startup evidence:

| Field                                                 | Actual               |
| ----------------------------------------------------- | -------------------- |
| dtype                                                 | bfloat16             |
| TP / PP / DP / EP / DCP                               | 2 / 1 / 1 / true / 1 |
| disable_custom_all_reduce                             | true                 |
| gpu_memory_utilization                                | 0.85                 |
| max_model_len / max_num_seqs / max_num_batched_tokens | 32768 / 16 / 8192    |
| prefix / chunked / speculative                        | false / true / None  |
| enforce_eager / compilation_mode                      | false / 3            |
| cudagraph_mode                                        | FULL_DECODE_ONLY     |
| cudagraph_capture_sizes                               | [1, 2, 4, 8, 16]     |

The selected-evidence.tar.gz archive contains the complete server/startup-command.sh, actual
argument arrays, receipt/effective.json, receipt/startup.log, service origin, package/hardware
receipts, frozen plan, client source and exact client execution commands. The client itself never
starts vLLM; the external orchestration script is archived separately. Reading the archived shell
command does not execute it.

## SHA-256 binding and public scope

- Result table: f2d855f4f245ada52fa03408f8fa1f092a34ae870c95c029dc0a068cace9990b
- Evidence listing digest (463 files):
  a4148b1b8e26553e69717bb3365c666529ca1205bdbf819cf5383a0e695898b1
- Original MANIFEST file: ae99f3a1f7a453b8aa19fd7bc71d9e0c3c923efbc2ca862258f0357559c4359b
- Frozen contract: 6aeed92ee39e6661e8862581f93711ed748aea997aa43b287e3308c786e627e0

The original MANIFEST binds result table, evidence listing, profile, effective receipt and startup
log. The table references the listing digest and contract digest. The full listing binds every
original solver/judge request record and the scoring source. This is plain SHA-256, not a digital
signature.

The selected archive preserves exact bytes except explicit operator-identity redactions, recorded
per file in [publication.json](publication.json) with original and published hashes. Original sealed
files were not changed. Full raw question/answer records and container environment dumps remain
outside the public package; the listing provides their filenames, sizes and hashes. Independent
regrading requires access to those omitted records.

Verify the public package from this directory:

```bash
sha256sum -c SHA256SUMS
python3 verify_public.py
```

Rebuild it from the original sealed run in a separate publication directory:

```bash
PUBLICATION_OPERATOR=operator-name python3 export_public.py /path/to/sealed-run /path/to/publication
```

The exporter checks the original seal before producing projections; it never overwrites source
evidence. Regenerate SHA256SUMS after editing only publication scripts/documentation. Do not
regenerate or relabel the original sealed result artifacts.

This report does not alter the formal dataset matrix, baseline contracts or leaderboard. PR #277
provided evidence-packaging precedent only; none of its MMLU-Pro scores or metrics were reused.
