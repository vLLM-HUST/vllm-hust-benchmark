# Request Throttling Controller evidence index

This index separates current deliverables from historical/interrupted measurements. Public capsules
contain deidentified metrics, configuration/source hashes and independent verifiers. Raw prompts,
answers, images, token payloads and internal endpoints remain private.

| Evidence                                                                                          | Status / result                                                                                                 | Limits                                                                                                   |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Offline three-pair summary](offline-burst-three-seed-20261017/)                                  | Paired output-rate median+91.686%, verified; [source protocol](offline-burst-protocol-20261017/)                | 96 programs/time0/no client cap; APC-on/512MiB;180s windows, not batch makespan or task accuracy         |
| [FrontierScience complete](dataset-matrix-20261018-r2/frontierscience/complete/)                  | Native/Controller Olympiad60/100 vs58/100; Research5/60 vs4/60; throughput474.1268 vs432.1431 tokens/s (−8.85%) | Quality/performance decrease retained; fails99% quality-retention budget                                 |
| [HLE Gold complete pair](dataset-matrix-20261018-r2/hle-gold/complete/)                           | Native164/668 versus Controller159/668;25 unresolved each; full quality and text/image performance verified     | Retained96.95%, below99% quality budget; no cost qualification                                           |
| [MMLU-Pro Native quality](dataset-matrix-20261018-r3/mmlu-pro/native-quality-checkpoint/)         | 9632/12032=80.05%;0 transport failures,62 invalid answers,1520 length-limited; private verification PASS        | Controller and independent performance pending; assembled stream records only, original SSE not retained |
| [HLE Revision Native quality](dataset-matrix-20261018-r3/hle-revision/native-quality-checkpoint/) | 303/1143=26.51%;39 unresolved retained;0 transport failures; private verification PASS                          | Controller and independent performance pending; substitute Qwen judge                                    |
| [Latest Gold website publication](website-hle-gold-20261018/)                                     | PR444 deployed and verified in the existing Gold row; same matrix and standard details                          | PR443 single-table structure retained, historical values preserved                                       |

Gold is delivered. HLE Revision and MMLU-Pro quality/performance continue on independent matching
TP2 services. Do not treat snapshots as final task scores. Each Controller arm is compared with its
own same-service Native; no cross-host baseline splicing. Full budgets and1warmup+3formal per arm
are retained. Current protocol and running status:
[implementation log](https://github.com/vLLM-HUST/request-throttling-controller/blob/codex/offline-batch-experiment-contract/docs/experiments/DATASET_MATRIX_EXTENSION_20261018.md),
[team TODO11](https://github.com/vLLM-HUST/inference-engine-project#todo%E4%BB%BB%E5%8A%A1%E8%AE%A4%E9%A2%86%E6%B8%85%E5%8D%95).

[Code PR3](https://github.com/vLLM-HUST/request-throttling-controller/pull/3) and
[data PR283](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/283) are draft review deliveries.
Old C32, interrupted runs, negative short trials and earlier website receipts are retained as
historical evidence, not rewritten into the current protocol. Each checkpoint README states whether
its verifier checks public consistency only or also private raw evidence via `--archive`.
