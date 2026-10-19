# Request Throttling Controller evidence index

This index separates current deliverables from historical/interrupted measurements. Public capsules
contain deidentified metrics, configuration/source hashes and independent verifiers. Raw prompts,
answers, images, token payloads and internal endpoints remain private.

| Evidence | Status / result | Limits | |
\---------------------------------------------------------------------------------------------------------
|
\---------------------------------------------------------------------------------------------------------------
| \------------------------------------------------------------------------------------------------ |
| [Offline three-pair summary](offline-burst-three-seed-20261017/) | Paired output-rate
median+91.686%, verified; [source protocol](offline-burst-protocol-20261017/) | 96 programs/time0/no
client cap; APC-on/512MiB;180s windows, not batch makespan or task accuracy | |
[FrontierScience complete](dataset-matrix-20261018-r2/frontierscience/complete/) | Native/Controller
Olympiad60/100 vs58/100; Research5/60 vs4/60; throughput474.1268 vs432.1431 tokens/s (−8.85%) |
Quality/performance decrease retained; fails99% quality-retention budget | |
[HLE Gold complete pair](dataset-matrix-20261018-r2/hle-gold/complete/) | Native164/668 versus
Controller159/668;25 unresolved each; full quality and text/image performance verified |
Retained96.95%, below99% quality budget; no cost qualification | |
[MMLU-Pro Native quality](dataset-matrix-20261018-r3/mmlu-pro/native-quality-checkpoint/) |
9632/12032=80.05%;0 transport failures,62 invalid answers,1520 length-limited; private verification
PASS | Full pair now complete; assembled stream records only, original SSE not retained | |
[HLE Revision Native quality](dataset-matrix-20261018-r3/hle-revision/native-quality-checkpoint/) |
303/1143=26.51%;39 unresolved retained;0 transport failures; private verification PASS | Controller
and independent performance pending; substitute Qwen judge | |
[HLE Revision Controller quality](dataset-matrix-20261018-r3/hle-revision/controller-quality-checkpoint/)
| 308/1143=26.95%;32 unresolved;0 transport failures; private verification PASS | Matched
Native303/1143, retention101.650165%; independent performance pending | |
[HLE Revision complete pair](dataset-matrix-20261018-r3/hle-revision/complete/) |
Native303/1143→Controller308/1143; retention101.650165%; quality/performance independent
verification PASS | Text395.72→368.90 tok/s (negative); image285.59→294.30; website pending | |
[Latest Gold website publication](website-hle-gold-20261018/) | PR444 deployed and verified in the
existing Gold row; same matrix and standard details | PR443 single-table structure retained,
historical values preserved |

| [MMLU-Pro complete pair](dataset-matrix-20261018-r3/mmlu-pro/complete/) | Native9632/12032 versus
Controller9599/12032; independent median output337.200931/360.525152 tokens/s (+6.917010%); all
formal200/200, zero failures | Retained99.657392%; single quality pass, ordinary forwarding, not
full-solution cost or scheduling causality | | [MMLU website publication](website-mmlu-20261018/) |
PR445 exact merge deployed; original MMLU row and seven-column details verified in live browser |
Historical Native9609/12032 preserved; matching9632/12032 in dated details |

Gold and MMLU-Pro are delivered. HLE Revision quality/performance continues on its independent
matching TP2 service. Do not treat snapshots as final task scores. Each Controller arm is compared
with its own same-service Native; no cross-host baseline splicing. Full budgets and1warmup+3formal
per arm are retained. Current protocol and running status:
[implementation log](https://github.com/vLLM-HUST/request-throttling-controller/blob/codex/offline-batch-experiment-contract/docs/experiments/DATASET_MATRIX_EXTENSION_20261018.md),
[team TODO11](https://github.com/vLLM-HUST/inference-engine-project#todo%E4%BB%BB%E5%8A%A1%E8%AE%A4%E9%A2%86%E6%B8%85%E5%8D%95).

[Code PR3](https://github.com/vLLM-HUST/request-throttling-controller/pull/3) and
[data PR283](https://github.com/vLLM-HUST/vllm-hust-benchmark/pull/283) are draft review deliveries.
Old C32, interrupted runs, negative short trials and earlier website receipts are retained as
historical evidence, not rewritten into the current protocol. Each checkpoint README states whether
its verifier checks public consistency only or also private raw evidence via `--archive`.

## SWE-bench Pro V1 ARM actual Agent attempt, 2026-10-19

[Single-task environment and real-Agent attempt](swe-bench-pro-v1-arm-20261019/) supersedes the old
V2 preflight as the current deployment route. Official V1 Agent and ready community ARM image
executed30 real Qwen calls, changed8 tracked files, exhausted the frozen budget and obtained
official false in a fresh grader container. Environment workflow executed; repair unsuccessful; no
MOD performance comparison. Private raw evidence independently verified and retained. New frozen
HUST runtime inputs were separately prepared and all4695 source/support files verified; CPU-only
bootstrap fails on device-architecture discovery and is not claimed qualified. Existing runs were
not migrated.

[Unified runtime NPU-visible bootstrap follow-up](unified-runtime-preparation-20261019-r2/): same
frozen HUST inputs and original official check pass in the owned NPU-visible worker,4695
source/support files independently verified. The earlier zero-NPU failure remains in the preceding
capsule. Existing Qwen service remains healthy; no migration, dependency substitution, new model
serving or performance claim.
