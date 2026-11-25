# 官方 B0 版本选择

## 四项实测怎么读

在[官网 B0/B1 对比页](https://vllm-hust.sage.org.ai/b0-b1.html)选择 B0 版本，查看「B0 四项实测原值」。课题四 V2 正式验收使用固定的原始 0.18。四个版本的独立数据见[指标报告](data/b0-four-indicators-20261125-v1/report.json)和[线上 JSON](https://vllm-hust.sage.org.ai/data/b0-four-indicators-20261125-v1/report.json)。报告中的 `baselines` 按 B0 标识分组，每项 `source` 提供源文件路径、SHA256 和字节数。

| 指标               | 工况与读法                                                                                                                                                                                                        | JSON 字段                                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 4.1 参考效率       | Qwen3.5-35B-A3B BF16、910B2 双卡 TP2，固定 MMLU-Pro 200 题，原生接口 out256，合计三轮。以每卡有效算力下界除以冻结参考算力上界，乘以 100 得到效率下界。                                                            | `F_interval`、`T_seconds`、`tflops_per_card_interval`、`reference_tflops_per_card_upper`、`efficiency_lower_percent` |
| 4.2 固定工况性能   | 依次读取 MMLU-Pro、HLE、FrontierScience、SWE-bench Pro 和 Terminal-Bench。SWE 和 Terminal 使用公开轨迹回放。每项列出完成数、失败数、成功输出吞吐及平均 TTFT、TPOT，时延单位为 ms。                                | `profiles`                                                                                                           |
| 4.3 上下文与四租户 | 单租户阶段为 4K、16K、32K，均为 C8。四租户阶段为 4K、总并发 24。官网明细从源文件读取 TTFT 和 TPOT 的 P95/P99，以及 32K 的五个 60 秒提交窗口。                                                                     | `phases`、`ttft_p95_16k_over_4k`、`tpot_p95_16k_over_4k`                                                             |
| 4.4 算力采样与成本 | 按 2026-11-17 公开租金 1,875 元/卡月折算。每百万成功输出 token 成本等于「每卡月租 × 占用卡数 × 测量窗口秒数 × 100 万」，除以「30 天的秒数 × 该窗口成功输出 token 数」。AI Core 以各窗口的逐卡采样及双卡均值为准。 | `estimated_cny_per_million_successful_output_tokens`、`c24_device_utilization`、`rental_projection`                  |

4.1 的 `identity` 标明实际运行身份。0.18 使用原始运行时，0.23、0.28 源码候选和 0.30 使用各自的 MTP2 KV 前缀补丁复测。4.2–4.4 使用对应版本的封存原测。0.23 和 0.30 的算力复测见[补丁组报告](data/b0-mtp-kv-prefix-repair-v1/report.json)。0.28 见[算力回执](data/b0-four-indicators-20261125-v1/0.28-prefix-repair-4.1.json)和 [CANN 9.1 内核与分块证据](data/b0-four-indicators-20261125-v1/0.28-cann-kernel-binding.json)。

[租金与 AI Core 原值](data/b0-independent-rental-observations-v1.json)保存 0.23、0.28 和 0.30 的 MMLU 与 C24 两个测量窗口。官网 0.18 的 MMLU 成本与 AI Core 明细使用 [R10 验收的同窗 B0 原件](data/r10-35b-paired-full-20261125/acceptance.json)，见 `targets["4.4"].formal_cost.mmlu`。独立 JSON 中 0.18 的历史 MMLU 租金估计使用[固定 B0 观测](data/fixed-b0-original018-v1.json)的窗口。

`engine-release-history-v1.json` 的 `comparison_baselines` 提供额外官方版本；原 `baseline` 和历史配对保持原样。待发布项使用
`pending`；候选版使用 `candidate-pending`，两者不含实测值。

实际原值通过 `observations: {path, sha256, bytes}` 引用已进入发布清单的 JSON，状态为 `partial` 或
`published`。消费者同时检查发布清单、完整文件摘要、字节数及身份。

观测结构为 `vllm-hust-official-baseline-observations/v1`，包含
`baseline_id`、`model`、`precision`、`tensor_parallel_size`、`runtime`、`hardware.chip_model`、`hardware.driver_version`
和 `profiles`。每项使用相应工况的 `id` 与
`arm`：`role: B0`、`requests`、`failed`、`source_seal_sha256`、`output_tokens`、`duration_seconds`、`output_tokens_per_second`、`mean_ttft_ms`、`mean_tpot_ms`。没有某指标时明确为
`null`；失败数保留。完整工况、权重、输入和预算的证据保留在原件来源中。

官方镜像的 `runtime.kind` 为 `official-image`，`image_digest` 必须与目录的固定摘要相同。源码候选的 `runtime.kind` 为
`source-build-candidate`，按目录 `source_identity` 精确匹配
`base_image_digest`、`vllm_source_commit`、`ascend_source_commit`，并提供
`build_receipt_sha256`；基底镜像摘要不是候选引擎的镜像身份。0.28 使用独立的 `candidate-0.28.0rc1-source`
标识，发布结果后仍保留“源码候选”名称。

独立原测可与所选 B1 并列显示。`pairs` 当前为 `null`，不自动生成收益、成本比率或验收结论。正式配对和算力适用性须使用各自经过核验的版本化契约，不能继承原 0.18 的资格。

原封存未记录驱动时，`hardware.driver_version` 为 `null`，`driver_version_status` 为
`not-recorded`；`npu_smi_version` 单独保留工具版本，不代替驱动版本。

## 原版运行记录与适用性

官方 0.23 的完整固定配置报告保留原 C8 profile，并增加固定五项和 `real-context-*`。`quality` 保留各 track
的计划、已评分、成功和未解析数；不能拼成完整准确率。`native_timing` 只记录原三次 T，不放入五项吞吐图。`automatic_ft_audit` 的适用性拒绝保留反例，F
与效率仍为空，不据此推断运行时正确性。归档为报告元数据闭包，非完整原始流。

官方 0.30 的独立原值包括固定五项和四个 `real-context-*`，保留完整分组评分及未解析人口。原三次 T 与后补 200 请求 companion
分别绑定来源；后者自动核验拒绝算量合同，F/效率为空。原整轮报告保持 failed；C8 合成负载本次仅保留原计数和时钟，不用输出预算代替实测 token，也不自动组成 B0/B1 正式配对。

0.28 源码候选的独立观测保留固定五项、四组真实上下文、质量分组、四租户和原三次 T。封存的 200 请求 companion 由公开入口进行 JSON 预检，实际反例为
call5/row0、1383→1385；F 与效率为空，完整 SDK 准入未通过。候选结果不自动组成新版 B1 配对，原失败和未解析人口不改写。
