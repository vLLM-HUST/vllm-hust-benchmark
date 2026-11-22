# 官方 B0 版本选择

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

官方 0.23 的完整固定配置报告保留原 C8 profile，并增加固定五项和 `real-context-*`。`quality` 保留各 track
的计划、已评分、成功和未解析数；不能拼成完整准确率。`native_timing` 只记录原三次 T，不放入五项吞吐图。`automatic_ft_audit` 的适用性拒绝保留反例，F
与效率仍为空，不据此推断运行时正确性。归档为报告元数据闭包，非完整原始流。

官方 0.30 的独立原值包括固定五项和四个 `real-context-*`，保留完整分组评分及未解析人口。原三次 T 与后补 200 请求 companion
分别绑定来源；后者自动核验拒绝算量合同，F/效率为空。原整轮报告保持 failed；C8 合成负载本次仅保留原计数和时钟，不用输出预算代替实测 token，也不自动组成 B0/B1 正式配对。

0.28 源码候选的独立观测保留固定五项、四组真实上下文、质量分组、四租户和原三次 T。封存的 200 请求 companion 由公开入口进行 JSON 预检，实际反例为
call5/row0、1383→1385；F 与效率为空，完整 SDK 准入未通过。候选结果不自动组成新版 B1 配对，原失败和未解析人口不改写。
