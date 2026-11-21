# 官方 B0 版本选择

`engine-release-history-v1.json` 的 `comparison_baselines` 提供额外官方版本；原 `baseline` 和历史配对保持原样。待发布项使用
`pending`；候选版使用 `candidate-pending`，两者不含实测值。

实际原值通过 `observations: {path, sha256, bytes}` 引用已进入发布清单的 JSON，状态为 `partial` 或
`published`。消费者同时检查发布清单、完整文件摘要、字节数及身份。

观测结构为 `vllm-hust-official-baseline-observations/v1`，包含
`baseline_id`、`model`、`precision`、`tensor_parallel_size`、`runtime.image_digest`、`hardware.chip_model`、`hardware.driver_version`
和 `profiles`。每项使用相应工况的 `id` 与
`arm`：`role: B0`、`requests`、`failed`、`source_seal_sha256`、`output_tokens`、`duration_seconds`、`output_tokens_per_second`、`mean_ttft_ms`、`mean_tpot_ms`。没有某指标时明确为
`null`；失败数保留。完整工况、权重、输入和预算的证据保留在原件来源中。

独立原测可与所选 B1 并列显示。`pairs` 当前为 `null`，不自动生成收益、成本比率或验收结论。正式配对和算力适用性须使用各自经过核验的版本化契约，不能继承原 0.18 的资格。
