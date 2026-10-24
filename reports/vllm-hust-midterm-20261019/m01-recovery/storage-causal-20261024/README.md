# M01 storage r6 诊断资格测试

## 测试对象与配置

Qwen3.5-35B-A3B，BF16，TP2；同一 MMLU-Pro 200 请求工况，每请求输出 256 token。候选由 BetterScale、Prefill Microflow、KV
materialization 及位置快照、草稿因果读取范围修正组成。实际运行版本与完整参数见原始 execution-plan.json 和 launch-command.json。

## 测试方法

使用独立服务及全新编译缓存，采集两 rank 的位置、读取端点、缓存依赖与 MoE 路由。请求完成后排空服务、释放设备并封存 23 项原始文件。计时包含诊断观测成本，本轮不是正式性能测量。

## 结果

| 项目               | 结果                                             |
| ------------------ | ------------------------------------------------ |
| 完成 / 失败请求    | 200 / 0                                          |
| 输入 / 输出 token  | 282265 / 51200                                   |
| 有效准备调用       | 每 rank 876 次，两项修正覆盖全部调用             |
| 第二步草稿读取端点 | 每 rank 19354 行，差异 0                         |
| 完整位置与槽位比较 | 差异 0                                           |
| 分阶段位置比较     | 每 rank 前 128 波次，六类比较差异均为 0          |
| MoE 路由覆盖       | 每 rank 41 层；target 每层 876 次、draft 1752 次 |
| 原始文件校验       | 23 项 SHA256 通过                                |
| 退出与资源释放     | 客户端 0；服务受控退出 130；会话排空、设备释放   |

原始 receipt 的 retained_count_complete 为 false。派生核算将 132 条未更新事件分为首次输出前 13 条及输出预算完成后 119
条，未覆盖原记录。逐请求接纳序列重建的必要草稿依赖与两 rank 物理缓存记录一致。

## 结论

本轮验证了位置及草稿读取范围修正。有效 FLOPs、独立参考时间和 M01 达标结论尚未确定；不以诊断耗时替代正式计时，也不将本轮算量与另一运行耗时拼接。

## 证据

- [原始归档与摘要](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/m01-storage-r6-qualification-20261024)
- release.json：附件 URL、字节数与 SHA256。
- qualification-summary.json：原样保留的诊断摘要及派生证据哈希。
- 归档内 SHA256SUMS：23 项原始文件清单。
