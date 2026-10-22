# 基线身份与证据复用审计

核查日期：2026-10-22。本文为工作审计，不替代原始记录或正式验收结论。

## 复用原则

验收基线采用 vLLM 0.18 / vLLM-Ascend 0.18 系列。已有结果优先复用，兼容补丁和环境差异按实际影响评估并披露，不自动否定数据或要求重测。开发运行栈结果独立保留，不改名为
0.18。不同数据集和计量范围不相互代替。

本次复查涉及技能错误账 E01、E03、E09、E38：先恢复证据，再判断真实缺口；不能以自写计划通过代替用户要求，也不能自行增加必须使用未修改二进制的门槛。

## 0.18 证据

上游发布身份已与本机参考仓库标签核对：

- [vLLM v0.18.0](https://github.com/vllm-project/vllm/releases/tag/v0.18.0)：`bcf2be96120005e9aea171927f85055a6a5c0cf6`。
- [vLLM-Ascend v0.18.0](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0)：`e18643f8a4d5bd9990727654318ad069ea0b56e2`。

### Qwen3.5 SWE-prefix-reuse

本机已有 `gennova-qwen35-v018-native/formal/20261001-v018-native-text-only`，其 `ALL-SHA256SUMS`
于本次审计全部通过。五点均为 900 秒，摘要均为 valid、零失败。

| 并发 | 窗口输出 token |       输出 token/s |
| ---- | -------------: | -----------------: |
| C1   |          26860 | 29.844444444444445 |
| C2   |          51057 |              56.73 |
| C4   |          95660 | 106.28888888888889 |
| C8   |         145724 | 161.91555555555556 |
| C16  |         222539 | 247.26555555555555 |

表值直接取原始 `workload-result/summary.json`，吞吐为窗口输出除以 900，不新增收益结论。 五点摘要、元数据及兼容补丁的逐字节副本随本报告保存在
`baseline018-sweprefix/`；这是一组摘要证据，不声称该目录包含完整逐请求归档。 工作负载为 `swe-prefix-reuse/v1`，prepared SHA256 为
`8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`。

原始元数据记录 vLLM 应用提交 `9878e0432107e69082f8837b1f275d3aad71fd4e`、Ascend 应用提交
`0f40ff06f1c6b7e6efb8ae4850eb72bd35a47762`。补丁涉及 Qwen3.5 text-only 模型注册、权重映射、Mamba 状态接口、MTP 配置和 RoPE
选择。保留为“0.18/0.18 + Qwen3.5 text-only 兼容补丁”证据，不宣称未经修改，也不因存在补丁丢弃。

采用范围：现有 SWE-prefix-reuse 性能证据可复用；正式配对收益仍须核对比较臂的输入、配置和资源边界。这些窗口不提供 MMLU-Pro 质量或 M01
基础运算计数，不能填入对应指标。当前无依据仅因补丁重跑这五点。

原环境路径 `/tmp/gennova-v018/image-rootfs` 已不存在，`gennova-runtime/v018/wheels` 为空；这影响新运行的环境恢复，不使已有结果失效。

### 其他既有 0.18 结果

`reports/issue-232-fixed-target-20261003` 已有官方 0.18 的同规格配对，但模型为 Qwen2.5-14B FP16，不能改作 Qwen3.5-35B
BF16 的验收结果。 `reports/issue-214-v018-three-repeat-20261007` 保留九组重复套件，其既有 README
已说明原始归档覆盖限制；不因版本符合就取消这些具体限制。

## 开发运行栈中断记录

FrontierScience 运行 `m01-recovery-model-fs-paired-b0-solver-r1` 因用户纠正基线而受控中断，不是模型自发失败。原始 receipt 保留
`failed / KeyboardInterrupt`，客户端退出码 -2，服务退出码 0，`all_devices_released=true`。

本次核查原始 SHA256SUMS 全部通过。已保存 144 条 status=ok 请求记录：Olympiad 100 条、Research 44 条；分别 33 和 5 条
finish_reason=length，共 1,133,623 completion tokens。成功请求不等于语义正确，144 条记录不等于完整 160 题求解，更不等于裁判评分完成。

原始清单 SHA256：`f92da0c0842fe5429b7d57ee9e8d464c878a4523ac3de3155bc7efb58a90e51f`。 原始 receipt
SHA256：`8b1b7342b7b66bb7ef5d5e1944d619aaec6f02d22945be6e2876feee9385fc7c`。

## 尚待完成

- 追溯其他已有 0.18 数据的输入和指标，确定哪些实际填补中期缺口；不按目录名或模型版本直接套用。
- 对真正缺少的指定基线测量恢复隔离运行环境，按同工况补采。
- 将开发栈对照与指定基线的身份同步到正式报告、规范结果、网站和 issue 正文。本审计落盘不代表这些发布工作已完成。
