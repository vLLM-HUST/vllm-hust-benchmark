# GSM8K 性能恢复与 ARC-Easy 回归

日期：2026-09-13

## 结论

- 历史截图中的 Draft `1.5x` 属于 GSM8K、B128、BF16、最大输出 512、FULL Graph
  的离线批处理口径。按同一口径在 NPU 7 冷启动复测后，Draft Adaptive 达到
  `2363.98 tok/s`，相对同卡 Target-only `1557.25 tok/s` 为 **`1.518x`**。
- 工程师模板采用 B16、FP16、最大输出 256、在线服务、`FULL_DECODE_ONLY` 和同步
  调度。该口径下串行 Draft 不能摊薄 0.5B 模型的多次 forward，不能与 B128 的
  `1.518x` 混为一组数据。
- EAGLE 在工程模板的 GSM8K 上达到 **`1.346x`**，在 ARC-Easy 上达到
  **`1.027x`**，两组均为相对同卡、同参数 Target-only 的正向提升。
- 动态 `FULL_DECODE_ONLY` 启动失败已修复：初始化时不再让不同验证宽度共享错误的
  padded descriptor，捕图时 runner 与 dispatcher 使用相同 query width。

## GSM8K B128 恢复结果

统一配置：Qwen2.5-14B-Instruct + Qwen2.5-0.5B-Instruct，200 条 prompt，B128，
BF16，最大输出 512，`max_model_len=1024`，`max_num_batched_tokens=33280`，FULL
Graph，async scheduling 和 prefix caching 开启。计时仅覆盖 `llm.generate()`，不含
模型加载、编译与捕图。

| 方案 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | 相对 baseline | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 38.5352 | 60009 | 1557.25 | 1.000x | 188/200 = 94% |
| Draft Adaptive `gamma=1..4` | 25.7118 | 60782 | 2363.98 | **1.518x** | 182/200 = 91% |

由于两组实际输出 token 数不同，耗时比为 `1.499x`；与历史报告一致，主指标采用
output token throughput，因此性能门槛结果是 `1.518x`。

## 工程模板 GSM8K

统一配置：200 条相同物化 prompt，B16，FP16，最大输出 256，TP1，
`max_num_batched_tokens=8192`，`max_model_len=32768`，`FULL_DECODE_ONLY`，同步
调度，关闭 prefix caching，开启 chunked prefill。

| 方案 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | 相对 baseline | AC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 93.1919 | 42318 | 454.10 | 1.000x | - |
| Draft Adaptive `gamma=1..4` | 245.6427 | 42337 | 172.35 | 0.380x | 81.89% |
| Draft 固定 `gamma=5` 诊断 | 184.8600 | 42276 | 228.69 | 0.504x | 66.65% |
| EAGLE Adaptive `gamma=1..4` | 69.1432 | 42265 | 611.27 | **1.346x** | 46.49% |

固定 `gamma=5` 比动态 Draft 高 `32.7%`，说明 B16 动态结果的主要损失来自在线控制器
在该负载主要选择较短宽度，以及串行 Draft 无法在小 batch 摊薄 forward；数据文件
与历史在线测试完全相同，不是 GSM8K 数据集变化。

## ARC-Easy 工程模板

ARC-Easy 只替换物化请求内容，其余服务和客户端参数与上一节相同。

| 方案 | 耗时 (s) | 输出 tokens | 吞吐 (tok/s) | 相对 baseline | AC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 73.9660 | 31892 | 431.17 | 1.000x | - |
| Draft Adaptive `gamma=1..4` | 238.1237 | 31746 | 133.32 | 0.309x | 57.88% |
| EAGLE Adaptive `gamma=1..4` | 71.6904 | 31731 | 442.61 | **1.027x** | 29.96% |

## 修复与复现约束

1. `FULL_DECODE_ONLY` 按可达 query width 初始化图键，过滤不能被当前宽度整除的
   mixed capture bucket，避免启动期断言失败。
2. 每个 capture 调用临时同步 runner 和 dispatcher 的 query width，退出后恢复，
   避免 Target graph 参数与 descriptor 宽度不一致。
3. 图模式默认设置 `VLLM_USE_AOT_COMPILE=0`，规避 PyTorch 2.10+ / Ascend 栈加载
   损坏 AOT 产物时的 `NoneType is not callable`；普通 compile cache 仍保留，用户
   显式环境值优先。
4. `vllm-hust-vspec-bench` 自动探测容器和宿主机 tokenizer 路径，避免宿主机压测在
   发请求前因写死 `/model/...` 失败。
5. 每个方法必须使用新服务进程。在线控制器状态会在同一服务中持续学习，跨数据集
   复用进程会污染后一个结果。

原始结果位于 `benchmark_results/regression_20260913/`。NPU 运行环境为
`/opt/vllm-hust-cann91`，vLLM-HUST revision
`762f85b311fbab0bcf8921dd216f5093cd58b9b8`，CANN 9.1。
