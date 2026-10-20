### 摘要

补充 #290 中 MMLU-Pro KV materialization/recompute 性能现象的后续调查。

核心结论：

- 历史筛选确实记录了明显吞吐提升，但执行的是**逐请求独立缓存盐／重算路径**，不是在复用、搬运和重算之间进行有效动态选择。
- 后续小规模复现曾出现类似的大幅差异；该次复现的缓存文件和源码证据确认，**无盐臂正式请求窗口内发生了首次 Triton specialization 编译**。
- 补齐启动预热并禁止服务期 Triton JIT miss 后，同一对照中的大幅盐隔离收益没有复现。
- 这不能证明原 #290 实验的全部差异都由编译造成，也不能证明 arrival 策略在其他负载下无效。

本次仅整理已有记录，未新增实验，不修改历史测量值。

### 1. 原始现象

原始 MMLU-Pro 筛选使用 200 个请求、无限到达率、每请求输出 256 token：

| 配置 | 无 KV 模块 | 加 KV 模块 | 同执行宽度吞吐变化 |
|---|---:|---:|---:|
| E16/R36 | 529.808 token/s | 695.251 token/s | +31.23% |
| E24/R36 | 561.509 token/s | 823.880 token/s | +46.73% |

BetterScale 基座为 `2c07ddd5eb04f2afeb77123ce066a3a70662a94c`。

KV 模块使用受限 school 0.28 适配器。原请求没有提供复用 anchor，全部 200 次策略决策均为 `recompute`，原因为 `no_reusable_prefix`；实际动作是注入请求独立的 `cache_salt`。部分复用未执行。

因此，这些数字不能直接解释为动态选择 KV 搬运／部分复用带来的收益。

### 2. 源码调查：盐改变了什么

在该 BetterScale 基座中，复用要求匹配完整的 GDN/MTP checkpoint，而不仅是共享一段文本前缀。

独立盐改变了缓存身份匹配。例如，原先同为 `None` 的盐会继续执行 token 切片和前缀比较；独立盐可提前拒绝不匹配的 checkpoint。因此，即使没有成功命中，盐仍可能减少失败查找的 CPU 工作。

但：

- 没有发现盐直接选择更快设备内核的分支；
- 独立盐不等于关闭所有 State 缓存或备份；
- 单凭服务日志打印的 `0.0%` 命中率，不能严格证明全部缓存路径没有命中；
- 源码存在工作量差异，不代表它足以解释几十个百分点的吞吐差异。

### 3. 首次小规模复现：确认存在正式窗口内 JIT 干扰

后续使用原输入的前 48 个请求进行诊断，两臂各有：

- 输入 68,844 token；
- 输出 12,288 token；
- 48/48 请求成功。

无盐臂先运行，独立盐臂后运行；分别启动服务，但共享 Triton 磁盘缓存。

| 指标 | 无盐 | 独立盐 |
|---|---:|---:|
| 请求窗口耗时 | 29.553 s | 14.094 s |
| 输出吞吐 | 415.80 token/s | 871.87 token/s |

**这不是有效的盐收益估计。**

源码及缓存产物定位到 `token_progress.correct`：虽然请求数参数不直接 specialization，但 `BLOCK=next_power_of_2(n)` 是编译期常量。单请求预检查没有覆盖多请求形状。

无盐正式请求窗口约为 **14:35:34.670–14:36:04.218 UTC**，期间新增：

| BLOCK | 缓存元数据时间，UTC |
|---|---|
| 4 | 14:35:38.851 |
| 8 | 14:35:43.179 |
| 16 | 14:35:47.111 |
| 32 | 14:35:51.691 |

后运行的盐臂请求窗口没有新增上述 specialization。

时间映射使用事后采样的 wall-clock/monotonic offset，存在精度边界；缓存时间不是编译耗时的直接计量。不过，结合源码与缓存 IR，可以确认这次复现受到**正式窗口内首次 JIT**污染。

两臂均未观察到实际前缀命中；聚合 scheduler CPU 时间只相差约 91 ms，不能直接解释约 15.46 s 的窗口差异。

### 4. 修复与受保护复测

复测保留原 BetterScale `2c07ddd5` 基座，仅叠加 `a7a43ed3fed34870ddd640a4c9f2960b0cfb7914` 中四个文件：

- `worker.py`
- `compilation_scope.py`
- `models/qwen35/__init__.py`
- `models/qwen35/serving_warmup.py`

改动在启动阶段准备相关 specialization，并在服务前封住 Triton JIT scope。服务期间出现进程内 cache miss，包括需要从磁盘加载的未预热 specialization，会直接失败，而不是静默污染计时。

该约束针对 **Triton JIT**，不是覆盖所有 Torch/Dynamo/CANN 编译路径的全局保证。

两臂使用相同 observer、TP2/MTP2/FULL、E24/R36、每卡 20 GiB State 与 8 GiB host 缓存。为避开已知 MTP AOT-load 故障，两臂对称禁用 Torch 磁盘编译缓存。

受保护复测 `smoke05`：

| 指标 | 无盐 | 独立盐 |
|---|---:|---:|
| 成功请求 | 48/48 | 48/48 |
| 输入 token | 68,844 | 68,844 |
| 输出 token | 12,288 | 12,288 |
| 正式窗口 | 14.818429 s | 15.067769 s |
| 输出吞吐 | 829.237724 token/s | 815.515532 token/s |
| 平均 TTFT | 4.913862 s | 4.959783 s |
| HTTP cached token | 0 | 0 |
| computed-prefix 命中调用 | 0 | 0 |
| Scheduler CPU，含预检查 | 513.543 ms | 437.965 ms |
| Store / seal 指令数 | 19 / 1 | 20 / 1 |

两臂各两个 rank 均记录 serving scope 已封口，**无 JIT scope 违规，正式窗口无新增 Triton 缓存元数据**。

独立盐吞吐低约 **1.65%**。这是单次配对小规模观察，不能解释成稳定性能退化；可以得出的结论是：**此前的大幅盐隔离收益没有在这轮受保护复测中复现。**

盐仍减少了部分失败匹配的 CPU 工作，但没有转化成该轮端到端吞吐提升。

### 5. 证据边界

- 这是 48 请求诊断子集，使用不同于原 leaderboard 的 HTTP 客户端，不替代原 200 请求 benchmark。
- 没有进行完整 MMLU-Pro 答题质量评估。
- 首次污染复现与受保护复测的机器／驱动不同，不跨机器比较绝对吞吐；结论来自各自内部证据。
- 使用原 school-qualified host binary，而非 #290 中重新构建的 host binary，保留该部署差异。
- 中间未完成的运行不计入结果；`smoke05` 两臂完成并正常退出，设备释放已有记录。
- **已确认的是后续首次复现中的编译污染，不是原 #290 主机上的编译事件。** 没有原主机对应缓存／编译证据，不能把两者等同。
- 此次没有验证完整 arrival controller 在真实可复用 checkpoint 或高搬运代价负载下的收益。

### 6. 对历史结果的建议解释

保留原始性能数字，但将其表述为：

> 逐请求缓存盐／重算配置的单次性能筛选正信号；因果归因未完成。后续复现发现预热不足可产生类似的大幅差异，在补齐预热并启用 Triton 服务期封口后，小规模配对没有复现大幅收益。

不宜将这些历史数字直接作为“动态 KV arrival 策略在 BetterScale 基座上已验证有效”的证据。

### 7. 已有证据入口

- [BetterScale 调查记录及受保护复测](https://github.com/vLLM-HUST/BetterScale/blob/6323e926/.agents/skills/repo-knowledge/scenarios/diagnose-qwen35-serving-performance/mmlu-cache-salt-static.md)
- [复测协议与复现脚本](https://github.com/vLLM-HUST/BetterScale/tree/6323e926/.agents/skills/repo-knowledge/scenarios/diagnose-qwen35-serving-performance/cache-salt-smoke)
- [四文件预热覆盖身份](https://github.com/vLLM-HUST/BetterScale/blob/6323e926/.agents/skills/repo-knowledge/scenarios/diagnose-qwen35-serving-performance/cache-salt-smoke/overlay-source.json)

完整原始归档及汇总已保留；本正文不公开请求原文、回答或内部机器访问细节。
