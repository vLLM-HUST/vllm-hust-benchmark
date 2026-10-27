# vSpec：问题、原理与实现定位

研究日期：2026-10-26。源码观察固定于
`vLLM-HUST/vllm-hust-vSpec@4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae`。本轮为文档与关键源码阅读，不是全量审计或 NPU 实测。

## 1. 它解决哪两个问题？

**通用问题：自回归 decode 的昂贵串行依赖。** 普通生成每新增一个 token，通常都要走一次 target 模型。小 batch
下，这些步骤可能不能充分利用计算资源，同时重复支付权重访存、调度和 kernel 启动成本。投机解码用便宜的 proposer 先生成候选，再由 target
一次验证多个位置，以额外草稿计算换取更少的昂贵串行轮次。它并不消除生成依赖，也不是免算后续 token。

**vSpec 聚焦的运行时问题：固定投机长度不一定划算。** Draft 速度、接受率、活动 batch、上下文长度和执行图都会改变最优投机预算。多猜几步既可能多提交几个
token，也可能只是多做无用计算；高 batch 时 target 的原生执行已经比较饱和，额外验证和 draft 更可能成为负担。

这是我们对实现目的的归纳；不代表所有场景都有性能收益。

## 2. 一轮投机怎样工作？

以严格 greedy、线性草稿、gamma=4 为简化例子：

```text
已确认前缀 h
  ↓ proposer 便宜地生成候选
候选 a → b → c → d
  ↓ target 对已知候选序列做带因果 mask 的批量验证
分别计算 P(.|h)、P(.|h,a)、P(.|h,a,b)、P(.|h,a,b,c) ……
  ↓ 假设 a、b 匹配，c 不匹配
提交 a、b 和 target 在第三位置的纠正 token；丢弃其余候选
  ↓ 更新/对齐状态，开始下一轮
```

能并行验证，是因为候选 token 已经给出，target 无须等自己生成前一个 token
才构造下一位置的输入。首次拒绝后的条件前缀错误，不能继续把其后的验证结果当作正确后缀提交。若全部接受，通常还可提交一个 target bonus token；EOS、长度上限等会截断实际输出。

严格 greedy 可用目标 argmax 匹配解释；随机采样则不是简单比较两个采样 token。标准精确 speculative sampling 使用接受概率
`min(1, p(x)/q(x))` 和拒绝后的修正分布，目标是保持 target 分布，而非保证同一随机种子下逐字相同。算法基础见
[Leviathan 等，ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)。本轮没有逐条审计 vSpec
所有随机采样后端。

## 3. vSpec 加上的核心：动态 gamma

`gamma` 表示一轮的投机 token 预算，启动上限同时限制 buffer 和图捕获范围。一般自适应路径按 decode step 决策，并在一个 batch 内采用统一
gamma；这不等于为每个请求独立选择完全不同的执行长度。

### Profile 策略：用成本模型选预算

源码 `GoodputController` 结合离线 forward 延迟模型、在线逐位置条件接受率和在线耗时修正。对预算 k：

```text
预期提交 token / request = 1 + p1 + p1*p2 + ... + p1*...*pk
预期吞吐 = batch_size × 预期提交 token / 一轮预计耗时
```

串行 Draft/EAGLE 的一轮耗时包含多次草稿 forward、target 验证及额外开销。逐位置接受率只统计已接受前缀及首次拒绝，不能把拒绝后未有效验证的位置全部记成失败。

### Online 策略：从运行反馈学习

`OnlineGammaController` 不要求离线 profile：按活动 batch 的二次幂 bucket 分组，把不同 gamma 当作候选动作，用已提交输出 token /
完成间隔作为奖励，以滑动统计、UCB 探索和迟滞控制切换。它优化的是有效输出速率，而不是单独最大化接受率。这里的 Goodput 不应自动解读为“满足某个业务 SLO 的请求吞吐”。

例如，**纯说明性假设，非测量**：某配置一轮平均提交 3 token、耗时 20 ms，得到 150 token/s；更长草稿一轮提交 4 token，却耗时 35 ms，仅约 114
token/s。提交更多 token 不等于更快。

证据：[控制器](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/controller.py)、[在线控制器](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/online.py)、[Adaptive 设计与约束](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/docs/vspec_adaptive.md)。

## 4. 不只是控制器，还有运行时适配

- **多种 proposer 接入**：patch dispatcher 有 Draft、EAGLE/EAGLE3、MTP、DFlash 分支。不同方法的草稿来源和开销不同，不能当作一种算法。MTP
  和 DFlash 路径借助宿主原生能力；存在分支不等于已经完成硬件验收。Adaptive 文档明确 DFlash 缺少匹配 checkpoint 的 NPU 端到端结论。
- **Graph / async 协调**：动态 gamma 改变 target 验证宽度、draft 图、占位 token 和异步反馈形状，需要额外捕获与状态对齐；不是只修改一个整数配置。
- **关闭投机的状态问题**：gamma=0 后 draft KV 可能落后。文档区分 sticky cohort（这批请求保持 target-only）和 shadow draft
  同步（仍支付草稿计算），不能同时宣称零草稿成本且无条件恢复。
- **请求补入控制**：`OnlineRefillController` 在给定 running count/gamma 下比较继续 decode 与补入等待请求的收益，试图摊薄 refill 的
  prefill/调度开销。代码有等待保护和 cohort 模式，但本轮未核实所有启动预设的实际启用路径；吞吐动机也不能替代 TTFT/公平性评估。
- **验收热路径优化**：严格 bare-greedy 分支直接取 argmax，避免通用概率路径中部分大词表张量转换/复制。投机控制以外的系统优化也可能贡献总体收益。

证据：[patch 分派](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/patches.py)、[refill](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/adaptive/refill.py)、[验收路径](https://github.com/vLLM-HUST/vllm-hust-vSpec/blob/4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae/src/vllm_hust_vspec/backends/eagle_rejection.py)。

## 5. 正确性边界：不能笼统叫“无损插件”

源码默认 `confidence_accept_margin=None`，严格 greedy 路径使用目标 argmax。但显式设置 confidence margin 可走 relaxed
acceptance；README 也称它为近似验收。它不能直接继承精确投机解码的分布/输出保证。

不要混淆两个功能：**entropy stop 是少生成/使用后续草稿；confidence accept 是放宽接受标准。**
前者改变工作量，后者可能改变输出语义。另需把算法理论保证与具体硬件数值、惩罚项、KV 对齐和图切换的实现验证分开。

## 6. 当前判断与下一层问题

**我们的判断：vSpec 更适合被理解为“多 proposer 的自适应投机运行时扩展”，而非一个新的小模型或单一投机算法。** 值得研究的是预算控制、执行开销与状态正确性如何一起闭环。

当前源码 README 的方法范围已超出首轮官网摘录中的 Qwen2.5 Draft/EAGLE；目录状态也包含不同证据维度。后续研究应固定具体版本/预设，不把官网旧结果移植为当前 HEAD 的收益。

尚未完成：所有 backend 的调用链；gamma 决策是否在各预设实际生效；随机采样语义审计；控制器与热路径优化各自的因果收益；refill 对
TTFT/公平性的影响；负载变化时学习成本与稳定性。

首轮只读源码位于临时 checkout，不作为持久依赖。以上不可变 GitHub 链接是可恢复入口；未安装插件、未修改宿主、未运行性能实验。
