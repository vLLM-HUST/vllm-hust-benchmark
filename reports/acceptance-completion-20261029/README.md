# 当前证据与缺口补齐（2026-10-29）

| 范围                     | 本次取得的结果                                                                        | 当前结论                                               |
| ------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| M01 参考                 | 原独立标定依赖图三场景回验；主场景参考 19.0819 TFLOPS/芯片，原计数轮诊断比值 37.2302% | 有参考模型，不是正式无插桩验收；B0 参考绑定仍缺        |
| M02–M04 / M06 质量适用性 | 原 out256 回答按原评分器离线复算：B0 21/200、B1 19/200；全部达到预算                  | 性能/租赁估算保留，不能借用 out8192 高分判业务验收通过 |
| M05 连续补测             | 固定 B0 的 4K/16K 正式窗口分别 153/73 请求，零失败                                    | 两个长度完成；合成压力场景，不代替五项任务质量         |
| M05 优化侧               | B1 预热 24 成功，正式 4K 的 8 请求失败，16K 未执行                                    | 状态卸载的锁页主机内存分配失败；本配置配对稳定性未通过 |
| 五任务日志               | MMLU、HLE、FrontierScience 原记录复核；SWE/Terminal 既有上下文拒绝和预算结束逐项索引  | 保留 HLE 原退出状态缺失和 Agent 失败，不填整体通过     |

## 固定 B0 连续结果

Qwen3.5-35B-A3B、BF16、TP2、同物理 5/6 两卡；B0 实际 vLLM/Ascend 0.18.0/0.18.0。 两组实际输入分别 4115/16407 token，各 8
个固定合成工程提示，C8 闭环、输出 256、ignore_eos、temperature=0、seed=17。 每组 60 秒预热及 300 秒正式发起窗口，均完整排空；预热不计正式成绩。

原始输入、逐请求双时钟、SSE、监控、源码与全部失败见
[原始包](continuous-raw.tar.gz)；[包身份](continuous-capsule.json)、[逐窗口复算](continuous-summary.json)。
吞吐的分子为该正式队列全部成功输出，分母包含正式发起窗口和排空；不能把排空输出全部除以 300 秒。 TTFT/TPOT 采用 V1.5 逐请求定义、nearest-rank P95/P99。每 60
秒窗口另列完整落在窗口内的请求用量和跨界数，时延按到达队列统计并含排空；没有逐 token 到达计数，不冒充精确 token 到达窗口吞吐。
两卡利用率按同轮单调时钟采样中点、左值保持、时长加权估计；采样估计不当作物理连续利用率置信区间。

## B1 失败与边界

B1 实际基座为 vLLM `0.28.1.post1.dev143+gf18cf803c` / Ascend
`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`，纠正 Worker、请求级 KV 重算与 Microflow 沿原组合生效。 32
次请求准入及两 rank Microflow 生效记录保留；24 个预热成功请求的 cached_tokens 均为 0。 正式阶段在 `host_state.py` 的
`torch.empty(...pin_memory=True)` 发生 `aclrtMallocHostWithCfg` 207001 / OutOfMemoryError，随后
EngineDeadError、8 条 SSE 错误。 不能将驱动提示的“memory/storage”直接解释为磁盘满；也没有据此证明宿主总内存或 cgroup OOM-kill。

原控制器稍后将已变更父子关系的本轮 worker 记为“Foreign NPU process”；该回执保持原样，不能倒置为先于引擎 OOM 的根因。两张保留卡已核实释放后封存。 另一次 B1 r1
在模型启动前端口占用检查失败，无模型/请求；后来端口可用才用新目录启动 r2，不推定未捕获的占用原因。
不降低并发、关闭状态迁移或重试择优来覆盖本次失败。后续联合性能/成本补测应先解决并验证该资源分配问题。

## 复算与原件

- [有效算力参考方法、结果与公开标定包](reference-method.md)
- [out256 原回答评分投影](out256-quality-audit.json)：只有 SHA、逐题布尔结果及长度，不发布私有题答；原评分器 fallback
  可提取未完成推理中的选项，不证明是最终答案。
- [五任务稳定性与剩余边界](five-task-stability.json)、[HLE 生命周期](hle-lifecycle-audit.json)
- [原固定基线性能/成本复算](../acceptance-gaps-20261029/README.md)
- [此前固定 B0/B1 检索配对](../long-context-fixed018-paired-20261029/README.md)：双方 6/6，与本次连续稳定性不同。

```sh
python scripts/audit_fixed018_continuous.py reports/acceptance-completion-20261029 \
  --capsule --output /tmp/recomputed-continuous.json
```

## English summary

The fixed 0.18/0.18 B0 completed both prospective C8 synthetic continuous windows (153 requests at
actual 4115 input tokens; 73 at 16407; no failures). Corrected B1 completed 24 warmup requests, then
failed all eight initial formal 4K requests because pinned host-memory allocation failed during
state offload. Its 16K phase did not run. Failure evidence and the pre-model port-check failure are
preserved; reserved devices were released.

Retained out256 task outputs were scored offline using the original scorer: B0 21/200, B1 19/200.
Out8192 quality scores cannot qualify this service/cost measurement. The independent B1 reference
model was reproduced at 19.0819 TFLOPS/chip, yielding a 37.2302% diagnostic against the counted run;
formal uninstrumented acceptance and B0 binding remain open. Five-task logs are indexed with
original failures and unknown exit status retained. These are measured outcomes and specific
remaining gaps, not a claim that all acceptance criteria passed.
