# 系统研究报告：分层分析与脚本契约

## 目标与图的组织

报告首先解释研究问题和系统结构，再按照模块、层次、请求链路及工作流组织指标和图。不同系统有不同分层，不能把某个项目的四层固定套用所有系统。

本项目的真实Agent对照使用以下层次，沿端到端链路关联，而不是混放图：

| 层次 | 回答的问题 | 必须区分的指标 |
|---|---|---|
| Controller | 准入、暂停和恢复改变了什么？ | active/paused program、队列、动作事件；不是引擎running数 |
| Engine（含设备） | 排队、prefill、decode和资源哪个限制执行？ | 引擎请求/吞吐/延迟、缓存池GiB；设备计算/总HBM/带宽另列 |
| Program | 每个Agent是否持续推进？是否有饥饿或工具等待？ | 每Agent进展分布、零回复、总Agent秒时间分配 |
| Request | 客户端实际收到多少？平均、长尾和输入长度如何？ | 正常返回调用、交付输出、端到端延迟、未完成/错误 |

汇报顺序：目标及结论 → 系统结构和对照条件 → 分层图/数据 → 跨层机制解释与反向结果 → 已知、未知和下一步。每幅图说明读哪里、支持什么、不支持什么。

## 规范入口

使用项目既有的Python环境和`benchmarks/plot_run.py`配色/风格，不新增运行时依赖或推理任务：

```sh
.venv/bin/python benchmarks/report_agent_comparison.py \
  --run-dir results/swe-bench-pro-20261018/pressure-diagnosis-20261019-r1/performance-window128-20261021-r2 \
  --window-seconds 1200
```

输出默认位于输入run的`derived-reports/layered-1200s/`，不会覆盖历史图或原件。

可选参数：

- `--question`：本次研究问题，打印到报告开头并保存在数据/manifest；默认是本项目Controller对照目标。
- `--output-dir`：新的派生报告目录；非空且没有本生成器manifest的目录会拒绝，输入目录/原件树/其父目录也会拒绝。
- `--phase-boundaries 300 600`：两个内部时间分界，默认是窗口的1/4和1/2。必须满足0<分界1<分界2<窗口。
- `--rate-window-seconds 10`：引擎向后速率窗口和客户端不重叠桶宽；最后不足整桶时用实际秒数。

脚本只读本地原件，不含推理、集群访问、提交作业、发布或改策略的调用。

## 输入适配器的实际覆盖范围

`benchmarks/agent_report_data.py`目前支持一个Native→Controller配对、一个TP后端的真实Agent归档：

```text
<run>/
  native-settings-private.json              # 可选；实际Agent预算
  controller-settings-private.json          # 可选；有记录时核对两臂预算
  terminal-extract/
    pair-1-native/task-*/arrival.json
    pair-1-native/task-*/agent-events-private.jsonl
    pair-1-controller/task-*/arrival.json
    pair-1-controller/task-*/agent-events-private.jsonl
    native-evidence/observations-private.jsonl
    controller-evidence/observations-private.jsonl
    native-evidence/action-log-observations-private.jsonl       # 可选
    controller-evidence/action-log-observations-private.jsonl   # 可选
```

到达文件提供`actual_start_monotonic`和`epoch`；观察文件提供`monotonic`、Prometheus文本`engine`，可选`scheduler`和`programs`快照。Agent事件沿用现有`agent_query_start/return/exception`及`tool_start/return`结构。每次query由现有request ID关联，不另造身份。

Agent数从实际文件读取，有设置时与`batch.program_instances`核对；缓存池容量从本轮`vllm:cache_config_info`的`kv_cache_memory_bytes`读取，不硬编码128或16GiB。检测到多个后端、容量变化、预算不一致、缺失Agent文件或计数器重置会停止并报出原因，不能混合成一个正常对照。

该适配器不声称支持任意系统、在线到达协议或多后端求和。其他系统先复用本规范，再接入其实际数据适配器；原有`plot_run.py`仍用于其已有回放数据，不被替换。

## 输出：数据→图→报告

- `report.md`：自动生成的中文分层报告和数字表；图按Controller、Engine、Program、Request分类，包含跨层限制、未完成请求和缺失项。
- `controller.png/svg`：准入、暂停、队列和动作。
- `engine-load.png/svg`：执行/等待、缓存GiB、prefix token命中率、本地输入计算与输出生成速率。
- `engine-latency-and-batches.png/svg`：分段queue/prefill/decode均值和原始iteration桶分布；不将其自动称为实际scheduler预算利用率。
- `engine-token-latency.png/svg`：引擎TTFT与TPOT聚合直方图均值，明确不包含Controller全部等待。
- `program.png/svg`：全部Agent进展热图、完成调用分布、时间分配；不删除零回复Agent。
- `request.png/svg`和`request-latency-by-interval.png/svg`：交付速率、延迟分布、长度关系、分段均值/p95，不筛除不利结果。
- `data.json`：结构化统计、精确采样边界、sum/count/bucket增量、缺失项和预算。
- `summary.csv`：分段核心指标；`engine-controller-samples.csv`与`normal-client-returns.csv`是可检查的无题面数字记录。
- `source-sha256.json`：原件来源摘要；`report-manifest.json`：生成器/样式源码SHA、参数及生成文件SHA。

自动生成报告负责数值和证据边界，不替代研究者审阅机制解释。新增解释应引用该数据，不能改写CSV/JSON或手填一份不同口径的报告数字。需要修改图，修改版本化脚本并重新生成，不在图片上手工修数据。

## 统计及缺失值规则

1. 时间按各臂最早真实Agent到达对齐，窗口0≤t<指定秒数。引擎段内首末采样覆盖可能小于理想窗口，记录实际起止与分母；不外推补齐。
2. 状态平均按相邻快照的持续时间加权。采样缺口不等于有更高频数据，不能冒充逐事件记录。
3. 直方图均值只用段内sum/count增量；bucket分位值只报告范围，不能冒充精确值。TTFT、TPOT、已完成延迟的样本集合不同，不能盲加或逐个匹配。
4. 客户端p95从正常返回调用原始时间求得，按返回时间归组；未完成与异常单列。非流式客户端不能凭请求首尾重建逐token时间。
5. 区分设备总HBM占用、缓存池使用量、Controller容量估计；缓存GiB仅为池占用率×池容量的等价换算。低KV使用率不能推出计算/带宽未饱和。
6. 区分引擎生成和客户端交付；全局完成直方图与正常返回数的差异保持可见，不直接认定为丢失请求。
7. 缺指标在JSON保持null，在报告/图标明未记录；不得用0、插值或另一轮历史数据补成实际观测。直方图桶变化、负增量或Inf/count矛盾报错。
8. 输出不含题面、工具内容、补丁或服务地址。发布是独立动作，不由绘图脚本隐式执行。

## 验证要求

对解析/统计边界做语义测试，并用已有真实归档生成报告；不能用自编mock性能证明系统收益。至少核对：计数器重置、直方图增量、非均匀采样、未完成请求、缺失字段、源目录保护。

同输入和环境重复运行应得到一致文件摘要；换时间窗口后，数字和标签都必须来自新窗口。检查PNG的轴、图注、字体和截断；保留SVG与数据。完整实验、硬件饱和和稳定因果收益需要各自证据，不由报告生成成功代替。
