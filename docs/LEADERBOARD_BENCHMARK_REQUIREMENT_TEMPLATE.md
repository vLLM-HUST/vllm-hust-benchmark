# Benchmark 需求征集模板

## 目的

当前 benchmark 体系只覆盖部分场景和部分模型。为补齐支持面、规划 official baseline、以及安排 benchmark 开发与执行优先级，建议将需求征集分成两层：

1. 需求登记表：由需求提出方填写，优先描述业务目的、目标场景、模型、指标和时限。
2. 技术归一化评估表：由 benchmark 维护方填写，将需求翻译成可执行的 same-spec benchmark 请求，并判断是否可直接支持。

如果一个需求包含多个模型、多组参数值或参数扫描，请不要把所有组合硬塞到一行里。推荐使用“主需求表 + 参数实例表 + 技术评估表”三张表协同管理。

## 使用原则

| 原则 | 说明 |
| --- | --- |
| 先收业务意图，再做技术翻译 | 需求方不必直接写 `vllm bench` 命令 |
| 区分探索性 benchmark 与正式对比 benchmark | 只有正式对比才强制走 official baseline / same-spec 约束 |
| same-spec 是 compare 合同，不是原始参数长相一致 | 官方对照时以归一化语义参数为准 |
| sweep 单独拆表 | 单点 benchmark 和参数空间需求不要混在一行 |
| 模型必须可唯一识别 | 不能只写简称，至少要有 canonical model 或 repo id |
| 每条需求都要能排期 | 必须带优先级、截止时间、成功标准 |

## 表 1：需求登记表（需求方填写）

用途：收集“为什么测、想测什么、何时要、用来做什么决策”。

| 字段 | 是否必填 | 填写说明 | 示例 |
| --- | --- | --- | --- |
| 需求编号 | 是 | 唯一标识，建议 `BR-YYYYMM-序号` | `BR-202606-001` |
| 提出团队 | 是 | 需求归属团队 | `推理平台组` |
| 需求接口人 | 是 | 便于追问细节 | `张三` |
| 提交日期 | 是 | 提交时间 | `2026-06-03` |
| 业务背景 | 是 | 为什么要做这个 benchmark | `评估长上下文在线客服场景是否可以替换现网方案` |
| 决策用途 | 是 | 结果用于什么决策 | `版本发布`, `方案选型`, `对外材料`, `回归验证` |
| 是否要求正式对比 | 是 | `否` / `是，对比 official baseline` / `是，对比已有内部基线` | `是，对比 official baseline` |
| 目标业务场景 | 是 | 用业务语言描述，而不是直接写 CLI flag | `长上下文在线问答` |
| 期望 benchmark 场景 | 否 | 如已知可填写 registry 场景名 | `prefix-repetition-online` |
| 目标模型 | 是 | 建议填 canonical model / repo id；多个模型可在“参数实例表”展开 | `Qwen/Qwen2.5-14B-Instruct` |
| 模型精度 | 否 | FP16 / BF16 / INT8 / FP8 等 | `FP16` |
| 目标硬件 | 是 | 芯片型号、卡数、节点数 | `Huawei 910B3, 1 card, 1 node` |
| 数据类型 | 是 | `公开数据`, `内部数据`, `脱敏数据`, `合成数据` | `公开数据` |
| 目标指标 | 是 | 可多选：吞吐、TTFT、TPOT、稳定性、成本、长上下文、多模态等 | `TTFT`, `TPOT`, `稳定性` |
| 成功标准 | 是 | 最低判断口径 | `TTFT 不高于 baseline，32K 下吞吐稳定` |
| 优先级 | 是 | `P0` / `P1` / `P2` / `P3` | `P1` |
| 期望完成时间 | 是 | 期望出结论时间 | `2026-06-15` |
| 备注 | 否 | 其它补充说明 | `需要保留对外汇报截图` |

### 需求登记表空表模板

| 需求编号 | 提出团队 | 需求接口人 | 提交日期 | 业务背景 | 决策用途 | 是否要求正式对比 | 目标业务场景 | 期望 benchmark 场景 | 目标模型 | 模型精度 | 目标硬件 | 数据类型 | 目标指标 | 成功标准 | 优先级 | 期望完成时间 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 表 2：参数/模型实例表（需求方或维护方补充）

用途：当一个需求需要多个模型、多个参数值或 sweep 时，用子表把请求拆成可执行实例，而不是把所有取值堆在主表一格里。

建议规则：

1. 一行对应一个候选 benchmark 实例或一个 sweep 维度。
2. 如果是 same-spec 正式对比，需标出是否已有 official spec 可映射。
3. 如果参数是区间或候选集，用统一格式记录，例如 `1|2|4|8` 或 `[1024,4096,8192]`。

| 字段 | 是否必填 | 填写说明 | 示例 |
| --- | --- | --- | --- |
| 需求编号 | 是 | 关联主需求 | `BR-202606-001` |
| 实例编号 | 是 | 同一需求下的子编号 | `BR-202606-001-C01` |
| 模型 canonical id / repo id | 是 | 可执行实例的模型标识 | `Qwen/Qwen2.5-14B-Instruct` |
| 模型精度 | 否 | 实际要求精度 | `FP16` |
| benchmark 类型 | 是 | `serve`, `throughput`, `latency` | `serve` |
| 候选场景 | 是 | 当前希望映射的 benchmark 场景 | `prefix-repetition-online` |
| 是否要求 same-spec | 是 | `是` / `否` | `是` |
| official spec 映射 | 否 | 若已知，填写 spec id | `official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3` |
| 数据集/数据源 | 否 | dataset_name 或数据源说明 | `prefix_repetition` |
| 关键参数名 | 否 | 单个参数或参数组名称 | `input_len`, `output_len`, `request_rate` |
| 参数值/候选值 | 否 | 单值、候选集或区间 | `4096`, `256`, `1|2|4` |
| 参数类型 | 是 | `固定值`, `候选值`, `区间`, `待定` | `固定值` |
| 硬件约束 | 否 | 与主需求不同则单独写 | `910B3 x1` |
| 备注 | 否 | 参数解释、特殊要求 | `若做 official compare，需要按 same-spec 展开成 prefix/suffix 参数` |

### 参数/模型实例表空表模板

| 需求编号 | 实例编号 | 模型 canonical id / repo id | 模型精度 | benchmark 类型 | 候选场景 | 是否要求 same-spec | official spec 映射 | 数据集/数据源 | 关键参数名 | 参数值/候选值 | 参数类型 | 硬件约束 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 表 3：技术归一化与评估表（benchmark 维护方填写）

用途：把需求方的描述翻译成 benchmark 平台可以执行、排期和归档的技术请求。

这张表是关键，因为它负责回答：

1. 现有 benchmark 是否已支持。
2. 是否已有 official same-spec spec。
3. 是否需要新增场景、模型支持或参数扩展。
4. 是否存在 official baseline 和 current benchmark 的参数差异，需要做归一化说明。

| 字段 | 是否必填 | 填写说明 | 示例 |
| --- | --- | --- | --- |
| 需求编号 | 是 | 关联主需求 | `BR-202606-001` |
| 实例编号 | 否 | 若评审到实例级则填写 | `BR-202606-001-C01` |
| 维护方评审人 | 是 | 负责归一化与判断的人 | `李四` |
| 当前支持状态 | 是 | `已支持`, `部分支持`, `不支持`, `信息不足` | `部分支持` |
| registry 场景映射 | 否 | 映射到哪个已注册场景 | `prefix-repetition-online` |
| official spec 状态 | 是 | `已有可复用 spec`, `需新增 spec`, `不适用` | `已有可复用 spec` |
| same-spec 约束结论 | 是 | `可直接 same-spec`, `需参数归一化`, `当前不可 same-spec` | `需参数归一化` |
| current benchmark 侧执行参数 | 否 | 维护方归一化后的执行参数摘要 | `backend=vllm; dataset_name=prefix_repetition; prefix_repetition_prefix_len=3840; prefix_repetition_suffix_len=256; request_rate=1` |
| official baseline 侧约束 | 否 | official spec 的关键 same-spec 约束 | `spec_id=official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3` |
| 参数差异说明 | 否 | 版本演进/接口差异/兼容性差异说明 | `registry 默认值用 input_len/output_len，official compare 需展开成 prefix/suffix/output/num_prefixes` |
| 缺口类型 | 是 | `缺模型`, `缺场景`, `缺 spec`, `缺数据`, `缺硬件`, `缺自动化`, `无缺口` | `缺 spec` |
| 建议动作 | 是 | 下一步动作 | `新增 official spec 并补 same-spec 参数模板` |
| 排期建议 | 否 | 建议迭代或 ETA | `下个 sprint` |
| 最终优先级 | 是 | 维护方综合优先级 | `P1` |
| 结论 | 是 | `纳入排期`, `暂缓`, `驳回`, `待补信息` | `纳入排期` |

### 技术归一化与评估表空表模板

| 需求编号 | 实例编号 | 维护方评审人 | 当前支持状态 | registry 场景映射 | official spec 状态 | same-spec 约束结论 | current benchmark 侧执行参数 | official baseline 侧约束 | 参数差异说明 | 缺口类型 | 建议动作 | 排期建议 | 最终优先级 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## 建议字段枚举

为避免不同团队自由发挥，建议对下面字段先统一取值集合。

### 决策用途

| 枚举值 | 说明 |
| --- | --- |
| `方案选型` | 评估多个方案的优劣 |
| `版本发布` | 发布前硬性检查 |
| `回归验证` | 版本或优化后的效果回归 |
| `对外材料` | 用于汇报、宣传或材料输出 |
| `容量规划` | 做吞吐、成本、硬件容量评估 |

### 当前支持状态

| 枚举值 | 说明 |
| --- | --- |
| `已支持` | 当前 registry、模型、参数和自动化都可直接执行 |
| `部分支持` | 可以复用部分能力，但仍需补模型、spec 或参数支持 |
| `不支持` | 当前缺少关键能力 |
| `信息不足` | 需求描述不完整，暂时无法判断 |

### same-spec 约束结论

| 枚举值 | 说明 |
| --- | --- |
| `可直接 same-spec` | 现有 current benchmark 可直接满足 official compare 约束 |
| `需参数归一化` | raw 参数长相不同，但可以归一化到相同语义 |
| `当前不可 same-spec` | 当前无法满足 official compare 合同 |

## 推荐流程

1. 需求方先填写“需求登记表”。
2. 如涉及多个模型或参数组合，再补“参数/模型实例表”。
3. benchmark 维护方基于 same-spec 约束填写“技术归一化与评估表”。
4. 对需要正式对比的需求，维护方再决定：
   - 复用已有 official spec
   - 新增 official spec
   - 暂不支持 official compare，只做探索性 benchmark

## 最小示例

### 示例 1：主需求表

| 需求编号 | 提出团队 | 需求接口人 | 提交日期 | 业务背景 | 决策用途 | 是否要求正式对比 | 目标业务场景 | 期望 benchmark 场景 | 目标模型 | 模型精度 | 目标硬件 | 数据类型 | 目标指标 | 成功标准 | 优先级 | 期望完成时间 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BR-202606-001` | `智能客服组` | `王五` | `2026-06-03` | `评估长上下文客服问答是否可切到新版本` | `方案选型` | `是，对比 official baseline` | `长上下文在线问答` | `prefix-repetition-online` | `Qwen/Qwen2.5-14B-Instruct` | `FP16` | `Huawei 910B3, 1 card, 1 node` | `合成数据` | `TTFT`, `TPOT`, `稳定性` | `32K 下稳定，无明显退化` | `P1` | `2026-06-15` | `希望能给出与 official baseline 的 head-to-head 结论` |

### 示例 2：技术归一化表

| 需求编号 | 实例编号 | 维护方评审人 | 当前支持状态 | registry 场景映射 | official spec 状态 | same-spec 约束结论 | current benchmark 侧执行参数 | official baseline 侧约束 | 参数差异说明 | 缺口类型 | 建议动作 | 排期建议 | 最终优先级 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `BR-202606-001` | `BR-202606-001-C01` | `李四` | `部分支持` | `prefix-repetition-online` | `已有可复用 spec` | `需参数归一化` | `dataset_name=prefix_repetition; num_prompts=200; prefix_repetition_prefix_len=3840; prefix_repetition_suffix_len=256; prefix_repetition_output_len=256; prefix_repetition_num_prefixes=10; request_rate=1` | `official-ascend-jan-2026-v0.11.0-prefix-repetition-online-qwen25-14b-910b3` | `registry 默认值仍用 input_len/output_len，official compare 需要展开成 prefix/suffix/output/num_prefixes` | `无缺口` | `按 same-spec 参数模板执行 current benchmark` | `本周可执行` | `P1` | `纳入排期` |