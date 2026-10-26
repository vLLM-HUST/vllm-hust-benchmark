# 质量、实用原生参考与 profiler：v5 准备草稿

本稿从已合并 canonical v4 提交 `ef849ef49577c491fee68317ffc96d52796cd379` 追加。当前只准备本地可审阅材料，尚未冻结 v5 的终态截止点、support 成员或上传资产。group 质量仍由运行负责人独占收集/评分；本稿不读取或引用其部分输出。默认原生 v2 两轮及 raw/group 混合 profiler 两轮均等待设备执行和既有审计。

## 已知质量结果及其边界

冻结任务为200题 MMLU-Pro、原5-shot CoT输入和提取器、8192输出预算、25个八请求批次。截断始终判错、分母始终200。实验使用 Qwen3.5-35B-A3B、BF16、实际910B2四卡、TP4/EP4；插件运行源码为 f7，Core 为 d0f22d2，Ascend 为03766ac，HCCL=true 和共同 fusion 后端保持冻结配方。原生保留其 chunked prefill 策略，插件采用完整P与独立D调度。这一调度/批形状差异是比较的一部分。

| 角色 | 终态收集 | 正确 / 200 | stop / length | 既有质量结论 |
| --- | --- | --- | --- | --- |
| Native1 | 完成，已在 v3 保存 | 165 | 191 / 9 | 严格无截断条件失败 |
| Native2 | 完成，已有新的本地胶囊 | 165 | 191 / 9 | 严格无截断条件失败 |
| Raw | 完成，已有新的本地胶囊 | 166 | 193 / 7 | 严格和补充答案不退化门槛均失败 |
| Group | 本稿未接纳终态结果 | 待定 | 待定 | 待原生产者/评分器完整结果 |

两原生控制的全部200条 token数组、文本与终止记录一致，共同正确集为165题。原生配对描述没有构造假候选或调用候选验收门槛；其最终 inventoried 文件范围由后补 custody 说明明确限定，未追改执行版报告。[原生配对原报告](originals/NATIVE-PAIR-README-v2.md.source)及[成对数据](analysis/native-pair/paired-description.json)。

Raw 相对任一原生有162题共同正确、3题由对变错、4题由错变对、31题共同错误。新增错题为 chemistry03766、computer-science10672、law01741。前两项是自然停止的错误答案，后一项新增8192截断；因此不能全部归因于原先严格协议的截断问题。四项收益也完整保留，其中三项挽回原生截断。+0.5个百分点的总分差不能越过预冻结共同正确集否决规则，区间和不显著性也不证明等价。[完整七项变化与已有区间](originals/RAW-QUALITY-README-v1.md.source)。

Raw supplementary collection通过，真实P进度/D退役覆盖通过（25/25批，每批10次，共250次）；答案条件失败使组合门槛仍失败。两个评分CLI退出0只说明评分完成。所有200案例、gold、输入、budget、parser、scorer及门槛未变。原2048/8192严格失败历史不被独立补充指标覆盖；补充指标在候选输出前冻结的公开协议仍是[原身份](../quality-protocol-v1/README.md)。

N2/raw 的收集控制器退出0并保留返回关闭/空设备回执，既有胶囊构建者验证其终态和成员集合。两原生与raw日志均保留 native worker 在宽限期后的 SIGTERM fallback；不能称为全局优雅或无信号关闭。当前设备占用属于后续运行，不能用于反推先前清理失败。本次准备只核对小元数据及复制摘要，没有再次解包大胶囊、重新评分或重哈希模型权重。

## 尚待的两个性能相关证据族

| 族 | 已冻结内容 | 此时缺少的既有交付 |
| --- | --- | --- |
| Practical native default backend v2 | 相同固定请求/token/到达/预热与两测量窗口；真正原生worker；async=None须读回true，npugraphEx=true/static=false；新增实际 inner-worker 识别修复 | 两个新鲜进程的终态、实际worker/backend/队列读回、原两轮描述分析、归档与复核 |
| Mixed raw/group profiler | 相同4×P128/D512 seed，在各返回32 token后加入P12000/D128；由入场前到长P首token捕获；同f7、fusion、HCCL=true | raw/group实际四rank trace、既有单轮审计和匹配捕获比较、终态归档与复核 |

默认参考 v1 已在计时前失败，0窗口，保留于[v4](../performance-lifecycle-v4/README.md)。v2只修正RPC参数为Wrapper时的真实inner worker读回，保留v1失败、相同workload/config/timing和接受条件。它是固定配方中的实用默认异步/后端参考，仍显式使用HCCL=true等控制；不宣称vendor最佳或全部默认，也不混入同步raw/group主ABBA估计。

Profiler是独立诊断，不提供吞吐估计。其实际CANN API计数、同批P进度/D返回、rank/PID/设备绑定和原生D replay须由冻结审计读取真实trace。即使通过，也不能单独推出Cube/Vector资源分区或内核物理重叠收益；inclusive host API duration之和不是端到端时延。反之，审计或收集失败必须保留，不能用纯D旧profile替代。当前 v4 的无profiler主性能结果及尾部暂停补充分析保持自己的范围。

## 源码审查与处置

[批不变算子审计](originals/BATCH-INVARIANT-SOURCE-AUDIT-v1.md.source)发现当前可发现的 Triton-only替换不覆盖FIA/add-RMSNorm/sum，GDN自定义状态核、MoE分组GEMM仍在其显式替换之外；预加载custom库使“禁用custom op”不等于算子必然缺失。`strict`控制通信归约，不能保证这些局部算子跨批次一致。source98的HCCL=true profile有意拒绝混入另一种配置；无源码依据将开启开关当作已完成修复。

[独立总体审查](originals/RAW-QUALITY-FAILURE-SCOPE-REVIEW-v1.md.source)保留图模式支持边界与可选的有界首次分歧定位建议。它不授权新设备实验，也不是要求无限重试或普适逐bit等价。本稿没有执行该诊断。

#11 的有界数值调查处置不等于通用质量证明。#23当前仍是已观察到的未解决质量问题，#3及PR10最终处置仍待既有有限评估；PR10保持draft。如果group后来完整保留raw的200题行为，应报告“保留已测raw行为及其原生相对损失”，不能描述为修复raw。任何实验性合并与“质量问题已修复”必须明确区分。这里不预判group、默认参考或profile结果，也不因更高总分而降低旧门槛。

## 发布准备状态

N2/raw各有37个原件成员的既有本地胶囊；其[小元数据](staging-state-v1.json)保留原构建回执、文件名、大小和SHA。这里不宣称这些新增资产已上传，未建立计划URL为可用下载的链接，也未将这两个包当作最终v5全体。N1已公开胶囊不重建；v2/v3/v4及协议的400个旧tracked文件逐字保留。

待已有实验终态后，负责人才能选择明确cutoff，将新终态胶囊和固定support成员送入原 `build_capsule.py` 链路，生成另存的最终清单/报告/校验和，再由root上传并取得匿名公开回验。无需重造归档工具；不改旧builder的exit0规则或既有Manager专用exit130类型。本草稿与其校验回执都不是最终发布回执。
