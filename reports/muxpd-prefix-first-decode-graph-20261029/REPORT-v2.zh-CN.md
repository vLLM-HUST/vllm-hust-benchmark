# 实际 prefix 首 D 图归约边界复核 v2

真实 TP4 图观测及独立审计通过：按 rank 对应，插件实际首 mixed decode 的 GDN0 TP 归约输入与历史 native 输入逐字节相同；图输出与此前单行 private B
逐字节相同，与 native A/原大形状 private C 均有 **625/2048** 个元素不同。这个结果把此前私有重放发现的数值差异延伸到了实际插件图执行边界。它不证明
geometry/graph 模式是唯一原因，也不证明答案问题已修复。

本次物理卡为 **2,3,4,6**，历史 native DTP 为
**0,1,2,7**；硬件及拓扑不同是混杂。`same_hardware_causal_attribution_allowed=false`，独立审计也明确
`geometry_or_graph_unique_causal_attribution_allowed=false`。这里的 reduction attribution
仅指同输入的归约边界比较，不能用作同卡控制实验或唯一因果结论。

## 固定实验与轨迹门槛

模型 Qwen3.5-35B-A3B，BF16 TP4/EP4，sync，FULL_DECODE_ONLY [1,2,4,8]，原 fusion
设置、HCCL_DETERMINISTIC=true、batch invariance 关闭；保持 clean `6c606f789450a45ef35afee398b0a3e516caa1a1`
插件源码及原外部 P-prefix。核心 vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`、Ascend
`03766ac696fde5ab1980d80ca0b8543d3580c989` 及既有 dirty 文件证明均按冻结输入验证。本次不是对 PR27 新文档 HEAD 的运行验证。

固定观察/对照各一次，运行目录 `runs-v2`，labels `prefix-dtp-graph-v2-observed-01` 与
`prefix-dtp-graph-v2-plain-01`。双方使用同一诊断 worker 和构造兼容 context；plain 只关闭 tap。原 batch7
的8请求、seed输出1后加入7个P、07338输出208时abort/drain和max8192保持不变。

两个角色全部8条完整保留前缀分别复现历史 `prefix-quality-full-01`，且观察/对照8条全部相同：seed 07151保留218
tokens，其余7条各208。独立审计验证16次历史前缀比较与8对新观察/对照前缀。这个诊断截至目标208 tokens，不是新的full200答案评测。

## 实际图结果

原启动 descriptor1 图中仅在 GDN0→原 TP→MoE0 窗口插入两次copy；请求期冻结首mixed D的input/output两行。四rank均记录真实图 replay
+1，capture/fallback/uncaptured/forward-failure增量全0，启动3个hook与请求期2个hook均恢复。未额外执行归约、补padding或切回eager取得“实际图”结果。两行及快照额外逻辑tensor
payload为16KiB/rank，不是模型总内存节省指标。

历史对照定义：A为native实际mixed D输出；B为同一各rank局部D输入在 `[1,2048]` 形状的私有原TP归约输出；C为原 `[9533,2048]`
形状，仅D对应行非零的私有原TP归约输出。B/C的历史执行为private eager，不能本来就当实际插件图；本次新增真实图结果如下，4个rank结论一致：

| 比较                                   | 原始BF16字节 | 不同元素 | 最大绝对差 |                RMS差 |
| -------------------------------------- | ------------ | -------: | ---------: | -------------------: |
| 实际图局部input vs 对应native局部input | 全相同       |   0/2048 |          0 |                    0 |
| 实际图output vs B                      | 全相同       |   0/2048 |          0 |                    0 |
| 实际图output vs A                      | 不同         | 625/2048 | 0.00390625 | 9.40213873039046e-05 |
| 实际图output vs C                      | 不同         | 625/2048 | 0.00390625 | 9.40213873039046e-05 |

这里没有要求不同rank的局部input相同；验证的是同一rank的新旧input。实际图最终output跨rank完全一致。独立审计从8条图raw BF16及8条对应NPY、16条native
raw及16条NPY重新核验，未只采信主分析的比较摘要。

## 构造兼容与保留的失败

原 graph v1 observed 于 **19:38:38.401218 UTC** 终态exit1：四rank启动图捕获完成，但原
`DecoupledMuxPDScheduler.__init__` 精确worker类名guard拒绝诊断子类，LLM构造未返回，0请求。plain未启动，v1矩阵与分析保持
**FAILED_INCOMPLETE_MATRIX**；不称为正常shutdown-returned，不挑选成功重试。原冻结source/preparation/rootreview、失败observed整树及分析/disposition/独立失败审查共197个绑定文件纳入legacy
payload，排除compiler/cache。

v2是独立设计、冻结和命名的兼容修正。只在driver的LLM构造期间，以owned context临时包装原scheduler
initializer；验证精确诊断worker及原继承方法后，仅在原initializer调用内可逆映射同一parallel-config的worker_cls。原scheduler类、selector、调度/更新方法及原guard仍执行。双方receipt均确认一次原构造、字段和class
initializer恢复、其他scheduler方法身份不变，真实scheduler实例、类型、来源及config
alias通过核查。原图观测组件、P-prefix、plan和数值输入逐字节复用v1。

观察/对照分别于 **19:54:19.658430 UTC**、**19:57:48.223221 UTC**
终态PASS：exit0、errors空、owned成员空、物理四卡空闲，无controller额外cleanup或forced kill；child
collection_complete与shutdown_returned均true。两个executor日志都有grace超时后SIGTERM，没有SIGKILL；不能写成全部worker自然退出。

## 结论边界与来源

这支持继续研究不同归约执行条件造成的实际插件边界数值偏移；没有观测HCCL内部算法或cached native
pointer，未证明这一处是07338失败的唯一根因。原full200答案不退化门仍FAIL，净正确数或局部数值吻合不能替代该门。没有新增吞吐、Goodput、TTFT、TPOT或总显存收益结果。

- 本版freeze：`b27108e753506c0fded2d64a33d37ef4acd8c6aff275511774b9e2d60b9e48e1`。
- 主分析 `analysis-v2.json`：`bfc4d4cfd3fd04e085511d10963b81ff02bd9585fc223eab1cf2ed0a9a0985fd`，PASS。
- 独立审计
  `summary-v2.json`：`01763ac2be24a662826d1f9d19d46eab270cdec505613bc7611b0ed8ff3c046a`，PASS；原脚本、测试、日志及primary
  crosscheck均按原字节入包。
- [历史native DTP固定报告](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/cdaedefd20d9161b790078cd1fb6573ac88e0254/reports/muxpd-first-decode-collective-20261029)及[原始Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-first-decode-collective-20261029)提供A/B/C来源；更早conv/full
  controls依赖见随包external-dependencies。

本报告封包时尚未公开上传或匿名读回，**公开可下载验证未完成**。本地archive的具体SHA、member数和gzip/member校验结果由 `archive-record-v1.json` /
`local-verification-v1.json` 单独记录。新归档不覆盖旧失败或旧下载INCOMPLETE；模型、运行时二进制及全部历史full
controls仍为外部依赖，不能声称完全独立可移植重跑。
