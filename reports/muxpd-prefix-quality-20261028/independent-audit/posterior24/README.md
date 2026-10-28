# 已终态 24 题运行的独立 CPU 审计

结果：PASS。对象仅为 `quality-runs/prefix-quality-losses-01`，已终态时间为
2026-10-28 07:23:07 UTC；不读取 full200 的 live child，不操作模型进程或设备。

独立程序没有导入/调用质量 driver、controller、原 scorer 或新 wrapper。
它直接读取原冻结 plan、200 cases 与三个旧 run，验证所有 hash，并仅复用
SHA 固定的原 canonical `extract_answer` 和 tokenizer，另写评分与生命周期
审计逻辑。最后才把独立结果与已保存 assessment 逐题交叉核对。

* 保留原完整 batches 3、6、14，24 题；seed 输出恰 1 token 后放 7 peers。
* native1/native2/旧 raw/prefix 正确数为 **22/22/19/22**。
* prefix 24 题全部 `finish_reason=stop`，无截断，无新增 native-correct 错误。
* 原三个失分分别恢复：03766=B（1053 tokens），10672=A（2613），01741=E（590）。
* 完整 token 序列仅 **5/24** 与每个 native 相同、3/24 与旧 raw 相同。
  答案正确率恢复不等于完整生成内容逐字等同；`rows.json` 保留逐题长度、
  token/text hash、首个完整输出差异位置及原评分字段。
* 每批 10 个实际 P progress+D retirement 混合步；每 rank 6 completed / 3 treated
  cohorts；全 rank 共 972 个实际处理调用，逐条验证 prefix geometry、BF16、
  一次原 collective、原 tensor alias 和 isolated P 路由。
* 实际 P prompt 指纹逐项匹配原冻结 token IDs；D 行来自实际 scheduler admission。
* 正常三 hook 恢复、worker/scheduler/KV 排空、无 live request 被取消、退出码0、
  无 terminal traceback、owned process group为空、保存的前后4卡 idle收据通过。

这是依据旧失分选择的后验24题诊断。`full_frozen_plan_passed=null`。
原完整 native controls各9个截断的事实未改变；没有改预算或完整性条件。
不涉及当前 main、安装包、性能或任意任务质量资格化。

文件：`audit_v1.py`、`summary.json`、`rows.json`、`source-identities.json`、
`attempt-01.log` 与 `SHA256SUMS`。summary中的 producer hash固定本次执行的
审计程序；这些产物不修改源运行或其冻结包。
