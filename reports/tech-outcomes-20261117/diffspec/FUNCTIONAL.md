# #11 合并版：两真实请求＋受控 dummy 功能回归

## 版本、请求与检查方式

源码固定为 `48b182803f2b1bc397d61fd8f5a50c8716f2c4ed`，31 个文件均对照公开 Git blob；四个 worker 加载的 runtime SHA256
均与该版本一致。Core `762f85b3`、Ascend `4e57439e` 加冻结本地补丁，20 个 backend 文件摘要全部与历史实测一致。见
[环境说明](ENVIRONMENT.md)。

两条真实请求使用历史固定输入的 4096、4352 token 头尾裁剪视图，各生成 256 token；不是标准答题准确率测试。逐请求摘要、启动配置和完整日志在
`functional/attempt01` 与 `functional/attempt02`。原始输入 token 数组和完整输出位于功能 Release 附件，Git 不重复存放，摘要和原始
SHA256 见 `originals-manifest.json`。

第一轮自然 metadata 为两行，未生成第三个 dummy。第二轮保留两条真实请求、原生 valid counts、真实 query/LSE/KV 数值，在检索入口增加一行显式 dummy
query/LSE/块表及元数据；调用未修改的 #11 实现后恢复快照引用。该构造与已提交的 `graph_padding=True` 回归方向一致，是受控 padding
边界测试，不声称服务调度器自然产生了该行。

观测脚本 [trace_worker.py](functional/attempt02/trace_worker.py) 位于测试目录，未修改插件源码、模型权重、真实 query
值或原生验证算法。它记录实际 kernel 收到的两条前缀长度、最后已提交 query、一致性摘要和所选 token 边界。

## 结果

- 聚焦已提交用例：36 passed，33 deselected；命令与 exit 原件保留。
- 受控组合：21 个检索周期，每个均有两个有效请求、三行 attention metadata，实际 kernel 和所选缓存只含两个有效请求。
- 21 个周期均出现草稿拒绝后缀；kernel 序列长度等于已提交前缀，所选 token 全部小于各自已提交边界。
- 第二轮总计 24 个检索周期，四 rank 的 chunk 列表及 token 摘要逐周期一致。不要把 4 个 rank 副本累计成 96 个独立周期。
- 两请求均完成 256 token，is_corrupted=false、抢占 0；不是质量准确率结论。

例如 cycle 4：两请求 valid counts 为 `[2,5]`，query ends 为 `[5,10,11]`，metadata seq lens 为
`[4118,4366,1]`；实际 kernel 长度为 `[4115,4366]`，选中 token 最大值为 `[4114,4365]`。第三行未参与检索，第一请求拒绝后缀被排除。

## 材料入口

本次记录用于指定两请求＋dummy 的功能边界回归，不外推完整输出等价或完整准确率。

完整原件见
[第二轮日志](functional/attempt02/model.log)、[四 rank 记录](functional/attempt02/artifacts/traces/)、[汇总](functional/attempt02/functional-summary.json)。关闭后的
[NPU 状态](functional/attempt02/npu-after.txt) 为四卡空闲。
