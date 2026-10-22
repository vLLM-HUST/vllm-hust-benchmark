# OASST1 原始历史续接视图

## 数据与能力边界

来源为 OpenAssistant/oasst1，固定 revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`。
validation 原始成员 `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` 共 4401 行，SHA-256：
`24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416`。

既有通用读取器保留单条历史消息和 parent_id，不假定发布文件顺序就是会话顺序。新增显式辅助函数仅恢复指定目标之前的完整原始祖先链：不拼接不同分支，不补造缺失消息，不修改正文，不把目标 assistant 回复当作输入。

这是历史上下文条件下的续接，不是原始 API 请求回放，也不是与真实用户重新交互的完整 episode。历史回复不是独立语义正确性标准；此接口不评分，不分析 KV 或计算科研特征。

## 使用

先按发布方条件取得原始文件，在仓库外保存；安装本仓数据组件的 `[data]` 可选依赖。此函数不隐式下载数据。

```python
import pyarrow.parquet as pq
from vllm_hust_workload_data import oasst_history_prefix

rows = pq.read_table("/data/oasst1/validation.parquet").to_pylist()
view = oasst_history_prefix(rows, "c8dc7c16-e493-4078-bdc7-368b24476ca9")
messages = view["messages"]  # 仅此字段作为历史消息输入
# view["reference"] 是分离的历史回复，不应传入模型或当作质量 gold。
```

调用者负责核验文件 revision、字节 SHA 和选样；函数不以语言或长度偷偷筛选。目标必须是 assistant，路径必须从同一原始根开始、角色交替、没有缺失/重复 ID 或环，并且路径各行明确 deleted=false、synthetic=false。正文保留原样；角色只作 prompter→user、assistant→assistant 映射。synthetic=false 是发布字段，不是对所有文本生成来源的额外保证。

该显式视图尚未写入通用 pipeline/source-index，也尚未得到研究消费者对应视图的联调回执。直接调用函数可用，不声称通用使用链路已完成。

## 已验证范围

在既有双 NPU Qwen3.5-27B 服务上，按执行前规则选取发布顺序中前三个不同 parent 的续接点：英文 assistant 目标，完整交替路径至少四节点，路径各行 deleted=false/synthetic=false。符合规则的 distinct parent 共 288 个；本次只执行三个，不代表全量 4401 行或 288 个点。

| 原始目标行（零起始） | 目标 message_id | 输入 token | 输出 token | 终止 |
| --- | --- | --- | --- | --- |
| 23 | c8dc7c16-e493-4078-bdc7-368b24476ca9 | 310 | 739 | stop |
| 28 | 48ac2156-f823-4e97-81ab-a66354549f59 | 85 | 493 | stop |
| 68 | 3f509e81-40ff-47dc-a677-4cb4a1623d7b | 506 | 714 | stop |

三点来自两个原始树，每个输入三条历史消息；不能说成三个独立完整会话。temperature=0，max_tokens=2048，非思考，未裁剪历史；原文件 SHA 未改变。原运行私有回执 SHA-256 为 `ae47a321e7c60b474d451c277911c07c6496e3261aef0d2eb6375ea711cf25ca`。原文、历史回复和模型回答不提交。

运行之后在服务器用新增辅助函数重新读取同一原始文件，三个生成输入消息的哈希全部匹配先前实际运行记录；该核对没有新增模型调用。候选辅助模块 SHA-256 为 `f8dd4a343f77c30c308a4754e9a13591ac4d917ddb5af48005a13b9f94347960`。

结论仅为三点历史续接生成正常终止和输入完整性成立；没有官方质量评分、性能结论、实时会话成功率或 KV 复用收益结论。源版本与模型部署信息沿用本批既有运行记录，不将此辅助函数的事后输入匹配宣称为安装包端到端验证。

CPU smoke 自动运行 `workload-preparation/tests`，覆盖本接口的乱序还原、目标隔离、原文保持、缺失链接、重复 ID、环、跨树、角色错误及不可用消息拒绝。夹具是单元测试数据，不是真实执行证据。

本地源码组件回归 138 项通过；另构建候选 wheel，在独立 venv（共享本机测试依赖）安装，使用 Python `-I` 确认从 site-packages 导入而非源码路径，完整组件回归同样 138 项通过。该本地候选 wheel SHA-256 为 `9a194bd2acfb6d2ce9a158e5c03d0b4a2b913e9296ca59ba1407aad2f0fcc32a`；它不是已发布 CI 工件，也不是研究端当前锁定的 wheel。
