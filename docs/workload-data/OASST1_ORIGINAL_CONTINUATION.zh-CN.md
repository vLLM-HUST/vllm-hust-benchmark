# OASST1 原始历史续接视图

当前来源级交付已收口，见[交付回执](validation/oasst-corpus-delivery-20261024.json)。训练集[prepare/resolve回查](validation/oasst-train-pipeline-20261024.json)三条完整历史与实际模型请求一致、参考回复未进入SQLite，0新增模型调用。下文保留分阶段历史证据，较早的“待核验”表述不代表当前状态；完成范围仅为原始语料和已声明的六点历史续接，不是官方质量评分或全量推理。

## 当前原始成员取得状态

2026-10-24服务器已取得同一固定revision的两个原始Parquet成员：train 84437行、validation
4401行，完整文件SHA均与登记的发布资产一致。validation复用原副本硬链接；train直连未完成后，通过同revision镜像取得并强校验，不替换数据版本。见[成员核验回执](validation/oasst-members-20261024.json)。这证明原资产取得及文件身份，不证明训练集已做模型生成或完整会话适配验证；公开续接入口当前固定validation身份，train的原始消息读取与相应实验入口仍需分别核验。

## 数据与能力边界

来源为 OpenAssistant/oasst1，固定 revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`。 validation 原始成员
`data/validation-00000-of-00001-134b8fd0c89408b6.parquet` 共 4401 行，SHA-256：
`24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416`。

既有通用读取器保留单条历史消息和 parent_id，不假定发布文件顺序就是会话顺序。新增显式辅助函数仅恢复指定目标之前的完整原始祖先链：不拼接不同分支，不补造缺失消息，不修改正文，不把目标
assistant 回复当作输入。

这是历史上下文条件下的续接，不是原始 API 请求回放，也不是与真实用户重新交互的完整 episode。历史回复不是独立语义正确性标准；此接口不评分，不分析 KV 或计算科研特征。

## 使用

运行入口支持`--split validation`（默认）与`--split train`，分别强制核对对应原文件SHA和行数；训练集必须显式重复指定`--target-message-id`，不会误用验证集三点。两个split的文件不能互换。训练集历史续接沿用完整祖先、目标回复隔离及相同生成/响应检查，不把原始会话语料包装成官方质量评分benchmark。

训练集新增三目标已在现有双NPU服务生成并完成[原文及响应审计](validation/oasst-native-train-audit-20261024.json)，按发布顺序预选英文、不同parent、至少三条完整历史输入的首三目标，未重跑验证集三点、未换失败题。固定CI数据包对train/validation合计88838行逐行核对原文本、角色、目标ID、parent/tree和输出参考标注；三个新增原历史与保存请求一致。原始语料没有官方语义正确率评分器，这里仅生成和关系/响应完整性成立，不是实时交互、独立质量认证或KV性能结论。训练集prepare/resolve交接仍需补核，尚不提升为来源级完成。

先按发布方条件取得原始文件，在仓库外保存；安装本仓数据组件的 `[data]` 可选依赖。此函数不隐式下载数据。

```python
import pyarrow.parquet as pq
from vllm_hust_workload_data import oasst_history_prefix

rows = pq.read_table("/data/oasst1/validation.parquet").to_pylist()
view = oasst_history_prefix(rows, "c8dc7c16-e493-4078-bdc7-368b24476ca9")
messages = view["messages"]  # 仅此字段作为历史消息输入
# view["reference"] 是分离的历史回复，不应传入模型或当作质量 gold。
```

调用者负责核验文件 revision、字节 SHA 和选样；函数不以语言或长度偷偷筛选。目标必须是 assistant，路径必须从同一原始根开始、角色交替、没有缺失/重复 ID 或环，并且路径各行明确
deleted=false、synthetic=false。正文保留原样；角色只作 prompter→user、assistant→assistant 映射。synthetic=false
是发布字段，不是对所有文本生成来源的额外保证。

通用 pipeline 配置可以显式指定 `history_targets` 原始 assistant message_id 列表。仅支持 OASST1、单个完整原始消息树文件，不与自定义
mapping 混用；默认不指定时仍输出原始单消息。`limit`
只进一步限制所选目标，不会裁剪祖先链。索引只保存原目标行、文件身份和视图选择，不保存历史正文或回复；回查时校验同一源文件并重新恢复完整祖先链。

```json
{"sources":[{"source_id":"OASST1","data_root":"/data/oasst1","manifest":"/data/bindings/oasst1.json","history_targets":["c8dc7c16-e493-4078-bdc7-368b24476ca9"]}]}
```

此通用视图的夹具准备/索引/回查已通过；随后在服务器独立 OASST 环境安装候选 wheel，使用 `-I` 确认从 site-packages 导入，真实文件经过
bind_source→pipeline→prepared_rows→resolve_input，三条还原输入哈希均匹配既有模型运行。历史回复未进入输入或 SQLite；0
新模型调用。研究消费者对应视图联调仍待补验。

索引身份额外固定 `input_ref.view=original-parent-history/v1`，使同一原行的历史消息与续接视图产生不同 record_id；默认旧视图 ID
不变。加入该身份隔离后重新构建 wheel（SHA-256
`933fbba59c4d1e31ce84734189854c750d6dddeaadc207a4ba5ed2f2448b9788`），服务器安装包三条原始输入索引回查再次通过，仍为软件补验、无新模型调用。

当前安装包验证使用的 wheel SHA-256 为 `1ebccebc7bf6e1c7b931705fb7f8a3f9850da87377a82aac1022f590b6902e20`，私有软件回执
SHA-256 为 `e97d88a0d2e40864c5c36e6cc85bfd0d993dec618c2c8d66aee84d3c7acf53a8`。这是本地候选工件，不是已发布 CI
wheel。服务器原始文件副本的 basename 与目录原成员不同，因此绑定明确报告 local snapshot 而非 publisher authenticated；另行核对文件 SHA
与固定原始成员相同，不借改名提升来源认证等级。

## 已验证范围

可复用脚本已在服务器使用安装包复跑同三点：全部stop，输入哈希与先前运行相同，仍只计三个唯一续接点。脚本 SHA-256 为
`f6ddc2eee93d2a4449c09d42fbbb5676fef315c478aff23d83700011d60acf03`；私有回执 SHA-256 为
`c2ebadba256d1d05ed0cd59d0ae7e59fefb1a0348f12ffafafac045e9dcfc098`。此复验不新增质量或性能结论。

可复用生成命令（输出目录必须不存在，所有输出留在仓库外）：

```sh
python scripts/run_oasst_original_continuation.py --data-file /data/oasst1/validation.parquet --output /data/private/oasst-run-001 --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

该入口默认下述三个目标并固定原始 validation SHA，预检完整输入 token 预算（128K、输出2048），不修改题目、不裁剪祖先链。只允许本地模型 endpoint。运行依赖数据组件
`[data]`，无额外 Agent 框架；原始请求和回答仅写入私有输出目录，不写入 Git。这是历史续接生成入口，不是官方质量评测器。

2026-10-24补齐原目标选择入口：默认仍为历史三点，可重复传入`--target-message-id ORIGINAL_ASSISTANT_ID`指定任意事前选择的原始目标列表；不是只能运行三个硬编码样本。目标顺序原样保留，重复ID直接拒绝，缺失或不合规祖先链在模型请求前拒绝，不替换失败目标。原文件SHA、完整祖先恢复、回复隔离、生成预算及响应检查不变。历史脚本摘要只对应当时实际执行版本；新增选择入口尚待真实服务器交接复核，本来源不提前勾选。

随后已完成[固定CI安装包及新增入口离线交接复核](validation/oasst-fixed-ci-handoff-20261024.json)：服务器原文件4401行及完整SHA核对，三个历史目标经安装包prepare/resolve后，与原祖先恢复及保存请求逐字一致；参考回复不进入SQLite。新增公开入口显式传入相同三目标，以保存响应替代网络传输核对三份完整请求载荷，均匹配。此步骤为离线保存证据复用，0新模型调用，不冒称新选择入口在线NPU实测或研究侧新依赖锁已接通；历史在线生成证据仍按原回执引用。来源其余原始成员及完整交付范围仍待核对，暂不勾选。

在既有双 NPU Qwen3.5-27B 服务上，按执行前规则选取发布顺序中前三个不同 parent 的续接点：英文 assistant 目标，完整交替路径至少四节点，路径各行
deleted=false/synthetic=false。符合规则的 distinct parent 共 288 个；本次只执行三个，不代表全量 4401 行或 288 个点。

| 原始目标行（零起始） | 目标 message_id                      | 输入 token | 输出 token | 终止 |
| -------------------- | ------------------------------------ | ---------- | ---------- | ---- |
| 23                   | c8dc7c16-e493-4078-bdc7-368b24476ca9 | 310        | 739        | stop |
| 28                   | 48ac2156-f823-4e97-81ab-a66354549f59 | 85         | 493        | stop |
| 68                   | 3f509e81-40ff-47dc-a677-4cb4a1623d7b | 506        | 714        | stop |

三点来自两个原始树，每个输入三条历史消息；不能说成三个独立完整会话。temperature=0，max_tokens=2048，非思考，未裁剪历史；原文件 SHA 未改变。原运行私有回执
SHA-256 为 `ae47a321e7c60b474d451c277911c07c6496e3261aef0d2eb6375ea711cf25ca`。原文、历史回复和模型回答不提交。

运行之后在服务器用新增辅助函数重新读取同一原始文件，三个生成输入消息的哈希全部匹配先前实际运行记录；该核对没有新增模型调用。候选辅助模块 SHA-256 为
`f8dd4a343f77c30c308a4754e9a13591ac4d917ddb5af48005a13b9f94347960`。

结论仅为三点历史续接生成正常终止和输入完整性成立；没有官方质量评分、性能结论、实时会话成功率或 KV
复用收益结论。源版本与模型部署信息沿用本批既有运行记录，不将此辅助函数的事后输入匹配宣称为安装包端到端验证。

CPU smoke 自动运行 `workload-preparation/tests`，覆盖本接口的乱序还原、目标隔离、原文保持、缺失链接、重复
ID、环、跨树、角色错误及不可用消息拒绝。夹具是单元测试数据，不是真实执行证据。

本地源码组件回归 138 项通过；另构建候选 wheel，在独立 venv（共享本机测试依赖）安装，使用 Python `-I` 确认从 site-packages 导入而非源码路径，完整组件回归同样
138 项通过。该本地候选 wheel SHA-256 为
`9a194bd2acfb6d2ce9a158e5c03d0b4a2b913e9296ca59ba1407aad2f0fcc32a`；它不是已发布 CI 工件，也不是研究端当前锁定的 wheel。
