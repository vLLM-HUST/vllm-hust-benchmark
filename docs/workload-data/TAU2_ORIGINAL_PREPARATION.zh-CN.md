# 规则约束业务办理：官方 tau2 原生验证准备

状态：登记telecom_full原生接入可用。复用旧ENV三题，新选ENV_ASSERTION＋ACTION原三题已完成官方CLI执行及完整RewardInfo复算，均user_stop、reward
1；合计六条唯一任务。固定依赖干净安装及研究侧全2285项引用交接已核验。不是2285题全量模型执行，也不代表其他domain可用。[具名范围交付回执](validation/tau2-native-delivery-20261024.json)。原PR
#405为当时交付记录，后续更新进入当前Draft PR #432，不合并。

003原始三任务及004引用交接复验均正常user_stop，官方环境reward均为1、1、0；[首次回执](validation/tau2-telecom-original-npu-20261022.json)、[引用交接回执](validation/tau2-reference-handoff-npu-20261022.json)。两批是同三条，不能计为六个独立任务。研究端新接口消费与最终固定CI
wheel接通另见下方新回执；可以勾登记telecom_full接入，不由此宣布整个workload类别、tau2全部domain或2285题模型运行完成。

## 固定来源与范围

来源 ID 沿用 `TAUBENCH`，上游实际为 `sierra-research/tau2-bench`。本轮选择原始文本任务发行标签 `v0.2.0`，提交
`f8de30c298689cbe0117d76a378e7315a17e5bd8`；该标签的 pyproject 包版本为 `0.2.1-dev`，不能仅按包版本认定来源。

上游 main 已演进到其他文本/多语言/语音协议，不将 main 与此发行版任务混合。旧机器中可编辑安装指向失效 QBI 项目路径的环境不复用。安装、源码和输入校验单列，不能因为
`tau2 --help` 成功就勾任务完成。

## 使用及逐项验证

当前数据包新增 `provider.tau2_task_references` 和 `workload-data native-tasks`：只返回官方固定版本、task
set、原始任务ID及文件/行号/SHA引用，不复制用户场景或评分答案。服务器新wheel已安装回查同三条真实任务，见[消费接口记录](validation/tau2-native-consumer-20261022.json)；这一步模型调用为0。

运行入口 `scripts/run_tau2_original_telecom.py` 支持 `--task-references`，会重新核对来源、版本、task
set、ID、文件SHA和行号，再交官方CLI。还会校验官方归档、安装源码和原始数据一致性。004已通过该引用入口完成同三条任务的链路复验，不计为新任务。回执记录实际执行脚本及私有结果SHA，后续仅注释/格式修改也不覆盖原执行指纹。

复现入口（先按授权取得固定官方归档及解包数据，安装本批固定wheel）：

```sh
workload-data native-tasks --official-archive /authorized/tau2.zip \
  --data-root /authorized/tau2/data --task-ids ID1 ID2 ID3 > native-refs.json
python scripts/run_tau2_original_telecom.py \
  --official-archive /authorized/tau2.zip --data-root /authorized/tau2/data \
  --task-references native-refs.json --output /private/fresh-run \
  --endpoint http://127.0.0.1:19466/v1 --model workload-qwen35-27b-128k-nonthinking
```

默认仍为官方telecom_full
loader预选的前三条。指定其他原始任务时，执行器同步添加可重复的`--task-id ID`，或使用`--full`选择完整任务集；全量与指定ID互斥。执行器始终保留官方loader顺序，引用列表须与该顺序一致，未知/重复ID拒绝，不构造替代任务。`--preflight-only`只验证源契约，不运行模型。官方tau2环境需安装
`litellm==1.65.0`，助手与用户共用已有本地NPU服务。原始完整任务/答案不由本项目打包分发。

- [x] 取得官方固定提交归档，两端SHA一致；不引入 QBI overlay。
- [x] 独立环境安装且完成三任务，不改全局模型服务环境；关键版本见回执。
- [x] 核对 telecom_full 原始 tasks，执行前固定loader前三个任务及ID，不按结果选题。
- [x] 正常 assistant/user 双方交互，两方模型均绑定已有本地NPU，不以solo替代。
- [x] 核对本发行版原生ALL评分，不走NL断言模型路径；模型端点白名单拒绝默认外发。
- [x] 原环境工具执行、正常终态、官方环境评分保存，第三题0分保留。
- [x] 原始任务引用准备及官方执行入口交接均已复验。
- [x] 固定CI数据包32份文件与工件一致；研究消费包的prepare及verify接口对全部2285项原生引用重验，原任务ID、顺序、文件位置和来源摘要与既有准备结果一致。仅数据/关系接通，不发模型请求，也不是2285题原生执行或研究观测验收。[研究交接回执](validation/tau2-research-handoff-20261024.json)。
- [x] 新增ACTION分支三任务使用原官方CLI及双角色，保留200步、10错误、一次trial、ALL评分；任务与原loader对象逐字段一致，保存的完整RewardInfo经原评分器复算一致。无新评分模型调用，不重跑旧ENV三题。[新分支原生审计](validation/tau2-action-native-audit-20261024.json)。
- [x] 独立Python3.12.13客户端从零安装125个固定依赖及原源码，pip
  check通过，89份原Python文件与官方归档逐字节一致；在新客户端对保存的三条新原生轨迹复评分，完整RewardInfo仍相等，无模型重跑。不是在新客户端重新执行三任务，也不宣称模型服务从零部署或构建隔离环境完全锁定。[干净安装及重评分](validation/tau2-clean-install-20261024.json)。
- [ ] 其他任务/domain与全2285题模型执行；不由接入完成推定通过。

官方 CLI 支持 `--agent-llm`、`--agent-llm-args`、`--user-llm`、`--user-llm-args`、`--task-ids`。底层使用 LiteLLM
completion 并传递模型参数；本批具名模型和 `api_base` 已实际验证工具循环，不扩称其他模型/模式可用。

已有 Qwen3.5-27B 是双卡可部署备选，不是官方参考模型等价替代；双角色同服务的相关误差必须披露。使用原始工具/规则，不改题或补答案。原始任务、用户私有说明、对话与答案不放公开 Git/CI。

## 2026-10-24可复用入口与环境补齐

- [x] 全2285个原telecom任务的源码/数据字节及加载预检完成，官方任务引用全量准备；完整ID索引只在服务器私有保存。
- [x] 补齐显式原task ID及全量入口，指定的前三个原ACTION类任务通过公共引用→官方入口预检，模型调用0。
- [x] 修复旧入口只接受ENV_ASSERTION的限制：原任务有2253个ENV_ASSERTION及32个ENV_ASSERTION+ACTION，均由原版ALL评分覆盖，不跳过32个任务、不改评分、不引入外部自然语言judge。
- [x] 现有独立Python3.12.13/Linux
  aarch64环境的125个包与[依赖快照](../../configs/workload-native/tau2-v020-py312-dependencies.txt)逐项相等，pip
  check通过。[CPU回执](validation/tau2-readiness-cpu-20261024.json)。
- [x] 新环境从零安装及修改后入口的新ACTION三任务原生交互/评分完成；复用旧三题，不将其冒充新执行器实跑。全2285题模型运行及其他domain仍未验证。

依赖文件是实测环境快照（含辅助工具），不是最小依赖或跨平台锁文件。tau2须从上述固定归档单独安装，不能仅按`0.2.1-dev`版本号从PyPI换包；原构建系统为pdm-backend，构建隔离依赖未完全锁定，不宣称wheel字节可复现；目标aarch64平台的新安装、125项运行版本及89份源码已实测一致。使用独立虚拟环境，不升级推理服务环境：

```sh
python3.12 -m venv /private/tau2-runtime
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/tau2-runtime/bin/python -m pip install \
  -r configs/workload-native/tau2-v020-py312-dependencies.txt
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/tau2-runtime/bin/python -m pip install \
  --no-deps /authorized/tau2-fixed-source
env -u PYTHONPATH -u PYTHONHOME -u LD_PRELOAD /private/tau2-runtime/bin/python -m pip check
```

全量预检可在前述原执行命令中加`--full --preflight-only`；它验证源码、数据、选择及评分类型，不发起角色交互、工具执行或评分，不计入已完成题数。正式运行仍保留200步、10错误上限、一次trial、两个角色及原ALL评分，不能为提速削减流程。旧实例曾出现主机锁页分配错误；新实例已按原模型配置恢复。在释放的NPU窗口中，已使用此前按loader顺序固定的前三个ENV_ASSERTION＋ACTION任务完成当前选择/引用入口执行，不替换结果差的任务，官方完整重评分已审计。模型及完整启动参数参见[已核验双NPU部署](BBH_ORIGINAL_INPUTS.zh-CN.md)。预检、引用接通与真实任务分开计数，不把旧三题重复计为新任务。

## 历史准备及失败记录

官方 GitHub 固定标签与源码接口已读取。服务器浅克隆最终因连接 GitHub 超时失败，终态已确认；没有把下载等待当成任务执行。

已从官方固定提交取得源码ZIP并传到服务器，两端完整SHA256均为
`65f1d59b3783bfb38664142fb166aa40b369e4b4083170723059d8f1883ccd64`。新目录解包，创建独立
`env-tau2-v020`。最初依赖安装与接口准备不计任务完成，后续完成状态见本页顶部及回执。

源码确认自然语言断言评分具有默认商用模型。所选三条reward_basis为ENV_ASSERTION，并进一步核对执行时实际evaluation_type；不能仅凭任务字段推断模型调用范围。

已核对原生 `run_domain` 固定采用 `EvaluationType.ALL`，并非 `ALL_WITH_NL_ASSERTIONS`；该版本只有后者调用自然语言断言模型。按官方
`telecom_full` loader 预选前三条，不按 tasks.json 文件显示顺序强制替换 loader
顺序。两角色均使用本地NPU服务，completion包装只拒绝非批准模型/端点，不改工具或任务内容。

运行001因缺MCP导入失败；补依赖后的002因最新版LiteLLM/MCP接口不兼容失败。两次失败均保留。独立环境将LiteLLM固定为官方要求范围内的 `1.65.0`
后完成003及004；失败仍保留，未用成功记录覆盖。

官方依据：[固定发行源码](https://github.com/sierra-research/tau2-bench/tree/f8de30c298689cbe0117d76a378e7315a17e5bd8)、[CLI](https://github.com/sierra-research/tau2-bench/blob/f8de30c298689cbe0117d76a378e7315a17e5bd8/src/tau2/cli.py)、[模型接口](https://github.com/sierra-research/tau2-bench/blob/f8de30c298689cbe0117d76a378e7315a17e5bd8/src/tau2/utils/llm_utils.py)。
