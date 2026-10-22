# 规则约束业务办理：官方 tau2 原生验证准备

状态：2026-10-22，前三条原生执行及评分、统一引用交接复验均已完成；Draft PR #405，未合并。

003原始三任务及004引用交接复验均正常user_stop，官方环境reward均为1、1、0；[首次回执](validation/tau2-telecom-original-npu-20261022.json)、[引用交接回执](validation/tau2-reference-handoff-npu-20261022.json)。两批是同三条，不能计为六个独立任务。研究端新接口消费与最终固定CI
wheel锁尚待接通，不勾整个类别交付或全数据集完成。

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

ID1–3须为官方telecom_full loader预选的前三条，不是任意构造ID。`--preflight-only`只验证源契约，不运行模型。官方tau2环境需安装
`litellm==1.65.0`，助手与用户共用已有本地NPU服务。原始完整任务/答案不由本项目打包分发。

- [x] 取得官方固定提交归档，两端SHA一致；不引入 QBI overlay。
- [x] 独立环境安装且完成三任务，不改全局模型服务环境；关键版本见回执。
- [x] 核对 telecom_full 原始 tasks，执行前固定loader前三个任务及ID，不按结果选题。
- [x] 正常 assistant/user 双方交互，两方模型均绑定已有本地NPU，不以solo替代。
- [x] 核对本发行版原生ALL评分，不走NL断言模型路径；模型端点白名单拒绝默认外发。
- [x] 原环境工具执行、正常终态、官方环境评分保存，第三题0分保留。
- [x] 原始任务引用准备及官方执行入口交接均已复验。
- [ ] 全依赖锁及最终CI wheel的研究端消费联调；其他任务/domain仍未验证。

官方 CLI 支持 `--agent-llm`、`--agent-llm-args`、`--user-llm`、`--user-llm-args`、`--task-ids`。底层使用 LiteLLM
completion 并传递模型参数；本批具名模型和 `api_base` 已实际验证工具循环，不扩称其他模型/模式可用。

已有 Qwen3.5-27B 是双卡可部署备选，不是官方参考模型等价替代；双角色同服务的相关误差必须披露。使用原始工具/规则，不改题或补答案。原始任务、用户私有说明、对话与答案不放公开 Git/CI。

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
