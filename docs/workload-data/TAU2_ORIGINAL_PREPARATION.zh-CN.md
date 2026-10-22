# 规则约束业务办理：官方 tau2 原生验证准备

状态：2026-10-22，前三条原生执行及评分完成；统一引用交接的完整复验进行中，类别PR尚未提交。

最新更新：003已完成原始三任务，正常user_stop，官方环境reward为1、1、0；[脱敏回执](validation/tau2-telecom-original-npu-20261022.json)。前文准备记录为历史过程，不覆盖失败尝试。最终公开脚本的无模型preflight亦已在服务器通过，已安装官方源码及原始数据与固定归档一致。原生任务引用消费接口和按类PR尚未闭环，不勾类别交付完成。

## 固定来源与范围

来源 ID 沿用 `TAUBENCH`，上游实际为 `sierra-research/tau2-bench`。本轮选择原始文本任务发行标签 `v0.2.0`，提交
`f8de30c298689cbe0117d76a378e7315a17e5bd8`；该标签的 pyproject 包版本为 `0.2.1-dev`，不能仅按包版本认定来源。

上游 main 已演进到其他文本/多语言/语音协议，不将 main 与此发行版任务混合。旧机器中可编辑安装指向失效 QBI 项目路径的环境不复用。安装、源码和输入校验单列，不能因为
`tau2 --help` 成功就勾任务完成。

## 预定验证

当前数据包新增 `provider.tau2_task_references` 和 `workload-data native-tasks`：只返回官方固定版本、task
set、原始任务ID及文件/行号/SHA引用，不复制用户场景或评分答案。服务器新wheel已安装回查同三条真实任务，见[消费接口记录](validation/tau2-native-consumer-20261022.json)；这一步模型调用为0。

运行入口 `scripts/run_tau2_original_telecom.py` 支持 `--task-references`，会重新核对来源、版本、task
set、ID、文件SHA和行号，再交官方CLI。还会校验官方归档、安装源码和原始数据一致性。004已通过该引用入口启动同三条任务进行链路复验；未完成前不把接口单测当实际运行证明，也不把复验计成新任务。

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
- [ ] 独立环境已建并可导入tau2；依赖锁及完整任务验证尚待完成，不改全局模型服务环境。
- [ ] 核对 telecom 原始 tasks，执行前固定原顺序前三个任务及 ID；不按答案或能力结果选题。
- [ ] 正常 assistant/user 双方交互；两方模型都绑定已有 NPU 本地服务，不使用 solo114 替代。
- [ ] 核查全部评分路径，包括自然语言断言是否调用额外模型；存在商用默认模型时，明确本地绑定或阻塞，绝不静默外发。
- [ ] 原环境工具执行、终态及适用官方评分保存；模型失败与 harness 错误分开。
- [ ] 取得来源级原始任务引用、准备/消费回查及脱敏回执。

官方 CLI 支持 `--agent-llm`、`--agent-llm-args`、`--user-llm`、`--user-llm-args`、`--task-ids`。底层使用 LiteLLM
completion 并传递模型参数；具体固定模型和 `api_base` 需在新环境中验证，不能只因支持 OpenAI API 就宣称能跑通工具循环。

已有 Qwen3.5-27B 是双卡可部署备选，不是官方参考模型等价替代；双角色同服务的相关误差必须披露。使用原始工具/规则，不改题或补答案。原始任务、用户私有说明、对话与答案不放公开 Git/CI。

## 当前进展

官方 GitHub 固定标签与源码接口已读取。服务器浅克隆最终因连接 GitHub 超时失败，终态已确认；没有把下载等待当成任务执行。

已从官方固定提交取得源码ZIP并传到服务器，两端完整SHA256均为
`65f1d59b3783bfb38664142fb166aa40b369e4b4083170723059d8f1883ccd64`。新目录解包，创建独立
`env-tau2-v020`，官方包依赖安装已启动，尚未确认完成。原生任务、评分和端到端消费均未完成。

源码确认自然语言断言评分具有默认商用模型。预定telecom前三条的reward_basis为ENV_ASSERTION，但必须继续核对执行时实际evaluation_type，不能仅凭任务字段断言绝无额外模型调用。该状态只证明准备动作，不是benchmark验证结果。

已核对原生 `run_domain` 固定采用 `EvaluationType.ALL`，并非 `ALL_WITH_NL_ASSERTIONS`；该版本只有后者调用自然语言断言模型。按官方
`telecom_full` loader 预选前三条，不按 tasks.json 文件显示顺序强制替换 loader
顺序。两角色均使用本地NPU服务，completion包装只拒绝非批准模型/端点，不改工具或任务内容。

运行001因缺MCP导入失败；补依赖后的002因最新版LiteLLM/MCP接口不兼容失败。两次失败均保留。独立环境将LiteLLM固定为官方要求范围内的 `1.65.0`
后启动003；是否完成及评分仍待核验，不能把启动当通过。

官方依据：[固定发行源码](https://github.com/sierra-research/tau2-bench/tree/f8de30c298689cbe0117d76a378e7315a17e5bd8)、[CLI](https://github.com/sierra-research/tau2-bench/blob/f8de30c298689cbe0117d76a378e7315a17e5bd8/src/tau2/cli.py)、[模型接口](https://github.com/sierra-research/tau2-bench/blob/f8de30c298689cbe0117d76a378e7315a17e5bd8/src/tau2/utils/llm_utils.py)。
