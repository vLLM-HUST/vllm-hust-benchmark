# 企业提供数据：请求回放与响应完整性验证

本入口不把企业数据集本身称为独立 benchmark，不把格式核验称为语义正确率。原始数据、模型响应和私有日志不分发，使用者须获授权自行取得文件。

三份 v1（会话1000、代码判断1000、JSON500）保留封装中的原始 API 请求体；四份 v2（各5000条）和超长文本4000条保留提供方原有 messages 与 API
字段投影，不自行生成、删减或重写内容。后五份属于企业提供的构造/生成数据，不冒充纯生产日志。全部八份文件已在服务器核对完整 SHA。

## 本次实际验证

每源执行前固定前三条，共24条，全部完成 NPU 回放、终止及响应完整性核验；JSON来源按原请求的 response_format/schema 检查。服务端 prompt token 数与同
tokenizer/template 预检逐条一致，24条未截断输入。对保存的本次真实响应使用最终固定校验器复核，不把复核算成新增模型任务。

模型/服务：双卡 Ascend 910B2，Qwen3.5-27B，BF16、TP=2、graph、text-only，131072上下文，服务端默认
enable_thinking=false。11份权重及9份配置/tokenizer文件与发布方固定提交身份匹配，见[可部署工件](validation/qwen35-27b-deployable-artifact-20261022.json)。这是适配验证配置，不是正式
leaderboard 基线或性能比较。

原消息、stream、response_format、显式输出预算均保留，只显式绑定实际服务模型。未提供 max_tokens
时继续省略、采用模型/服务默认值；metadata.output_tokens_target 不是 API 参数，**不会自动补为
max_tokens**。超长文本前三条实际输入约7.7万～8.2万token。

初始16K/思考配置下的[既往尝试](validation/enterprise-v1-attempts-20261022.json)保留。新配置同时改变上下文及思考模式，不用于归因任何引擎性能收益，不覆盖或删除失败历史。

## 判定边界

- 流式必须接收 `[DONE]` 与有效 finish_reason，不把断连当完成。
- 本抽样入口要求完整 stop、非空输出及声明格式通过才标完整性通过。length 是正常的 API 预算终止，单列“质量未验证”，不直接称服务故障；也不自动当完整任务通过。
- json_object 必须是严格 JSON 对象；json_schema 按原 schema、固定 jsonschema 4.26.0检查，不创造 schema 或
  gold。内部引用可用，不静默获取外部引用；缺引用、缺校验器、未知 dialect 明确不能验证。
- 普通文本仅核验终止及非空，不暗中使用 LLM judge。八份数据没有独立语义 gold/官方任务评分器，因此 quality_claim=false，不能报告任务准确率。
- prefix/group/reuse 等标签不证明 token 前缀相同、KV命中、缓存收益或真实依赖；本批没有做这些科研分析或性能测量。需要工具环境闭环的任务不得用单次回放替代。

## 使用

### 八个来源的参数

下表只列接入参数，不分发文件；`--data-root`指向使用者获授权取得的对应外部目录。文件版本和SHA须与固定目录/原回执一致，不能用同名不同内容文件替代。

| source-id                  | revision           | files                           |
| -------------------------- | ------------------ | ------------------------------- |
| PAIO-CHAT-1000             | v1                 | normal_chat_1000.jsonl          |
| PAIO-CODE-EVAL-1000        | v1                 | code_eval_ttft_1000.jsonl       |
| PAIO-JSON-500              | v1                 | json_requests_500.jsonl         |
| PAIO-LONG-PREFILL-5000     | v2                 | long_prefill.jsonl              |
| PAIO-PREFIX-SHARED-5000    | v2                 | prefix_shared.jsonl             |
| PAIO-REUSE-CONV-5000       | v2                 | reuse_conversation.jsonl        |
| PAIO-SEMANTIC-SIMILAR-5000 | v2                 | semantic_similar.jsonl          |
| PAIO-LONGTEXT-4000         | long-text-delivery | api_long_text_merged_4000.jsonl |

完整服务参数见[固定服务配置](profiles/source-validation-qwen35-27b-128k-nonthinking.json)，MODEL_DIR替换为已校验的实际模型目录。该配置不含数据、不自动启动模型；现有服务满足它时不必重复加载。尤其在服务端设置
`--default-chat-template-kwargs '{"enable_thinking":false}'`，而不是改写原请求。编译/HF缓存放到本任务数据卷，TMPDIR使用短路径，不占满root分区。

```sh
python -m pip install '.[workload-validation]' './workload-preparation[data]'
python scripts/run_source_request_samples.py --source-id PAIO-CODE-EVAL-1000 --data-root /private/company-v1 --files code_eval_ttft_1000.jsonl --revision v1 --model workload-qwen35-27b-128k-nonthinking --endpoint http://127.0.0.1:19466/v1/chat/completions --service-profile text-only-128k-nonthinking/v1 --timeout 600 --output /private/runs/code-new
```

profile 须与实际服务配置一致。数据与输出分离，输出为新的私有目录。exit 0表示三条请求与适用完整性核验完成，exit
2表示有未通过项；HTTP200不等于任务正确。timeout是socket读取超时，不是性能SLO。

各源回执见[可勾选清单](DATASET_CHECKLIST.zh-CN.md)。公开记录仅保留来源/版本/SHA、样本标识、模型/环境配置、终态和核验状态，不包含企业正文、schema内容、答案或凭据。

### 按原记录用于后续批量实验

前三条入口用于固定集成验证，不限制数据能力只能读取三条。完整原文件通过`workload-data pipeline bind`绑定，再用`pipeline run`准备原记录索引；配置省略`limit`即准备完整文件，原数据不被复制进索引。使用`prepared_rows`枚举记录身份、`resolve_input`取回原记录，调用`vllm_hust_workload_data.execution.replay_request`回放，再以`vllm_hust_benchmark.workload_response_validation.check_response`核验原声明格式；这套公开API支持原任意记录，不需要研究仓或观测模块。单条命令`workload-data replay --database PRIVATE_DATABASE --record-id ORIGINAL_RECORD_ID --data-root AUTHORIZED_ROOT --endpoint LOCAL_CHAT_COMPLETIONS_URL --model VERIFIED_MODEL_ALIAS --output FRESH_PRIVATE_RESPONSE`复用同一API，但自身只保存响应、不宣称任务已评分；需另外执行上述完整性校验。

批量实验须自行明确全量/固定子集和模型环境，不能改写消息、把metadata输出目标补成API预算，或把需工具/环境的任务冒充单次请求。原请求及响应/完整索引均保存在外部授权目录；运行失败保留，不换题。本文已验证范围是每源原前三条，不声称全文件模型执行或任务准确率。

### 固定工件交付复核（2026-10-24）

当前服务器已只读确认原八份文件、24条原响应和私有索引均可访问；原文件SHA及响应束SHA与历史NPU回执逐源一致。固定3f397e69的CI数据wheel独立安装后，24条原输入/outcome往返与旧索引一致，全部保存响应重新通过原格式及输入token核验；0新增模型调用。当前公开回放脚本及响应校验器的规范LF摘要与实际NPU执行时一致，未为凑通过改原请求或校验器。见[八来源交付复核](validation/enterprise-fixed-ci-handoff-20261024.json)。

依此按私有请求语料的固定抽样链路可用范围打勾；原企业正文、schema内容、答案和原响应不公开。旧SSH不可达是历史访问状态，不再作为当前缺口；旧失败尝试和全部原回执仍保留。研究侧原八源消费证据另见[固定消费回执](validation/enterprise-private-source-consumer-20261024.json)，不冒称本次升级了研究依赖锁或开展了新观测/性能实验。
