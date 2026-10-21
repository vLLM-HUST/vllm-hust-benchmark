# Frontier 成绩提交指南：从跑测试到看到榜单上的点

本指南使用 [swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) 做评测，演示怎样准备服务、运行测试、整理证据、提交
PR，以及在哪里查看结果。

**本目录的 [snapshot.json](snapshot.json) 和 [evidence.json](evidence.json) 仍是已发布的 BetterScale /
Qwen3.5-35B-A3B BF16 / TP2 / C4 的历史 AgentX 成绩。** 它们只供理解配置、point 和证据的组织方式，不能作为 SWE Prefix Reuse
的报告、指标映射或 workload 模板，也不能重复上榜。下面的流程必须使用你自己测得的新 run；本次指南更新没有生成新成绩。

```text
准备推理服务 → 编译 SWE workload → 短测检查协议 → 固定窗口评测
→ 本地核对 summary/config → 整理成绩与证据 → website 数据 PR
→ 审查与合并 → 部署完成 → Frontier 查看与下载配置
```

## 1. 准备一个可以测的推理服务

先在获准使用的设备上部署模型并保留完整启动配置。`swe-prefix-reuse` 是客户端，不会下载模型、启动 engine、分配加速卡或修改服务配置。

服务端必须支持：

- `/v1/completions` 的整数 token ID 输入，而不只是 chat 接口。
- 流式响应通过 `return_token_ids=true` 返回增量 `token_ids`，同时提供 prompt-ID echo、最终 usage、`ignore_eos=true` 和
  `cache_salt`。
- 足够的真实上下文容量，覆盖 prepared workload 的每条完整会话。下面以 262,144 为配置示例，不代表每条请求都有 256K tokens。
- 与编译 workload 完全匹配的 tokenizer。客户端不提供文本重新分词的降级路径；仅通过 prompt echo 也不能证明词表身份一致。

记录模型与 tokenizer revision、engine/后端/MOD commit、权重/计算/KV 精度、总卡数、TP/PP/DP、批处理、KV/host memory、prefix
cache、graph 和 MTP 设置。engine 与 MOD 分开填写，只记录实际启用的 MOD。使用 MTP 时保留真实生成和真实 acceptance，**不要沿用历史 AgentX 的
synthetic sampler 或固定 AL2.63**。

客户端保证后续轮次接上真实输出 ID，但这不等于 engine 一定复用了缓存。需另查服务端 prefix-cache 命中指标；多副本还需核对路由亲和性。

## 2. 安装客户端并编译 workload

需要 Python 3.10+ 和本地 tokenizer，不需要在客户端安装 vLLM、torch 或加速卡驱动。以下固定到本指南核对过的工具 commit：

```bash
git clone https://github.com/vLLM-HUST/swe-prefix-reuse.git
cd swe-prefix-reuse
git checkout 6861242dbd9f17b707003191e4200b7752911d7c
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[prepare]'

swe-prefix-reuse prepare \
  --source data/open-swe-sample.json.gz \
  --tokenizer /path/to/Qwen3.5-35B-A3B \
  --max-context 262144 \
  --output prepared/qwen35.json
```

将 tokenizer 路径换成真实目录。`prepare` 只做本地编译，不向推理服务发送请求：它用 thinking-enabled
模板提取原始输入增量和每轮输出预算；运行时模型自由生成，再把实际输出 ID 接入历史，不重新渲染历史。超长会话会整条拒绝，而不是截断；检查准备结果中的拒绝记录，不能静默更换接受的会话子集。

baseline/MOD 对比必须复用**同一份 prepared 文件**。其哈希、tokenizer 指纹、编译策略和接受的会话子集都是 workload 身份的一部分。更换
tokenizer、模板或子集后，不能仍归入旧合约。

在客户端准备 `server-metadata.json`：它是你填写的 JSON 对象，不是工具自动探测的证明。至少记录以下真实信息，不能留下占位值再公开：

| 内容                | 应记录什么                                             |
| ------------------- | ------------------------------------------------------ |
| engine / 后端 / MOD | 名称、版本、commit；MOD 是否实际启用                   |
| 模型 / tokenizer    | 模型名称、固定权重 revision、tokenizer 身份与 revision |
| 精度与硬件          | 权重/计算/KV 精度、设备型号、全部参与卡数              |
| 服务参数            | TP/PP/DP、上下文、批处理、KV 预算、缓存、graph、MTP    |
| 启动命令            | 去除凭据和私有信息的完整启动命令                       |

元数据只声明服务状态，不会修改服务。API key 如有需要，通过 `OPENAI_API_KEY` 环境变量安全提供，不要写进文件或命令；endpoint 不能带凭据。仅测试已获授权的服务，不拿付费
API 试跑。

## 3. 先做短测，再跑固定窗口

将模型名、endpoint、总卡数及上下文改成实际值后执行：

```bash
swe-prefix-reuse run \
  --workload prepared/qwen35.json \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model YOUR_SERVED_MODEL_NAME \
  --server-max-context 262144 \
  --concurrency 4 --duration 60 --chips 2 \
  --server-metadata server-metadata.json \
  --output results/c4-check
```

`C4` 是 4 条顺序请求 lane，每条最多一个在途请求；上一轮完成并校验后立即发下一轮，会话结束后循环补入新会话。**不是 AgentX 的 4 棵会话树，也不回放工具/人的等待时间。**
`--chips` 是全部参与加速卡数量，多副本时不能只填 TP。

每次会话播放有新的 `cache_salt`，同一会话各轮共享它；因此保留会话内复用、排除跨会话共同前缀复用。原生 vLLM DP 可在确认服务支持 rank header 后加
`--data-parallel-size N`，令 lane 按编号固定到 rank。该参数不创建 DP 服务；带自有亲和协议的 relay 应省略它，并确认转发的
`X-Correlation-ID` 能保持会话路由。记录实际路由策略和每 rank 的负载/缓存情况。

短测通过后，可使用同一 prepared 文件做一个明确的 900 秒窗口：

```bash
swe-prefix-reuse run \
  --workload prepared/qwen35.json \
  --endpoint http://127.0.0.1:8000/v1/completions \
  --model YOUR_SERVED_MODEL_NAME \
  --server-max-context 262144 \
  --concurrency 4 --duration 900 --chips 2 \
  --server-metadata server-metadata.json \
  --output results/c4-900s
```

工具没有 `plan` 或 `--profile smoke/formal`。这里的 60 秒用于调通，900 秒是明确选定的测量窗口，不是自动获得的正式认证。测试**没有预热**：从新 salt
冷启动，窗口结束停止发新请求，再排空在途请求。总耗时可能更长；`--timeout` 默认 1800 秒且按请求计，不是整个 run 的超时。新输出目录必须不存在，工具拒绝覆盖。

需要曲线时分别测 C1/C2/C4/C8 等点并保留结果；比较 baseline/MOD 时先对齐 C、窗口、prepared 文件和客户端协议。不同 C
在有限窗口内到达的轮次混合可能不同。不要修改输出预算或截短会话来改善成绩，也不要把 60 秒短测和 900 秒结果混成同一 workload。

## 4. 跑完之后，先在本地看结果

```bash
cat results/c4-900s/summary.json
cat results/c4-900s/config.json
```

| 文件                   | 看什么                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `summary.json`         | 有效性、窗口吞吐、P90 解码速度、TTFT、样本数、排空及覆盖情况                          |
| `config.json`          | 原始 `run_id`、工具版本、workload 哈希、tokenizer、路由、窗口、卡数及声明的服务元数据 |
| `requests.jsonl`       | 每请求真实输出 ID、prompt 指纹、chunk 时间、lane/play/turn、usage 和错误              |
| `error.json`（异常时） | 运行异常；不能用缺失 summary 的 run 生成成绩                                          |

确认退出码为 0、`valid=true`、`aborted=false`、`failed_requests=0`，且实际 `measurement_seconds` 与计划窗口一致。缺少 token
IDs、echo/usage 不一致、输出预算不足或流不完整都会使 run 无效；不能手改有效性标记。检查 `decode_speed_samples`，缺少可测解码区间时 P90 为
`null`，不是零。

还需核对
`max_prompt_tokens_observed`、`max_turn_index_reached`、`sessions_completed`、`mean_client_inflight` 和
`full_concurrency_fraction`。短测可能没走到长轮次；配置 262,144 上下文不代表测到了 256K。客户端调度、JSON 和网络开销都在观测路径里，设置 C 不等于
engine 始终有 C 个活跃 decode。

`valid=true` 仅说明协议和形状检查通过，不证明硬件身份、真实缓存命中、充分样本、稳态或模型质量。记录的工具观察可能与新生成答案不一致：这不是 SWE 解题正确率或完整在线 agent
评测。**测试结束不会自动上传到网站。**

## 5. 把报告整理成 Frontier 数据

先保留原始 prepared 文件、结果目录及服务配置。公开时清理凭据、内网地址、机器绝对路径和私有内容；`requests.jsonl` 含生成内容且可能很大，不要将整个目录直接提交。保留原始
run ID、workload 哈希及数值，说明公开材料是完整报告还是脱敏摘录。

### 图上两个坐标怎样对应报告？

| Frontier 字段或坐标                  | SWE `summary.json` 字段                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `metrics.decode_p90_tps`，横轴       | `decode_tokens_per_second_p90`                                                                          |
| `metrics.output_tps`，部署总输出吞吐 | `output_tokens_per_second`                                                                              |
| 纵轴 output tokens/s/chip            | 网页将总吞吐除以 `configuration.hardware.accelerator_count`，应等于 `output_tokens_per_second_per_chip` |
| `metrics.ttft_p95_ms`                | `ttft_seconds_p95` × 1000；若源值为 `null` 则保持 `null`                                                |

不要把每卡吞吐填入 `output_tps`，否则会重复除卡数。总吞吐只计算窗口内收到的实际输出 IDs；排空请求在窗口内收到的部分算入，窗口后的部分不算。不能用包含排空的总 token
数重新计算，也不能套用历史 AIPerf 字段。

P90 是窗口内完整完成、且有可测解码区间的请求的 `(output_tokens - 1) / (last_token_time - first_token_time)` 的 P90，不能用
`1000 / TPOT P90` 替代。时间来自客户端 token-bearing SSE chunks；MTP 一个 chunk 可含多个 token，工具不虚构逐 token 时间。该
summary 不直接提供 Frontier 的 TPOT 指标，未另行定义、计算并留证据时应省略或为 `null`。

### 一条成绩还需要什么？

- **cohort**：模型及 revision、精度、SWE 固定 workload 合约和上下文要求。合约保留 source/工具 commit、prepared
  哈希、tokenizer/模板、接受子集、输出预算、无预热、盐与复用规则、窗口和聚合方式。只有合约完全相同才复用已有 cohort；不能挂到 AgentX cohort。
- **point**：唯一 ID、所属 cohort、engine/MOD、总卡数、完整服务配置、lane 并发及指标。
- **evidence**：`config.json` 的原始 `run_id`、可访问的 HTTPS
  报告链接、summary、配置和有效性/缓存证据；注明窗口、覆盖、客户端占用及已知限制。优先用固定 revision 链接，不编造缺失指标，不平均多个 run 的 P90 来冒充 pooled
  P90。

字段含义见 [Frontier 数据契约](../../../docs/LEADERBOARD-FRONTIER.md)。本目录的历史 AgentX JSON 不能直接改 workload
名称后提交；新成绩需要自己的 SWE 合约和证据。

## 6. Fork website，提交自己的数据 PR

在 GitHub Fork [vllm-hust-website](https://github.com/vLLM-HUST/vllm-hust-website)，把
`YOUR_GITHUB_NAME` 换成自己的用户名：

```bash
git clone https://github.com/YOUR_GITHUB_NAME/vllm-hust-website.git
cd vllm-hust-website
git switch -c frontier/my-measurement
```

真实上榜提交修改生产数据，而不是只把结果留在本示例目录：

1. 在 `data/leaderboard_frontier.json` 的 `points` 追加新 point，必要时在 `cohorts` 追加新合约。保留已有记录，不整份替换成自己的
   snapshot。
1. 在 `data/leaderboard_frontier_swe_evidence.json` 的 `runs` 追加 SWE 证据，保留对应的 summary、config、point/run
   ID 和限制；若目标分支尚无该文件，新增明确标识 SWE 协议的证据文件。不要写入历史 AgentX 的
   `data/leaderboard_frontier_evidence.json`，也不要继承其 runtime 或 acceptance 校准。
1. 检查 ID 唯一、point 与证据一一对应、报告链接可访问，且不是重复提交已发布的 run。

安装 Node.js 后，复用现有结构校验：

```bash
node - <<'JS'
const model = require('./assets/leaderboard-frontier-model.js');
const data = require('./data/leaderboard_frontier.json');
model.validate(data);
const runs = data.points.flatMap(p => p.evidence.run_ids);
if (new Set(runs).size !== runs.length) throw new Error('重复 run ID，请核对');
console.log('结构及 run ID 检查通过；仍需核对原始报告和配置');
JS

git diff --check
git diff --stat
git add data/leaderboard_frontier.json data/leaderboard_frontier_swe_evidence.json
git commit -m "data: submit SWE Prefix Reuse Frontier measurement"
git push -u origin frontier/my-measurement
```

在 GitHub 点击 **Compare & pull request**，选择上游 `vLLM-HUST/vllm-hust-website` 的 `main` 为 base，先打开 Draft
PR。正文可以按下面填写：

```text
模型与精度：
SWE 工具 commit / prepared workload 哈希 / 测量窗口：
硬件、总卡数、lane 并发、DP 路由：
engine / 后端 / MOD 版本及服务配置：
cohort / point ID / 原始 run ID：
P90 解码速度、总输出吞吐、每卡输出吞吐、TTFT P95：
报告和复现配置链接：
有效性、真实缓存命中、上下文覆盖、客户端占用及本地检查：
已知限制或与已有合约的差异：
```

检查 Files changed 和 CI，回答审查问题后等待维护者合并。结构校验通过不等于证据审查通过，PR 创建成功也不代表已经上榜。维护者核对 workload
归属、指标与配置，并完成发布所需的缓存版本更新。

## 7. 在哪里看到自己的成绩？

- **测试刚结束**：在客户端 `results/c4-900s/` 查看报告，网站尚无变化。
- **PR 审查中**：在 PR 查看 diff、检查结果与讨论。可在 website 仓库运行 `python3 -m http.server 8774 --bind 127.0.0.1`，打开
  `http://127.0.0.1:8774/leaderboard-runs.html#frontier` 预览。
- **合并并部署完成后**：打开
  [官网 Frontier](https://vllm-hust.sage.org.ai/leaderboard-runs.html#frontier)，选择对应的**模型＋精度**及 **SWE
  workload**，点击自己的点，再下载配置核对 point/run ID。

合并不等于部署完成。若没看到新点，先确认部署状态，再使用 `leaderboard-runs.html?v=<合并提交SHA>#frontier` 避免旧 HTML 缓存，并核对 cohort 与生产
snapshot。仅提交到 `data/examples/` 的示例不会被页面读取，也不会产生新的榜单点。
