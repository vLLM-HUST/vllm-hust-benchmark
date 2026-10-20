# SWE-bench Pro V1：运行一个真实 Agent

通用环境、任务与镜像校验、官方 Agent 运行入口和用户指南由 **vllm-hust-benchmark** 维护。 `request-throttling-controller` 只提供可选的
Controller 接入与验证；运行本指南不需要克隆或安装它。 这是独立诊断工作负载，不会自动成为网站榜单成绩。

## 运行链路与前提

```text
本仓库入口 → 固定版本官方 Agent → 已有 OpenAI-compatible 模型 API
                          ↓
                一个独立 ARM Docker 任务容器
                          ↓
                逐步日志、官方轨迹、私有补丁
                          ↓
                可选：官方评分器独立复验
```

需要 Linux ARM64 CPU 环境、Python 3.10+、可用 Docker daemon/CLI，以及到模型 API 的网络。 首次准备镜像还需要 `crane`
和对应仓库访问权限。Agent 在 CPU 上运行；模型可以由另一个已部署的推理服务提供。 入口不部署、重启或修改模型，也不要求特定 NPU、RIDE、Ray 或 Controller。 默认示例只选
NodeBB 的一道 V1 任务、一个独立 Agent。任务容器需要至少示例指定的 2 CPU/4 GiB； 这不是已证明足够所有测试的资源承诺。

请使用独立的 **CPU 客户端虚拟环境**，不要把客户端依赖装进现有模型服务环境。 现成已验证的客户端可以直接复用，随后仍会重新校验来源和输入。

## 1. 获取本仓库与固定版本上游源码

本迁移尚在 PR 分支，合并前请使用下列分支；合并后可改用 main。 从本仓库根目录运行；`WORK` 必须是自己的绝对路径，`sources` 目录必须尚不存在：

```bash
git clone --branch codex/swe-bench-pro-environment https://github.com/vLLM-HUST/vllm-hust-benchmark.git
cd vllm-hust-benchmark
export WORK=/absolute/private/swe-pro-v1
mkdir -p "$WORK"
PYTHONPATH=src python3 scripts/prepare_swe_bench_pro_environment.py --root "$WORK/sources"
python3 -m venv "$WORK/client"
"$WORK/client/bin/python" -m pip install -r docs/examples/swe-bench-pro/client-requirements.txt
"$WORK/client/bin/python" -m pip install -c docs/examples/swe-bench-pro/client-requirements.txt -e "$WORK/sources/agent"
"$WORK/client/bin/python" -m pip install --no-deps -e .
```

源码下载器输出 `source-receipt.json`，逐文件核对本仓库
[`swe_bench_pro_v1_lock.json`](../src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json) 的 SHA256。
客户端依赖文件记录已验证组合；安装结束后保存 `pip freeze`，如依赖解析改变组合应先解决，不能当作同配置复现。

固定上游来源：

- [官方 Agent](https://github.com/scaleapi/mini-swe-agent/tree/d74716a3c8104a113f77cc9ab94cf407ecdcf1e9)
- [官方 Pro V1 评分与任务代码](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862)
- [社区 ARM 镜像构建来源](https://github.com/bdqnghi/swe_benchmark_arm/tree/b2834f1b0cf94294f409e93c7dccc398b0a2699b)

实际 Agent、任务格式化代码及评分文件都会在运行前检查，不能仅凭已安装包的版本号替代固定源码身份。

## 2. 准备 V1 数据及私有配置

从
[ScaleAI/SWE-bench_Pro](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/tree/2d52cb3df914a3fcf80c7f66738b3a88ae37fc50)
获取有权访问的 **v1** parquet，放入私有目录。锁定文件的 SHA256 为：

```text
c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8
```

复制示例，按实际环境编辑绝对路径、Docker socket、模型名称及 API 地址：

```bash
cp docs/examples/swe-bench-pro/single-agent.json "$WORK/settings-private.json"
```

需要填写的字段：

| 字段                                | 含义                                                            |
| ----------------------------------- | --------------------------------------------------------------- |
| `paths.dataset_parquet`             | 已下载的固定 V1 parquet                                         |
| `paths.agent_source`                | 上一步的 `sources/agent`                                        |
| `paths.official_v1_source`          | 上一步的 `sources/official-v1`                                  |
| `paths.docker`、`paths.docker_host` | 本用户有权使用的 Docker CLI 与 daemon                           |
| `paths.crane`                       | `crane` 可执行文件绝对路径                                      |
| `model.model_name`                  | LiteLLM 的提供方前缀及服务模型名，例如 `openai/Qwen3.5-35B-A3B` |
| `model.api_base`                    | 已有模型服务的完整 `/v1` 地址                                   |
| `model.api_key_env`                 | 需要认证时填环境变量名称，密钥值仅放环境变量                    |
| `run_id`                            | 本次唯一运行标识；每次新运行使用新标识和新目录                  |

不要把凭据写进 URL 或 JSON。示例 `api_key_env: null` 用于无需认证的自有服务。 示例保持
`size=1`、`program_instances=1`、`independent_agents_per_task=1`，不自动扩大批次。 当前锁定清单覆盖 8 道 NodeBB V1
任务；扩展其他任务需要补齐对应官方文件哈希与 ARM 镜像验收。 可以用 `batch.instance_ids` 显式指定一道任务，必须同时匹配 repository 与 size。

## 3. 检查、准备与运行

安装后命令 `swe-bench-pro`、`python -m vllm_hust_benchmark.swe_bench_pro` 和
`PYTHONPATH=src python scripts/run_swe_bench_pro.py` 使用同一个实现。以下使用客户端虚拟环境：

```bash
export PYTHON="$WORK/client/bin/python"
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro check \
  --settings "$WORK/settings-private.json" --output "$WORK/check"
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro prepare \
  --settings "$WORK/settings-private.json" --output "$WORK/inputs"
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro run \
  --settings "$WORK/settings-private.json" --inputs "$WORK/inputs" \
  --output "$WORK/run-001" > "$WORK/run-001-private.log" 2>&1
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro summarize \
  --settings "$WORK/settings-private.json" --output "$WORK/run-001"
```

`prepare` 校验数据/源码，固定 ARM 镜像 digest 和本地 image ID，并检查容器代码起点。 Docker 29 环境采用 `crane pull` 加
`docker load`，不依赖 Docker Hub 的直接 pull。 如果已有本入口或历史环境生成的私有准备材料，可用下列命令复用；它会重新核对完整数据行、源码及不可变镜像身份：

```bash
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro prepare \
  --settings "$WORK/settings-private.json" --reuse-inputs /absolute/previous/inputs \
  --output "$WORK/inputs"
```

输出目录必须不存在。运行写入独占 intent；失败不自动重提、不自动改配置。每个 Agent 新建独立任务容器， 不挂宿主代码目录，结束后只清理本次容器，保留模型服务。CPU 运行环境若要求由
Ray 管理进程， 可把同一命令交给自己的具名 Ray job/actor；这是外部进程管理，不改变通用入口。

## 4. 逐步观察与判断结果

```bash
tail -f "$WORK/run-001-private.log"
cat "$WORK/run-001/task-000/progress-private.json"
cat "$WORK/run-001/task-000/report-private.json"
```

- `agent-events-private.jsonl`：模型请求起止、成功回复 usage、工具命令及完整工具输出。
- `trajectory-live-private.json`、`trajectory-private.json`：官方格式实时/最终轨迹。
- `patch-private.diff`：已修改但可能未完成的补丁；未主动提交时也保留。
- `report-private.json`：`status`、`model_calls`、`agent_submitted`、补丁字节及容器清理结果。
- `batch-report.json`：本次一个 program 的汇总；`quality_score: null` 表示尚未评分。
- `runner-provenance.json`、`input-verification.json`：入口源码身份和输入检查。

示例 100 步、1200 秒、每次最多输出 4096 tokens、工具超时 60 秒、temperature=0、thinking=false， 是显式诊断协议，不是最优修题参数。HTTP
read=null 只取消读取时限，其余三阶段仍各 180 秒。 模型上下文上限由服务端决定，包含输入和预留输出；本入口不会自动调大上下文。 固定官方 Agent
保留完整累计历史，仅对单次过长工具输出裁剪字符，没有全局历史压缩。 因此 100 步也可能先遇到上下文限制；日志会保留这个真实失败，不把它写成任务完成。

运行进程 exit=0 只说明编排与结果保存结束，Agent 是否完成必须读取 report 的 status。

`Submitted` 只表示 Agent 主动结束并给出补丁，不能证明修复正确。`ContextWindowExceededError`、
`LimitsExceeded`、超时均是执行失败；未评分不等于 0 分。迁移验收状态见
[单 Agent 验收记录](../reports/swe-bench-pro-environment-migration-20261020/README.md)。

## 5. 独立评分（可选）

```bash
"$PYTHON" -m vllm_hust_benchmark.swe_bench_pro grade \
  --settings "$WORK/settings-private.json" --inputs "$WORK/inputs" \
  --batch-directory "$WORK/run-001" --output "$WORK/grade-001"
```

此步骤在模型计时之外调用固定版本官方评分器，用独立容器复验保存的补丁，不发送模型请求。 评分器另有自己的 Python 依赖，按固定源码的 requirements 安装到专用 CPU 环境后执行。
保存 `grading-status.json` 及评分原件；只有官方输出完整才能判定 resolved，缺失输出保留未决。

## Controller 接入与迁移边界

Controller 可选实现 `integration.hook = "module:factory"`：`before_run`、`before_program`、
`after_program`、`after_run` 四个生命周期钩子。不配置则不导入任何 Controller 模块。 Controller 专属身份验证、Native/Controller
选择、会话释放和机制观测留在
[request-throttling-controller](https://github.com/vLLM-HUST/request-throttling-controller)。
通用入口不含模型部署、容量调参或政策实现。

通用准备/评分逻辑从 Controller 的历史 `ea13820` 提取，官方核心保持固定来源； Controller 历史脚本、配置和实验记录可以作为备份保留，但新用户入口以本仓库为准。
本次迁移不改写历史运行、网站矩阵或已有实验成绩。
