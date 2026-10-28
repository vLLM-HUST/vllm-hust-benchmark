# Terminal-Bench 2.1：ARM64 单题运行与评分

入口为 `scripts/bootstrap_terminal_arm64.py`，使用原 Harbor 0.24.0、Terminus-2 和原任务评分器。 目前只支持
`openssl-selfsigned-cert`。这是 ARM64 环境与用户操作链验收，不是 89 题全量成绩， 也不是 B0/B1 性能配对。

## 当前实测（2026-10-28）

在已有 ARM64 CPU 工作区，用新客户端、新任务镜像和新模型服务窗口执行：

| 运行                       | Reward | 原评分项 | 状态                     |
| -------------------------- | -----: | -------: | ------------------------ |
| 空答案 nop                 |      0 |      0/6 | 已评分                   |
| 官方参考解 oracle          |      1 |      6/6 | 已评分                   |
| Qwen3.5-35B-A3B 真实 Agent |      1 |      6/6 | 一次完成，客户端正常退出 |

真实 Agent 使用 vLLM 0.18.0 / vLLM-Ascend 0.18.0、两张 Ascend 910B2、BF16、TP2/EP。 共 7 个 Agent 回合、16 次工具调用，原始
usage 为输入 23,378、输出 2,400 tokens。 本次模型独立生成答案后直接调用原评分器，不是历史答案的补评分。
[脱敏验收记录](./terminal-arm64-acceptance/README.md)列出来源、资源、完整计数和证据哈希。

## 使用前

在本人已有的 ARM64 CPU 执行工作区打开终端。工作区需有独立 Docker 服务、Python 3.12 及以上、 可持久保存的目录，以及现有模型服务。模型和执行工作区分别管理，运行入口不申请
NPU、不部署模型。

可以直接让助手使用 [ops skill](https://github.com/yancanmao/npu-private-cloud-ops-skill) 完成准备：

> 请在我的 ARM64 CPU 工作区按此指南运行 Terminal-Bench 2.1 的 OpenSSL 单题。沿用已授权模型服务， 自动填写运行目录和私有模型
> profile；完成原评分器正负控制、一次真实 Agent 与评分，保留失败和原始记录。 不扩成全量，不把 ops skill 提供给被测 Agent。缺少平台能力由运维处理。

运维负责确认本人工作区、镜像下载源、Docker socket、客户端与 daemon 共享路径、资源与现有 API， 给出实际入口及填好的命令。使用者无需寻找集群管理员凭据。
本入口验证已有执行工作区中的客户端路径；新账号建工作区和平台网页终端需单独验收。

## 下载固定版本入口

下载本次实际执行的固定脚本提交：

```bash
python3 - <<'PY'
import base64, json, urllib.request
from pathlib import Path
revision = "2109c36ba3fd6523e00cf3e6b2ecc725d693609a"  # pragma: allowlist secret - public source commit
url = ("https://api.github.com/repos/vLLM-HUST/vllm-hust-benchmark/contents/"
       "scripts/bootstrap_terminal_arm64.py?ref=" + revision)
with urllib.request.urlopen(url, timeout=120) as response:
    item = json.load(response)
Path("bootstrap_terminal_arm64.py").write_bytes(base64.b64decode(item["content"]))
PY
```

使用 GitHub contents API，可避免某些平台无法连接原始文件域名的问题。下载成功不代表运行已通过。

## 准备、控制和一次运行

下面两条路径由运维替换成本人已授权的持久目录和私有配置；不要照抄占位符运行。

```bash
RUN_DIR="/PERSISTENT_PATH/terminal-first"
MODEL_PROFILE="/PRIVATE_PATH/terminal-model.json"
python3 bootstrap_terminal_arm64.py prepare --directory "$RUN_DIR"
python3 bootstrap_terminal_arm64.py controls --directory "$RUN_DIR"
python3 bootstrap_terminal_arm64.py run --directory "$RUN_DIR" --profile "$MODEL_PROFILE"
python3 bootstrap_terminal_arm64.py status --directory "$RUN_DIR"
```

`prepare` 自动安装独立客户端及 Compose 插件、下载和校验固定任务与依赖、构建原生 ARM64 镜像， 验证 Docker 共享路径。`READY` 仅代表准备完成。构建和任务均采用
1 核、2 GiB，独立网络 MTU 1400； 运维可用 `--cpuset` 限定工作区内已分配 CPU。磁盘使用所在持久卷额度，未实现每题独立 10 GiB 配额。 首次准备要求目录所在卷至少有
5 GiB 剩余，这是准备检查线，不是整个工作区容量保证。

`controls` 依次用两个新容器执行 nop 和官方 oracle：空答案应得 0/6、参考解应得 6/6。 参考解控制不调用模型，不计入模型解题成绩，也不传给被测 Agent。

`run` 只运行一次模型任务，自动调用原评分器。固定 100 turns、单次输出最多 4096 tokens、 Agent 900 秒、评分 900 秒、temperature 0、seed
17、32K 上下文、关闭 thinking 和总结。 整题重试为 0，LiteLLM SDK 重试为 0；Harbor/Terminus 原内部调用两层各最多 3 次尝试保留， 所以 100
turns 不等于最多 100 次 HTTP 请求。不同模型或预算需单独冻结配置，不能悄悄改本入口。

同一目录已启动的任务会拒绝再次执行。准备阶段失败可在定位、修正问题后用 `--resume-prepare`， 它只允许尚未启动任何控制或模型任务的目录；不会自动重跑 Agent。

## 模型配置由运维填好

私有 profile 示例不包含密钥值：

```json
{
  "api_base": "https://MODEL_HOST/v1",
  "model": "SERVED_MODEL_NAME",
  "api_key_file": "/PRIVATE_PATH/model.key"
}
```

密钥文件须仅本人可读写。也可用 `api_key_env` 指定已有安全环境变量名。 运维应确认实际服务模型、运行版本和资源窗口；模型列表或 HTTP 200 不代表真实任务成功。

## 查看结果

运行目录包含 `prepared.json`、`requirements-installed.txt`、各角色 `*-job.json`、
`*-started.json`、`*-receipt.json`，以及 `jobs/terminal-user-*/` 下的 Harbor 原始结果、轨迹与评分输出。
`model-receipt.json` 的 `reward`、`tests`、`passed` 是最简结果入口。

- `reward=1` 且原评分器六项通过：这一次任务解题成功。
- `reward=0` 且评分完整：真实质量失败，保留为有效结果。
- `reward=null`、准备异常或没有评分输出：未完成评分，不写成 0 或通过。

原始题目、答案、完整轨迹、模型地址与凭据保留在私有工作区。公开报告只提交脱敏身份、计数、 评分与证据哈希。任务结果不证明全量兼容、全量准确率、性能收益或新账号开户成功。

## 固定来源与依赖修复

- 任务：`harbor-framework/terminal-bench-2-1`，提交 `7131e4375048a0e408a8fb404b5f499d726b695b`。
- Harbor：PyPI `harbor==0.24.0`；关键执行文件与上游 `b53b8134e1241686dca7759af188f987ecc48e8b` 核对。
- ARM64 Python 父镜像清单：`sha256:608b27b2db85abf38d81685085cd1166a5ed7ada66549f0dfe53a75bb830cd5d`。
- 原 `instruction.md`、`solution/solve.sh`、`tests/test.sh`、`tests/test_outputs.py` 保持字节不变。

历史 OpenSSL Agent 首次评分因下载 uv 超时未得分；恢复同答案后 6/6 属于历史补充评分。 新入口在构建时准备固定 uv 0.9.5 ARM 包、官方安装器、pytest 8.4.1
与 pytest-json-ctrf 0.3.5。 只有原评分脚本中的精确安装器 URL 使用经哈希校验的本地副本，其余 curl 请求仍走真实客户端。 构建阶段也使用 MTU 1400
和资源限制，避免只限制任务而遗漏准备过程。

本次单题实测覆盖准备、正负控制、真实 Agent、原评分器和结果读取。其他 ARM64 任务仍须逐题核验。
