# Terminal-Bench 2.1：在 Coder 中运行 ARM64 单题

使用 Harbor 0.24.0 和 Terminus-2，在独立 Docker 沙箱中完成 `openssl-selfsigned-cert`，随后运行任务原评分器。

## 1. 创建工作区

打开 [Coder](https://coder.sage.org.ai:9443)，选择 `swe-bench-pro-cpu` 模板创建工作区，家目录选 20 GiB 或以上，然后打开终端。这个
CPU 模板同时适用于本页的 Terminal 任务，提供 Python 3.12 和独立 Docker 服务。

准备好模型的 OpenAI 兼容 API 地址（含 `/v1`）、模型名和所需 API 密钥。模型通过 API 连接；工作区内运行客户端和任务沙箱。

## 2. 下载入口并准备环境

在工作区终端执行：

```bash
SHA=5b468a8a20750148c59f0a112f4bded579bd28abcd0908bea5356c2da6a21871 # pragma: allowlist secret - public artifact hash
BASE=https://coder.sage.org.ai:9443/swe-bench-pro-artifacts/terminal
curl --fail --location "$BASE/$SHA/bootstrap_terminal_arm64.py" --output bootstrap_terminal_arm64.py
echo "$SHA  bootstrap_terminal_arm64.py" | sha256sum --check

RUN_DIR="$HOME/terminal-first"
BASE_IMAGE=docker.m.daocloud.io/library/python@sha256:608b27b2db85abf38d81685085cd1166a5ed7ada66549f0dfe53a75bb830cd5d # pragma: allowlist secret - public image manifest
python3 bootstrap_terminal_arm64.py prepare --directory "$RUN_DIR" --base-image-ref "$BASE_IMAGE"
```

看到 `READY` 后继续下一步。入口会安装 Harbor 和 Compose，下载并校验固定任务及依赖，构建 ARM64 镜像，再检查客户端与 Docker 的目录共享。首次准备需要至少 5
GiB 可用空间。构建和任务沙箱各限 1 核、2 GiB，网络 MTU 为 1400。

这里使用镜像源下载固定 Python 基础镜像，同时核对 manifest、镜像 ID 和 ARM64 架构。可访问 Docker Hub 的环境也可省略 `--base-image-ref`。

## 3. 检查评分器

```bash
python3 bootstrap_terminal_arm64.py controls --directory "$RUN_DIR"
```

两次控制分别使用新沙箱：空答案 `nop` 应为 0/6，官方参考解 `oracle` 应为 6/6。完成后即可运行模型。

## 4. 填写模型 API 并运行

替换下面的 API 地址和模型名。通过环境变量传入密钥：

```bash
MODEL_API="https://MODEL_HOST/v1"
MODEL_NAME="SERVED_MODEL_NAME"
read -rsp "API key: " MODEL_API_KEY; echo
export MODEL_API_KEY

python3 bootstrap_terminal_arm64.py run --directory "$RUN_DIR" \
  --model-api "$MODEL_API" --model-name "$MODEL_NAME" --api-key-env MODEL_API_KEY
python3 bootstrap_terminal_arm64.py status --directory "$RUN_DIR"
```

服务无需认证时，省略读取密钥的两行，将 `--api-key-env MODEL_API_KEY` 换为 `--no-api-key`。已有私有配置文件也可继续使用
`--profile /path/to/model.json`。

`run` 执行一次 Agent 任务，并自动进入原评分器。固定预算为 100 回合、每次最多输出 4096 tokens，Agent 和评分各 900 秒，temperature 0、seed
17，关闭 thinking 与总结。整题和 SDK 重试为 0；Harbor 内部调用重试沿用上游设置。

## 5. 查看结果

| 阶段       | 结果入口                                                        |
| ---------- | --------------------------------------------------------------- |
| 环境就绪   | `prepared.json`、`install.log`、`base-pull.log`、`build.log`    |
| Agent 执行 | `model.log`、`jobs/terminal-user-model/*/agent/trajectory.json` |
| 原评分器   | `model-receipt.json`、`jobs/terminal-user-model/*/verifier/`    |

`model-receipt.json` 中，`reward=1` 为通过，`reward=0` 为未通过，`reward=null` 为未评分；`passed/tests`
给出评分项计数。原始报告、终端记录和评分日志都保存在 `RUN_DIR` 下。

每次模型尝试使用新目录。准备失败时先查看对应日志，修复后可加 `--resume-prepare` 继续；已经启动控制或模型任务的目录保留原结果。

## 验收记录

本入口覆盖 OpenSSL 单题，Terminal-Bench 2.1 共 89
道任务。新工作区验收记录和历史结果见[验收页](./terminal-arm64-acceptance/README.md)。

## 固定来源

- [入口源码](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/980f89884a7036526adca2c5811fd69189c8187d/scripts/bootstrap_terminal_arm64.py)。
- 任务：`harbor-framework/terminal-bench-2-1`，提交 `7131e4375048a0e408a8fb404b5f499d726b695b`。
- 客户端：`harbor==0.24.0`，Agent 为原 Terminus-2。
- 评分依赖：uv 0.9.5 ARM64、pytest 8.4.1、pytest-json-ctrf 0.3.5。

题目、参考解和两份测试脚本保留上游字节。镜像在构建时准备评分依赖，原测试脚本中的固定 uv 安装器 URL 使用经校验的本地副本。源码、依赖和镜像身份记录在运行目录中。
