# Native MTP4 / vSpec gamma4 启动包

2026-10-27 Qwen3.5-35B-A3B 匹配有效 gamma4 实验的中性、可读启动包。 这是离线 `vllm.LLM.generate` harness，**不是
`vllm serve` 或 HTTP 压测**。 本包新做了 CPU 检查，没有重新执行 NPU 实验，不能据此声称适配任意新版软件栈。

## 范围与来源

- 包含六轮正式性能实验的启动链、冻结的 56 题输入、14 题预热输入、配置、私有兼容层与补丁。
- 不包含模型权重、CANN/驱动、vLLM 安装产物或可移植的完整环境镜像。
- 原始来源是同报告目录的 `raw/manifest.json` 所绑定证据胶囊，归档提交 `0ce4365692c335f31e2cbbb043d251e7f3d185ec`。原胶囊不修改。
- 本包将历史私有前缀统一改为 `BENCH_` / `bench`，删除未启用的可选诊断分支， 添加物理卡选择与启动入口，做格式规范化。它是明确标识的派生启动包，不是历史逐字节副本。
- 本包只覆盖最终 gamma4 对照；不启动此前的 14B、DiffSpec、norm 或分叉诊断实验。

## 宿主前提

当时为 Ascend 910B2 双卡，Linux/root，系统环境已经具备 CANN、ACL/HCCL、 PyTorch、torch-npu、Transformers、vLLM Core 与
vLLM Ascend。必须先加载该宿主的 CANN 环境，保留 ACL 等已有 `PYTHONPATH` / 动态库路径。没有通用的一键安装宿主栈步骤。

| 组件        | 历史提交                                   |
| ----------- | ------------------------------------------ |
| vLLM Core   | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM Ascend | `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd` |
| vSpec       | `4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae` |
| MMLU-Pro    | `b189ec765aa7ed75c8acfea42df31fdae71f97be` |

模型路径固定为 `/models/Qwen3.5-35B-A3B`，需要完整 tokenizer、config 与 MTP 权重。 若路径不同，同时修改 `gamma4-configs.json`
两臂的 `model`，并记录差异。 历史目标权重只有本地 config/index/size/mtime 证据，没有核验过不可变 upstream 权重 revision，
因此不能承诺重新下载同名模型即具有相同权重身份。旧目标清单在原胶囊 `research/speculative-mmlu-pro/evidence/q35-target-manifest.json`。

## 准备私有环境

在独立 Linux 工作目录解压后运行；不要把本包放到历史证据目录中运行。 先确认系统 Python 已能导入上述宿主依赖，再创建使用系统包的私有 venv。

```bash
python3 -m venv --system-site-packages venv
mkdir -p sources evidence
git clone https://github.com/vLLM-HUST/vllm-hust-vSpec.git sources/vspec
git -C sources/vspec checkout --detach 4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae
git -C sources/vspec apply --check ../../vspec-stats-compat.patch
git -C sources/vspec apply ../../vspec-stats-compat.patch
venv/bin/python -m pip install --no-deps --no-build-isolation -e sources/vspec
```

此处不自动升级宿主依赖；若缺少构建工具，需在私有环境补齐，不要修改共享宿主。 补丁只处理缺失的逐位置统计字段及空调度步反馈；`vspec_compat.py` 另处理宿主 API、 reorder
缓存与动态草稿宽度兼容，保留拒绝式接口检查，不修改 target 数学或放宽验收。 源插件许可证见 `VSPEC-LICENSE`。

## 启动完整六轮

先用 `npu-smi info` 确认已获准使用的物理卡空闲。历史使用物理 `0,1`； 映射不是固定合同，launcher 会现场查询，并在 worker 建立 context
后核验实际物理位置。 不要移除卡锁或放置检查来强行运行；新宿主输出格式不兼容时应停止检查。

前台方式（`BENCH_CARDS` 是物理编号）：

```bash
BENCH_CARDS=0,1 bash run.sh
```

后台方式：下列保留历史的 `Popen`、无 stdin、独立 session、日志落盘方式， 只把工作目录改为当前解压目录；日志已存在则拒绝覆盖。

```bash
BENCH_CARDS=0,1 venv/bin/python - <<'PY'
import os
import subprocess
from pathlib import Path

root = Path.cwd()
root.joinpath('evidence').mkdir(exist_ok=True)
with root.joinpath('evidence/gamma4-campaign.log').open('x') as log:
    process = subprocess.Popen(
        ['bash', str(root / 'run.sh')],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
print(process.pid)
PY
```

前后台二选一，不要同时启动。总控顺序为 native→candidate、candidate→native、native→candidate； 每轮重新创建引擎，每臂超时 1800
秒。失败不自动重试。现有证据不覆盖，重新运行使用新的解压目录。

## 调用链与配置

```text
run.sh
  → campaign_gamma4.py
    → venv/bin/python launch_gamma4.py PHYSICAL_CARDS LABEL screen_gamma4.py ARM perf56
      → gamma4-guard/sitecustomize.py
        → guard/sitecustomize.py（物理放置检查、candidate 私有兼容层）
        → gamma4_witness.py（计时区间外 metadata RPC）
      → screen_gamma4.py（LLM 初始化 → 预热 → generate 计时 → 保存结果）
```

`ARM` 为 `q35-mtp4` 或 `q35-adaptive`。不要仅复制最内层命令：总控会先清除遗留 `HUST_VSPEC_*` / `VSPEC_*` / `BENCH_*`
变量，再加载 `q35-environments.json` 的对应环境； launcher 设置 placement guard、插件、离线模式和 OMP 线程数。 原生臂复用环境文件的
`q35-mtp2` 键只是沿用历史环境键名，**模型配置实际为 MTP4**。

- 两臂 TP2+EP、BF16、B16、context16384、maxbatched8192、memory0.9、block128。
- APC、async、chunked prefill、FULL_AND_PIECEWISE 与同一请求 capture 列表开启。
- 原生 `VLLM_PLUGINS=ascend`、`num_speculative_tokens=4`。
- candidate `VLLM_PLUGINS=ascend,vspec`、引擎最大深度6、online候选2/4/6、cohort-refill开启。 历史实际始终
  gamma4，无切换；若新运行发生切换，总控拒绝称为匹配有效深度对照。
- 不设置 confidence margin，不安装 norm 修正，不开启逐步诊断探针。
- 14题×16 tokens 预热；56题一次提交，最多16条活跃，temp0/top_p1/max4000。
- 计时排除加载、编译、capture、预热、前后RPC；包含 generate 与逐请求完成 JSONL 落盘。

## 输出与验收

`evidence/gamma4-campaign-status.json` 必须为 `complete`。每个
`gamma4-performance-q35-{mtp4,adaptive}-r{1,2,3}` 标签产生 admission、command、config、
placement、partial、witness、result、exit 和日志文件。

检查全部 exit0、56题完整、invalid0、preemptions0、无 token0 退化，candidate 三轮 `gamma_switches=0`；还需核对两 rank、计时前后
gamma4 target FULL keys 覆盖相同 B1..16。 总控只执行其中的停止门，不要把退出码本身当成全部验收完成。

运行后直接离线复算（不加载 accelerator 库）：

```bash
python3 gamma4/analyze.py > evidence/recomputed.json
```

`gamma4/analyze.py` 检查题目顺序、重新解析答案、抢占、gamma 日志与各 rank 图键。 冻结的 `evidence/screen-manifest.json`
随包提供，不能根据运行结果重选题目。

原结论：原生/vSpec output tokens/s 中位数为811.05/795.60，未测到额外吞吐收益。 输出长度不同，不能把 wall 比当等工作量提速；共享宿主干扰和非完全相同
runtime 仍是边界。 详见 [历史结果](../research/speculative-mmlu-pro/gamma4/RESULTS.md)。
