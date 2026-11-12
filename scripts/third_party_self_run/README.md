# 第三方一键推理测试

默认 `one_click.py` 现在调用固定的 vLLM-HUST Engine 三数据集入口，首先校验工具、原件、SDK、模型及完整任务清单，再启动真实 NPU
推理并自动评分、出图。旧的每臂十条诊断不再是默认测试。

```bash
python3 one_click.py --model /absolute/Qwen3.5-35B-A3B --b1-python /absolute/fixed-sdk/bin/python --physical-devices 5,6 --out /absolute/new-run
```

首次需要两张可独占的 Ascend 910B2、模型 BF16 权重、宿主驱动及 CANN9.1/ATB、固定 B1
SDK（Torch2.13、torch-npu2.13rc1、Transformers5.14.1）。B1 运行时及 MOD 从私有 r4 冻结原件恢复；已有部署可把 `--b1-python` 换成
`--deployment /absolute/b1-r4`。B0 自动按官方 ARM64 镜像 digest 恢复
Python3.11.14、Torch2.9.0+cpu、torch-npu2.9.0.post1+gitee7ba04、Triton3.2.0.dev20260322、CANN8.5.1
及原文本补丁，不使用旧 pip/CANN9.1 兼容基线，也不要求 Docker 服务。已有固定 B0 可传 `--baseline-deployment`。

正式执行首先核对 `--out` 所在文件系统的实际余量，至少保留原4GiB启动余量；不足时在SDK核验和模型启动前生成封存失败报告。不同路径可能位于不同卷，请按输出目录检查空间。

默认执行清单在模型启动前冻结为 11 个阶段：两臂 C8 资格、MMLU-Pro 原固定200题配对、FrontierScience 两臂推理与共同 B1 裁判、HLE 两臂推理与共同 B1 裁判。

| 数据集                          | 本轮任务范围                                                       |
| ------------------------------- | ------------------------------------------------------------------ |
| MMLU-Pro                        | 原固定200题，max8192、温度0、seed0、原评分器                       |
| HLE-Verified                    | Gold668 + Revision1143，保留全部211道图像题，solver16384/judge8192 |
| FrontierScience                 | Olympiad100 + Research60，solver16384/judge8192                    |
| SWE-bench-Pro、TerminalBench2.1 | 需另建CPU任务工作区，当前入口尚未接入                              |

HLE/Frontier 沿用固定原客户程序、原提示与裁判解析，两臂统一使用同一 B1 Qwen 替代裁判，未决裁判单列。小样本诊断须显式加
`--smoke-evaluators`（HLE4、Frontier2；MMLU仍原200），不能充当全量成绩。首次第三方测试应重新执行本机 C8，省略
`--continuous-evidence`。

工具固定
[Engine主分支fafec75](https://github.com/vLLM-HUST/vllm-hust-engine/commit/fafec75763ec7acecd7a1323398b7dea1dd73d27)
与
[r4.3.5 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4.3.5)
的附件 id、大小、SHA256，以及独立固定的源码清单 SHA256。离线缓存仍逐文件核验，不能修改本地清单来放行任意源码。原 r4 四个附件及历史原始记录保留。

私有仓库凭据读取 `GH_TOKEN`/`GITHUB_TOKEN`，或当前目录、父目录、用户主目录 `.env` 中的这两个键；也支持 `--env-file /absolute/.env`。不执行
`.env`、不输出凭据，跨主机下载重定向去掉认证头。已有完整工具缓存可离线使用；原件和数据也须缓存完整。`--tools-cache` 指定工具缓存，`--cache`
指定运行时/数据缓存，`--dataset-assets` 指定含 `hle/`、`frontier/` 的已有数据根目录。

```bash
# 只验证固定工具：不表示 SDK 或模型已合格
python3 one_click.py --verify-only
# 完整准备、生成11阶段清单和四目标报告，明确标记未执行
python3 one_click.py --b1-python /absolute/fixed-sdk/bin/python --out /absolute/new-preparation --prepare-only
```

报告逐项显示四个目标是否达标及原因，区分本次运行与历史合格参考。原 M01 和正式 B0/B1
配对已有记录，默认自动校验复算；它们不能冒充本次新结果。三数据集成功也不能标记五数据集整体验收完成。失败、长度终止、非法答案、未返回项和原始 SSE 均保留；不自动重试发题、缩短预算或降级
eager。中断后等本次服务清理及租约释放完成才封存。

既有内存与磁盘准入修复通过52项Engine回归；本次3项报告回归核验大纲TPOT的请求完成边界、末块时间篡改与乱序时钟。Ruff通过。原始SSE可用于另存修订报告；运行中的冻结代码和原报告保留。

两臂C8已完成（B0 247/247、B1
618/618）。原self-run-kit/one_click.py实际执行的新MMLU配对，两臂各200请求成功，正确150/149、长度终止25/18、0未解析；吞吐465.78→935.99
tokens/s（+100.95%），平均TTFT/TPOT改善51.73%/50.16%。不据此推断统计质量等价。此前140正确的旧B0和启动失败记录独立保留。当前Frontier/HLE整轮仍在进行，尚未完成五数据集验收。

SWE731/Terminal89原任务范围已冻结，尚未执行。CPU工作区应由模型侧平台控制入口创建，再向NPU容器交回连接信息；当前会话未提供此入口。容器Agent的401只说明没有用户创建权限，不能据此称Coder禁止这种流程。用户已要求全部五项，不能写成用户延期。

旧诊断仅显式 `--legacy-diagnostic` 可进入，仍需对应旧部署配置；旧档案只读复核使用 `--review`。`original-kit/` 保留上传原件，`.py.source`
防止格式化改写。这些历史脚本及分数均不作为默认正式执行入口。

本轮实际推理使用启动前封存的r4.3.2源码；r4.3.5核验原始流首末时钟，严格保留大纲的请求完成TPOT；保持固定B1 r4引擎和原生成协议。

r4.3.4曾错误改用末文本块计算正式TPOT，已撤回。V1.5大纲3.4.2明确为`(请求完成时刻−首token时刻)/(输出token数−1)`，逐请求等权平均。首末文本块间隔仅作为独立辅助指标，不参与正式目标判定。运行中的r4.3.2及MMLU配对原报告已采用正确边界，不受影响。
