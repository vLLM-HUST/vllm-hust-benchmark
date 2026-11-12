# 第三方一键推理测试

一键测试已集成到[vLLM-HUST Engine根目录one_click.py](https://github.com/vLLM-HUST/vllm-hust-engine/blob/main/one_click.py)。版本恢复、任务冻结、推理、评分和出图实现由Engine维护；本目录入口仅负责核验下载并转交Engine根入口，保留原上传诊断的显式兼容选项。

推荐从Engine仓库根目录运行下列命令；本目录兼容入口接受相同参数。

```bash
python3 one_click.py --model /absolute/Qwen3.5-35B-A3B --b1-python /absolute/fixed-sdk/bin/python --physical-devices 5,6 --out /absolute/new-run
```

首次需要两张可独占的 Ascend 910B2、模型 BF16 权重、宿主驱动及 CANN9.1/ATB、固定 B1
SDK（Torch2.13、torch-npu2.13rc1、Transformers5.14.1）。B1 运行时及 MOD 从私有 r4 冻结原件恢复；已有部署可把 `--b1-python` 换成
`--deployment /absolute/b1-r4`。B0 自动按官方 ARM64 镜像 digest 恢复
Python3.11.14、Torch2.9.0+cpu、torch-npu2.9.0.post1+gitee7ba04、Triton3.2.0.dev20260322、CANN8.5.1
及原文本补丁，不使用旧 pip/CANN9.1 兼容基线，也不要求 Docker 服务。已有固定 B0 可传 `--baseline-deployment`。

正式执行首先核对 `--out` 所在文件系统的实际余量，至少保留原4GiB启动余量；不足时在SDK核验和模型启动前生成封存失败报告。不同路径可能位于不同卷，请按输出目录检查空间。

默认执行清单在模型启动前冻结为 15 个阶段：两臂 C8 资格、MMLU-Pro 原固定200题配对、FrontierScience 两臂推理与共同 B1 裁判、HLE 两臂推理与共同 B1
裁判，以及SWE/Terminal公开轨迹的两侧真实推理。

| 数据集                 | 本轮任务范围                                                                  |
| ---------------------- | ----------------------------------------------------------------------------- |
| MMLU-Pro               | 原固定200题，max8192、温度0、seed0、原评分器                                  |
| HLE-Verified           | 固定分层200：Gold文本64/图像10，Revision文本113/图像13；solver16384/judge8192 |
| FrontierScience        | Olympiad100 + Research60，solver16384/judge8192                               |
| SWE-bench-Pro trace    | 官方公开轨迹按仓库比例固定32任务/64请求，max2048/并发8，只测推理性能          |
| TerminalBench2.1 trace | 作者完整provider请求固定32任务/48请求，max2048/并发8，只测推理性能            |

HLE/Frontier 沿用固定原客户程序、原提示与裁判解析，两臂统一使用同一 B1 Qwen 替代裁判，未决裁判单列。小样本诊断须显式加
`--smoke-evaluators`（HLE4、Frontier2；MMLU仍原200），不能充当全量成绩。首次第三方测试应重新执行本机 C8，省略
`--continuous-evidence`。

工具固定
[Engine主分支448362c](https://github.com/vLLM-HUST/vllm-hust-engine/commit/448362c7565597c8963c970dcfea0b235fd5fcf7)
与
[r4.3.8 Release](https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261030-r4.3.8)
的附件 id、大小、SHA256，以及独立固定的源码清单 SHA256。离线缓存仍逐文件核验，不能修改本地清单来放行任意源码。原 r4 四个附件及历史原始记录保留。

私有仓库凭据读取 `GH_TOKEN`/`GITHUB_TOKEN`，或当前目录、父目录、用户主目录 `.env` 中的这两个键；也支持 `--env-file /absolute/.env`。不执行
`.env`、不输出凭据，跨主机下载重定向去掉认证头。已有完整工具缓存可离线使用；原件和数据也须缓存完整。`--tools-cache` 指定工具缓存，`--cache`
指定运行时/数据缓存，`--dataset-assets` 指定含 `hle/`、`frontier/` 的已有数据根目录。

```bash
# 只验证固定工具：不表示 SDK 或模型已合格
python3 one_click.py --verify-only
# 完整准备、生成15阶段清单和四目标报告，明确标记未执行
python3 one_click.py --b1-python /absolute/fixed-sdk/bin/python --out /absolute/new-preparation --prepare-only
```

报告逐项显示四个目标是否达标及原因，区分本次运行与历史合格参考。原 M01 和正式 B0/B1
配对已有记录，默认自动校验复算；它们不能冒充本次新结果。三项质量子集与两项trace性能分别报告，回放不能标记五项端到端质量整体验收通过。失败、长度终止、非法答案、未返回项和原始 SSE
均保留；不自动重试发题、缩短预算或降级 eager。中断后等本次服务清理及租约释放完成才封存。

既有内存与磁盘准入修复通过52项Engine回归；本次3项报告回归核验大纲TPOT的请求完成边界、末块时间篡改与乱序时钟。Ruff通过。原始SSE可用于另存修订报告；运行中的冻结代码和原报告保留。

两臂C8已完成（B0 247/247、B1
618/618）。原self-run-kit/one_click.py实际执行的新MMLU配对，两臂各200请求成功，正确150/149、长度终止25/18、0未解析；吞吐465.78→935.99
tokens/s（+100.95%），平均TTFT/TPOT改善51.73%/50.16%。不据此推断统计质量等价。此前140正确的旧B0和启动失败记录独立保留。Frontier160已完成；新的HLE固定200按原冻结代码继续执行，五项端到端质量验收未完成。

SWE731/Terminal89原任务范围作为来源边界保留，端到端任务评分未执行。本轮按用户要求默认改用公开真实trace的固定性能抽样，无需执行任务Docker环境。CPU工作区已由平台侧交接、SSH/Docker与CPU至NPU只读health核验，当前不作为trace性能阻塞，也不再向NPU索要用户token。公开source、抽样分母与quality_measured=false见[执行大纲补充](../../reports/vllm-hust-midterm-20261019/milestone-v15/PUBLIC-AGENT-TRACE-SCOPE.md)。

旧诊断仅显式 `--legacy-diagnostic` 可进入，仍需对应旧部署配置；旧档案只读复核使用 `--review`。`original-kit/` 保留上传原件，`.py.source`
防止格式化改写。这些历史脚本及分数均不作为默认正式执行入口。

本轮实际推理使用启动前封存的r4.3.2源码；r4.3.7保留HLE固定200；r4.3.8新增公开trace固定抽样，代码与本机新测量分别标记；保留正确TPOT、固定B1 r4引擎和原生成协议。

r4.3.4曾错误改用末文本块计算正式TPOT，已撤回。V1.5大纲3.4.2明确为`(请求完成时刻−首token时刻)/(输出token数−1)`，逐请求等权平均。首末文本块间隔仅作为独立辅助指标，不参与正式目标判定。运行中的r4.3.2及MMLU配对原报告已采用正确边界，不受影响。

MMLU-Pro为固定200题子集，源test12,032题、14学科；不是本轮随机抽样。逐题ID、学科、源行/prompt哈希和参数见[大纲执行补充](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/reports/vllm-hust-midterm-20261019/milestone-v15/MMLU200-SCOPE.md)及[冻结清单](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/reports/vllm-hust-midterm-20261019/milestone-v15/mmlu-fixed200.json)。生成seed0不作为抽样seed；结果仅适用于该冻结子集，不能称为官方全集成绩。Engine在准备阶段保存mmlu-task-scope.json，并将范围与哈希绑定到执行计划。

HLE默认采用本轮确认的固定分层200题，全部ID、原任务哈希、抽样规则及完整回答/裁判预算见[HLE200大纲执行补充](../../reports/vllm-hust-midterm-20261019/milestone-v15/HLE200-SCOPE.md)。23道真实图像题保留，两臂同题同序；显式`--hle-profile full1811`可选全集，诊断HLE4另列。历史611题是短输出性能负载，不称历史质量抽样。旧全量HLE受控停止并保留全部部分输出；新200题重新执行，不以先返回的题目构造子集。

SWE/Terminal原文件64份共95,661,969字节已校验；112条所选请求通过独立目标模板渲染与编码核验，输入最长16350/16294
token。Terminal只有单条合格请求的任务不补齐。原生成历史与工具返回保留；SWE内部格式按固定官方版本转换，Terminal为真实provider请求体；公开清单不附原始对话全文，运行时按固定SHA下载，完整缓存可离线使用。实际服务发题前先逐条核对tokenization，B1须与封存B0的输入token
ID一致；不符则零生成停止。

83项Engine CPU回归、Ruff、真实数据prepare
CLI和准备报告HTML/PNG/SVG通过；准备未执行SDK/NPU验收。本机新trace配对尚未执行，须在当前HLE释放设备后运行；不能把工具准备写成实测结果。
