# TerminalBench2.1 OpenSSL原答案恢复评分

TerminalBench2.1
OpenSSL采用Qwen3.5-35B-A3B官方0.18/0.18，同一次Agent共5次模型调用、86.812555秒。原评分因uv下载在900秒超时，原reward=null保留。固定uv0.9.5传输缓存后仅恢复评分，原6项测试全部通过、reward=1，耗时11.784143秒，仍使用原900秒预算；新增Agent、模型调用及控制实验均为0。四个SSL文件复用原快照；check_cert.py由原命令及终端回显确定性重建，非现场原文件快照，原模式未知。已执行1/89题，该题恢复评分通过，另88题未执行；独立ARM64派生镜像，不构成全量或B0/B1配对验收。

## 配置与方法

Harbor 0.24.0 /
Terminus2，原任务、测试及评分器不变，1CPU、2GiB、MTU1400。原Agent最大100轮、900秒、max_tokens=4096；原Harbor及Terminus内层重试策略各保留3次上限，SDK重试0，不套用SWE零重试合同。恢复评分仅一次，预算900秒。固定官方uv安装器及ARM64包经哈希核验，仅更换下载传输为本地缓存；不声称传递依赖wheel与历史完全相同。

## 结果与证据

评分前后5文件SHA及模式一致；原env文件缺失警告保留，随后原uvx/pytest执行成功。所有模型服务和评分容器均已释放。原超时/null见original-trial.json；恢复结果见summary.json、tests.json和recovery-receipt.json。独立复核见independent-review.json及profiling-review.json；受限原件索引见restricted-evidence-index.json。私有题目、答案、证书私钥和原始轨迹未公开。

离线校验：

```sh
python3 verify.py
```

离线入口复核公开聚合及绑定，不声称可脱离受控原件重跑原始审计。
