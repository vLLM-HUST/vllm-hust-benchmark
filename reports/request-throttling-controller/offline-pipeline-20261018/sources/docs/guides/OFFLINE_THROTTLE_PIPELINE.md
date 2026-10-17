# 离线轨迹窗口实验统一入口

目标是将完整输入批次在time0提交，客户端不限流，用同代理路径上的调度关闭/开启对照验证KV压力下的窗口吞吐。入口是 `scripts/run_offline_throttle_pipeline.py`，不包含新的调度算法。

## 复用边界与配置

| 层次 | 现有实现 | 统一入口的职责 |
|---|---|---|
| 工作区、模型、Controller供应 | 平台Coder与RIDE标准入口 | 读已就绪部署plan；可选调用显式供应命令 |
| 文件分发 | RIDE `deployment_payload.prepare_payload` | 校验Pod/cluster UID、冻结文件SHA、拒绝覆盖 |
| 执行 | Ray Jobs REST API | 具名任务、提交前落盘intent、同集群窗口互斥 |
| 测量 | `run_throttle_window_trial.py` 与既有replayer | time0/不限流校验、逐program依赖、同定义窗口、owned program清理 |
| 回收、公开副本 | `collect_short_throttle_trial.py` / `export_short_throttle_trial.py` | 核对run/job/root/report/seed，保留私有包，脱敏并独立验证 |
| 汇总、备份 | `summarize_throttle_windows.py` | 配对比取中位数、顺序复测单列、公开备份本地暂存 |

从 `examples/offline-throttle-pipeline.json` 和 `examples/offline-throttle-window-template.json` 复制一份私有配置并填写真实路径；相对路径以配置文件所在目录为准。地址、运行时环境、凭据不能放进公开报告。

部署plan来自已有RIDE部署记录，包含 `payload_identity`（namespace、cluster及UID、pod及UID）、当前 `ray_address`、`runtime_env`、`execution_resources`。Ray地址必须匹配当前已核验Pod IP。window模板提供模型、ON/OFF代理、输入SHA与测量窗口；新run的operator/root/output/replayer路径由入口统一生成，避免复制时漏改目录。

当前测量driver验证的是已审计的APC/512MiB混合KV配置（容量49152、每program buffer14335）。其他容量/模型必须另行审计、使用新配置与run-id；不能仅换模型名沿用结果。

## 从准备到公开副本

```bash
python scripts/run_offline_throttle_pipeline.py all --config private/offline-pipeline.json
```

每个seed按 `runs` 顺序串行执行：冻结输入与源码 → RIDE分发 → Ray提交/查询 → 终态回收 → 公开导出与独立验证 → 全部配对汇总。整批大小由CSV决定，所有arrival_time必须为0；`concurrency=0` 表示不设客户端slot上限。window内依赖不被展开成无依赖请求。

如果部署plan尚不存在，配置可选的 `provision_command` 为现有平台/RIDE供应入口的**显式argv数组**。入口只在 `all` 且plan缺失时执行它，随后要求该命令产出承诺的ready plan；不修改共享模板、不擅自创建第二套调度/部署机制。供应命令应使用绝对路径。没有供应配置时，缺失plan直接拒绝。

分步检查：

```bash
python scripts/run_offline_throttle_pipeline.py prepare --config private/offline-pipeline.json
python scripts/run_offline_throttle_pipeline.py launch --config private/offline-pipeline.json
python scripts/run_offline_throttle_pipeline.py status --config private/offline-pipeline.json
python scripts/run_offline_throttle_pipeline.py finish --config private/offline-pipeline.json
```

`prepare` 拒绝同名目录；`launch` 拒绝已有提交intent。连接丢失后先用 `status`，再用 `all` 恢复观察：不重复提交已有intent，不因本地watch超时取消远端任务。Ray失败/停止保留原始状态，不能进入成功汇总。

对已有具名任务，显式在相应run条目写入真实 `job_id`，保留原有 `plan.json`、`window-plan.json`、输入与原始副本，再运行 `finish`。这条路径只查询并复核既有结果，不重新测量。已有正确archive/public/summary会核对后复用，内容不同直接拒绝。

产物包括每run的冻结payload与source manifest、提交intent、实时状态、私有raw/source包、公开副本、`pipeline-receipt.json`，以及总 `summary.json`。可用 `summary_output` 指定不与其他实验冲突的汇总路径。

可选 `benchmark_repo` 指向专用benchmark checkout；公开副本暂存到 `reports/request-throttling-controller/<run-id>/`，不覆盖不同内容。该动作只完成本地暂存。Git远端备份、代码/数据PR、矩阵审阅和网站部署分别记录，不能把暂存当推送或发布。工作区停止由平台生命周期入口执行，归档/审阅材料交付后核实Pod、NPU释放与持久化数据保留。

## 本轮验证与适用范围

新增6项测试覆盖路径重绑定、arrival/客户端限制、重复目录、冻结文件变更、seed换标签，以及真实生成的RIDE/Ray启动代码与提交intent。统一入口的 `finish` 已在实际四个Ray终态上运行：r15/r16/r17/r18均VERIFIED，重新生成的汇总与已公开冻结三seed结果逐字段一致，没有提交新测量。fresh-run启动分支复用了此前真机提交所用的RIDE/Ray API，并做配置/生成代码测试；这次没有为包装脚本额外重跑模型或新建workspace。新供应命令hook没有另做一次真机供应验收。

结果为seed11/17/23配对窗口输出吞吐比中位数1.916861倍；r16仅为seed11顺序复测。它不是完整96条批次makespan、SWE-bench修复正确率、原矩阵配置或多后端routing收益。
