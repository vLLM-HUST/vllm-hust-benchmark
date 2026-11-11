# 第三方测试复现工具：先核对受测版本

本目录修复上传的
self-run-kit。默认入口首先核对指定测试的版本和实现身份，再允许模型启动；预检不通过也生成四项目标的报告和图表。当前仍未完成整套正式复现，不能把工具修复或十请求诊断称为验收通过。

```bash
python3 one_click.py --verify-only
python3 one_click.py --rebuild-runtime /tmp/new-isolated-runtime --out /absolute/path/to/new-run
python3 one_click.py --review /absolute/path/to/new-run
```

版本依据是 benchmark 固定提交 `d67aea6eb3eb0dee271a01b4d66acce8b65bd034` 中的《项目里程碑测试大纲》V1.5 课题四及中期报告 V1.03。B0
固定 0.18/0.18；B1 按数据集、工况和指标选择实际受测的优化组合，不能用容器里的任意旧版本或当前 main 代替。网站版本页也明确把 main 快照与通过验证的运行组合分开。

[版本依据](version-authority/tested-runtime.json)保存 V1.03 完整回答工况的实际版本、配置、原源码清单和私有源码包 SHA256。该工况使用
0.28.1/0.25.1rc2、Torch 2.13、BetterScale E24/R36、MTP2、KV 重算及 Microflow 等组合。`runtime-profile.json`
只选择本机安装位置，不提供合格证明。检查器读取实际解释器的包元数据和导入路径。reference 身份要求历史实现包及原安装源码清单；reconstructed 身份从固定 SHA256 的 SDK
恢复并校验实际源码文件，明确作为新构建，不声称恢复历史优化组合。包版本相同不能替代组件激活验证。

容器已重建 0.28.1/0.25.1rc2 基础环境并校验 4,771 个源码/库文件；当前默认本机配置明确采用 reconstructed 身份。完整历史 B1
实现包与原源码回执尚未绑定。旧八月源码已从默认启动流程移除；针对它的 GDN 修复只保留历史诊断身份，不据此断言目标 B1 或最新引擎 main 有同样缺陷。

[执行清单](EXECUTION-CHECKLIST.txt)列出固定输入、任务分母、四项门槛及缺项。既有 `pipeline/` 后端仍只执行每臂十个真实诊断请求，尚不能替代 V1.03 的 200
题完整回答、高并发测试和五数据集原生任务。完整源码恢复后，还须接通对应冻结工作流，不能仅修改版本号后运行旧诊断负载。

本容器依赖 Python 3.12、用于 B0 的 Python 3.11 与开发头文件、CANN、物理 NPU 5/6、同一份本地 Qwen3.5-35B-A3B
权重及既有私有物化输入。源目录里的绝对路径记录此次部署，不附带模型、题文、gold 或完整答案。不会在通用 CPU 主机上自动产生真实 NPU 结果。

每次使用新目录；失败、未执行请求和原异常均保留。先确认本轮全部写入进程停止，再封存校验清单。`--review` 重算文件清单、哈希、请求分母、回执和配对指标。退出 0
只表示对应执行与归档成功；四项正式结论以报告字段为准。

```bash
.report-venv/bin/python -m unittest test_failure_reporting test_native_timing test_process_cleanup test_version_gate
python3 -m unittest test_hust_shape
```

第一组验证失败报告、时钟归一化、孤儿进程清理及版本拒绝；第二组需要本容器 PyTorch，只验证旧版 reshape 的数值等价和动态编译。B0 兼容补丁已完成 10/10 实机诊断；目标 B1
尚未完成版本订正后的实机执行。详见 `REPAIR-RECORD.json`。

`original-kit/` 保留上传脚本的原始字节。仓库副本中的 `.py.source` 对应原 `.py` 文件，以免格式化器改写原件；旧 SHA256SUMS
保留原文件名。当前源码校验清单与旧清单分开。
