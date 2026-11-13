# ascend-91 重启后的独立批次（restart-02）

工作区重建后，物理卡从 0/7 变为 3/4，逻辑卡仍为 0/1。原运行目录不在新 overlay 中； 旧失败及已取回抽取记录继续保留。本批结果不能与旧硬件批次或 9 月记录合并。

本批基于 PegaFlow PR #49 固定提交 `e2cdc2f7b32d8294b54a74545c05d103e5a5ecaf`， 完整重建 Ascend
wheel；框架与模型版本维持已记录配置。14 个权重分片完整 SHA-256 均匹配， SWE 45 项 CPU 测试通过、2 项跳过；prepared workload hash 与上批一致。

## 当前资格检查

- 两臂各 20 次 case-4 短输入输出一致，并且两臂全部相同；另四组短输入也逐 token 一致。
- 两个真实 continuation 前缀分别为 7551、8712 tokens。Native 本地命中分别为 4096、6144 tokens。 PegaFlow 清空本地缓存后分别实际执行 2
  次成功加载（TP 两个 worker），保存/加载失败增量均为 0； 冷请求、本地命中、外部恢复的 32-token 输出全部与 Native 相同。
- 两臂 C2/60s 协议资格检查均有效、失败 0。此短检查不是正式吞吐对照，也不是 SWE 解题质量。 先前单次输出差异本批未复现，不能据此宣称已定位或修复其原因。
- 初次启动因权重校验留下约 61 GiB 文件缓存，CANN host-pinned 分配失败。 仅对本任务读过的模型分片使用文件缓存释放提示后，容器占用降到约 1 GiB； 两个逻辑卡均通过 4
  KiB/1 MiB/16 MiB pinned 分配测试。失败与清理收据均随原件保留。 没有全局 drop_caches 或删除文件；后续仅在服务启动前处理本任务模型文件缓存。

完整、逐文件 SHA 校验的原始诊断包见 [capsules.json](capsules.json)。原件包括请求 payload/响应、 SWE 逐请求
token/时间流、指标、命令、服务日志、源码身份、编译/模型收据和失败记录。 日志中的注销/共享内存警告发生在完成检查后的关闭阶段；没有被改写为检查内错误或删除。

## 正式对照的预注册配置

[formal-plan.json](formal-plan.json) 固定两臂 C1/C2/C4/C8/C16 各 900 秒，先 Native 后 PegaFlow； primary 为
C8。若 C8 为正，倒序 PegaFlow→Native 各 900 秒复测 C8；否则若 C16 为正则复测 C16。 保留所有正、零、负和失败结果，不在第一个正结果后停止。

两臂均 BF16/TP2/PP1/DP1、20 GiB KV/卡、MTP2、FULL_AND_PIECEWISE、相同 seed/工作负载； 候选使用 30 GiB host pool。8
组短输入及两个强制恢复样本必须通过，随后 C2/180s 资格检查。 这不验证 RDMA、旧 T1–T5、完整 256K 并发能力或 SWE 解题质量。

每个已完成窗口先保存到持久卷，打包原始字节并在 E 盘核验后 ACK，再等待 15 秒开始下一档。 采集/压缩/下载均位于计时窗口之外；每 60 秒检查选定 NPU 的进程归属，外来进程使窗口中止。
执行源码原字节快照为 [run_formal.py.source](run_formal.py.source)、
[formal_runtime.py.source](formal_runtime.py.source)，其 SHA 均写入计划。 使用时复制为对应 `.py` 文件，准备计划所列同版本环境及
workload，以 `PEGAFLOW_RUN_ROOT` 指定独立运行根目录；发布计划后写入含计划 SHA/公开 commit 的 `plan-publication.json`，再运行控制器。
这些是本批的执行来源，不能把本批资格检查或尚未完成的窗口写成正式结果。

验证：在本目录运行 `python verify.py`；根目录验证器也会调用它。

## 正式资格探针的增补记录

首次正式资格探针在 Native 的第 7 组（1877-token 输入）冷/热调用之间出现 token 差异； 当时尚未启动任何 900s 窗口。核对代码发现它遗漏了先前诊断使用的 1 秒等待。
失败原件和旧计划/源码保存在 `failed-back-to-back-oracle.tar.gz`，见 `supplements.json`。 新计划明确在每次计时外的正确性探针前等待至少 1
秒，随后重新检查；正式 SWE 客户端继续连续发请求。 这是对探针条件不一致的修正，尚不能归因或声称解决运行时的普遍数值差异。旧失败不会删除。
