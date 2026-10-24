# #326 活跃消费者收尾与历史保留清单

本轮结束“逐篇迁移旧数值”的开放式队列。保留原始记录与历史身份；仅当影响网站、报告、CI、索引或实际工具读取时重新打开修复。归档数量增加不代表新增有效实验结论。

## 本轮核查范围与处理

- 当前网站和报告：网站提交 `8a05ce2323bbf6240b77277e4c380045cdd74690`；部署和 CI 固定 benchmark
  `fe0c3f18e81234d62cb50f0561fbf8de931aff82`。核查 V0.78 加载器、报告身份与发布清单，按固定来源生成并验证 137 个数据文件；不更改测量或报告版本。
- 数据索引与 Release 入口：刷新当前网站源码引用、线上发布 manifest 与附件 API。修复过期入口快照：1074 条引用、18 个 Release、253
  条附件元数据。上传失败状态不隐藏，元数据可见不冒充下载/SHA 已核验。
- Profiler 实际工具：controlled-fault matrix、runtime-hook pair-plan、concurrency anomaly、trace diagnosis
  四个 CLI 已用外部校验 collection 读取并生成隔离临时输出，输入逐文件 SHA 不变；补入 canonical CI，未修改运行时逻辑。
- Profiler 其他消费者：复用既有 public export、paper exporter/stager、M0/preflight 输入输出门禁和 package
  CI；不重新生成历史原件、不重启 NO_GO 实验。
- 组织防回流：66 仓库的当前 Git/Release 指纹与中央策略提交 `ceaba13bc55210e8c66f44b3cc4a620ed5acaed9`
  基线比较，未发现新增漂移。这不等于零历史债务，也不是所有脚本都已动态执行。

索引来自 [data-entrypoints-sources.json](data-entrypoints-sources.json)，查询入口为
[data-entrypoints.json](data-entrypoints.json)。本轮最终 CI、线上验证和合并回执记录于
[#326](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/326)；这些状态不能由本页的清单数量推断。

## 历史保留：不再作为持续迁移队列

[机器可读清单](issue326-retained-history.json)保存组织盘点中的现有路径/附件身份及以下处理规则：

- 已迁移数据在 benchmark 保持原字节，原 MOD 中的方法、版本、协议阈值、模板和历史叙述可以保留；不以清空所有数字或 legacy 字符串为目标。
- 重复研究笔记、旧 smoke 数值和开发期测试计数，仅保留历史身份及既有原文入口；未发现活跃读取问题时不再逐篇拆 PR。
- 安装包、wheel、sdist、镜像/部署依赖和兼容性载体不等同于实测副本。无零消费者证明的资产保留，不以归档完成为由删除。
- 清单中的 `retain-reviewed-baseline-not-zero-consumer-proof`
  仅表示维持已审查状态，不证明每个命中都无消费者、无数据或可删除。中央检测是回流检测，不是所有代码路径的完备性证明。
- 已退役的项目本地审批框架只作历史语境；不作为当前门禁或新的待办。

## 缺失、私有和外部依赖

- 33 处旧 probe：保留丢失说明，停止追查；不启动 GDN 补测。只有用户提出新的明确实验目标才考虑新测，不为消除旧缺口补测。
- Profiler 旧 validation JSON、私有 G4/G1 记录和临时
  trace：保留有边界的未匹配说明，原文表格不当作恢复原件。仅在原保管者提供同一身份原件或活跃任务确实依赖时重开。
- 旧论文 PDF digest：已知归档 PDF 身份不同，不冒认恢复；找到匹配原 digest 的文件再处理。
- Issue19 私有日志、校准资料：私有/不公开与丢失分开，继续使用已核验的脱敏公开 export；由保管者提供合规公开投影或受控读取途径。
- Triton #10、LMCache #1、SGLang #2：原审核 runner、硬件和准入门禁继续由各仓库维护者处理。不换 runner、不代加准入标签、不放宽保护，以原门禁通过为解除条件。

具体身份与公开边界见
[M0 消费者说明](https://github.com/vLLM-HUST/vllm-hust-request-lifecycle-profiler/blob/23913f321753493b8ad80a64d73d8d7d1b9937a9/docs/m0_consumer_evidence.md)、[论文身份说明](https://github.com/vLLM-HUST/vllm-hust-request-lifecycle-profiler/blob/23913f321753493b8ad80a64d73d8d7d1b9937a9/docs/notes_evidence.md)及机器清单。缺失叙述不是全局永久丢失的证明。

## 收口规则

本轮可完成部分以入口索引、活跃读取回归、历史保留清单和最终 CI
回执收口。剩余项标为历史保留、接受缺失、私有边界或外部依赖；不再使用“其他混合文档继续盘点”作为无限队列。只有可复现的活跃消费者故障、结论/来源错误、CI/索引回归或新出现的同身份原件才重新处理。

本轮未新增设备实验、未改 B0/B1、未宣称 Profiler M0 通过、未删除安装资产。#326 的外部阻塞可继续开放，但不应被解释为要求继续搬运所有历史文字。
