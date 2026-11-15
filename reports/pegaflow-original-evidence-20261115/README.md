# PegaFlow 历史实验原件补充归档

2026-11-15。此批补齐当前仍保留的原始证据，方便离线复算与继续归因。原实验数值、失败状态和结论保持原样。

## 大型日志有多大

迁移后快照包含：资源采样日志 **4.687GiB / 20 个文件**，调度/执行 JSONL 轨迹 **0.719GiB / 31 个文件**，数值诊断 `.pt` 张量 **6.652GiB / 4,155 个文件**。这些是快照中的分类规模，不等于所有历史数据的总量，也不等于此前从未发布的字节量。

其中四卡标定目录仍有约 2.68GiB 内容。其原报告明确说明尚未完整归档大型原始日志；本补充包保留当前存在的 R/V 原件，并从仍保留的 `raw-M.tar.gz` 包中收录符合范围的 M 原件。其他已经清理、但在此前 benchmark 归档中保管的文件，继续以原有入口为准。

## 应该上传什么

[agent.md](../../agent.md)要求实验记录保留版本、模型、硬件、配置、原始结果入口等来源信息；[DATA_INDEX.md](../../DATA_INDEX.md)要求保留原件与失败记录，并将派生分析独立版本化。这些规则要求结论有可追溯的证据，未要求上传整个服务器目录。

- **保管并提供入口：** 用于复算的逐请求结果、原始计时、相关资源/调度日志、环境与启动配置、采集和分析程序、失败记录，以及支持诊断结论的必要张量。
- **大包的存放方式：** GitHub Release 保存原始字节，仓库保存说明、下载入口和 SHA256；包内 manifest 列出所有原路径、长度及哈希。避免把大包作为普通 Git blob 反复加入历史。
- **不纳入实验原件包：** 编译缓存、可重建运行环境、安装包、公开模型权重、重复归档；公开输入以来源和版本标识为准。凭据和无授权内容不得公开。
- **没有用于结论的临时探针：** 可以按明确保留策略清理；不能将失败实验或异常样本当作无用文件丢掉。

## 本次发布

原件：[GitHub Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-pegaflow-originals-20261115-v1)。附件及哈希见 [release.json](release.json)；全部附件已进行一次匿名下载，大小和 SHA256 与本地核验的附件一致，见 [public-readback.json](public-readback.json)。公开原件是保留快照的补充，不能据此声称所有历史文件都已恢复。

| 归档                                            | 原路径数 | 原路径逻辑大小 | 压缩包大小 |
| ----------------------------------------------- | -------: | -------------: | ---------: |
| pegaflow-original-text-20261115-v1.tar.gz       |    30544 |       7.573GiB |   971.7MiB |
| native-apc-diagnosis-20261013-tensors-v1.tar.gz |     2041 |       3.855GiB |  1455.7MiB |
| native-apc-semantics-20261013-tensors-v1.tar.gz |     2114 |       2.797GiB |  1858.5MiB |

日志包共 30,544 个成员，包含配置、资源采样、调度和执行轨迹、采集程序，以及启动失败/异常记录。`embedded-archives/raw-M/` 表示来自历史归档成员的原始文件；manifest 保留其原归档路径与成员名。日志包进行了明确的凭据模式筛查，未命中这些模式；该检查不构成无条件隐私保证。

两个张量包对应 2026-10-13 原生 APC 与缓存语义调查。各包内部相同字节只保存一个 SHA256 命名对象，所有原路径仍完整列在 manifest 中。对比和归因程序位于日志包及既有报告；张量不是新增性能测量，也不是完整模型权重。核验过程不执行 pickle、不加载模型。

此次没有删除服务器原件。运行环境、工具链和尚未纳入范围的文件仍按原身份保存，不应根据本归档的存在直接清空整个服务器目录。

## 如何核验

从 Release 下载需要的 `.tar.gz` 附件。使用 Python 3.10 或更新版本运行：

```sh
python verify.py /path/to/pegaflow-original-text-20261115-v1.tar.gz
```

它校验包内每个成员的长度和 SHA256，并输出整个包的 SHA256；再与 `release.json` 中对应值核对。日志包从 `files/` 下读取，张量包从 `objects/` 按 manifest 中的原路径映射读取。不要执行归档脚本或对来源不明的 `.pt` 直接调用 `torch.load`。

## 其他项目的现有做法

- Irisuko 发布的 [MuxPD mixed intervention 原件](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-first-mixed-intervention-20261029)：505,248,445 字节归档，2,172 个成员，包含输入、失败记录、张量、分析和成员清单。
- ShuhaoZhangTony 发布的 [runtime micro formal retest](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/runtime-micro-formal-retest-20261107)：用 Release 保管正式复测证据。
- [Frontier 历史归档](../../archive/frontier-recovered-20261021/README.md)明确采用选定原件归档及去重引用，排除权重、运行时二进制与 KV 存储；不是全服务器备份。

因此，本次沿用仓库已经采用的“大原件放 Release、报告和入口进 Git”方式，不将他人的选择性归档误称为所有同学都已上传全部文件。

## 以后每轮怎样管理

1. 工作目录放在对应工作区的 `/root/hust-zsx/runs/<run_id>`，记录清晰的模型、代码版本、配置与状态。
1. 每轮结束封存上述必要原件；小结果随报告提交，大原件作为带清单的 Release 附件。原件与模拟/派生分析分开标记。
1. 上传后核验公开下载的大小和哈希，并抽查或完整核验包内成员；保留清单和下载入口。
1. 确认可恢复后清理对应本地运行数据和可重建缓存。不要按“整个目录已有报告”推断所有原件都有备份。
