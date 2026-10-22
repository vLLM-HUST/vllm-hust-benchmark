# docs/ — 知识库导航

本目录是插件的规范化知识库:新人上手、agent 编程规范、架构契约、发布流程、
踩坑史与进度记录都收在这里。根目录 README 只保留"是什么、怎么装、证据在哪",
背景与纪律一律下沉到本目录。

## 阅读顺序(新人 15 分钟路径)

| 顺序 | 文档 | 回答的问题 |
|---|---|---|
| 1 | [../README.md](../README.md) | 这个仓库是什么、装完怎么验证 |
| 2 | [architecture.md](architecture.md) | 插件壳/kernel 库/宿主三者什么关系,manifest 各字段什么语义 |
| 3 | [agent-guide.md](agent-guide.md) | 在这个仓库写代码必须遵守什么(人机通用) |
| 4 | [support-matrix.md](support-matrix.md) | 某个能力在我的配置下会不会生效、不生效是哪一条 |
| 5 | [release.md](release.md) | 怎么构建、检查、发布、停止/卸载,翻 `active` 前要过哪些门槛 |
| 6 | [pitfalls.md](pitfalls.md) | 前人踩过哪些坑,怎么避免重踩 |
| 7 | [progress.md](progress.md) | 项目走到哪了,接下来计划做什么 |

## 单一事实源分工

与仓库既有约定一致,每个主题只在一处维护,其余地方只放指针:

| 主题 | 权威位置 |
|---|---|
| 使用方法、env 开关、验证证据 | [../README.md](../README.md) |
| 能力 × 准入条件 × 代码位置 | [support-matrix.md](support-matrix.md) |
| **cohort 边界（我们能声称什么）× 四项归属** | [cohort-and-ownership.md](cohort-and-ownership.md)（约束入口 [../AGENTS.md](../AGENTS.md) 硬约束 6） |
| 宿主提案补丁（tier-2，未提交上游） | [`host-proposals/README.md`](../host-proposals/README.md)（章程）+ `host-proposals/<id>/`（`proposal_meta.json` = env 语法/哈希/版本点钉/上游状态）；`docs/upstream/` 只是路径消歧 stub |
| 宿主 seam 与 monkeypatch 面清单 | [../HOST_CONTRACT.md](../HOST_CONTRACT.md) |
| 历史实现溯源（索引 + 逐条核对结论） | [../PROVENANCE.md](../PROVENANCE.md) / [provenance-verification.md](provenance-verification.md) / [../provenance/](../provenance/)(只读) |
| 算子落点:CCE(`fa_fp32_stage1` / `lse_merge`)在仓库外 wheel;**Triton/纯 Python 载体在本仓** | `ops/kernels/ascend-kernel/README.md`(工作区,仓库外) + 本仓 `src/vllm_ascend_split_batch/` |
| 算子调度(选择)接口 | [design/operator-dispatch.md](design/operator-dispatch.md)(骨架,未接线) |
| 架构语义、编程规范、发布、踩坑、进度 | 本目录 |

改行为必须同步改对应权威文档;本目录只放指针,不复制内容。
