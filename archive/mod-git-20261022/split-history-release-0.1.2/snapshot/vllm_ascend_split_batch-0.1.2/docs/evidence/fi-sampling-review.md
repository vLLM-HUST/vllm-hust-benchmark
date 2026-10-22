# Review 材料：fi-sampling（flashinfer 语义采样接入）

> 一页材料，可独立阅读。结论先行：**能力已落地、default-off、FI 路由 100% 生效，无正确性缺陷；
> 但 e2e 收益落在噪声带内不可判、joint 分支在常规 4k 部署下不可达**——因此**建议不因性能翻
> catalog `qualified`**；若要翻需先补 manifest 注册与不可变安装目标。qualified 的对外提交动作
> 留给用户/团队 review，本材料只给证据与建议，不自行对外发布。

## 1. 它是什么

不引入新 kernel：把 W2 已 review（approve）的 flashinfer 语义采样（triton-ascend 移植）接进
vllm-ascend 的采样路径。

- 机制：`vllm.general_plugins` entry `fi-sampling` → `fi_sampling_plugin.load()`（env 门控）→
  类级替换 `vllm_ascend.sample.sampler.AscendTopKTopPSampler` 为子类，**仅覆盖 `forward_native`**
  （与 fork `_310p/sample/sampler.py` 同款先例）。**宿主源码零改动**，未注册 `vllm.platform_plugins`。
- 分流（纯逻辑 `fi_sampling_route.py`）：无截断 → FI api1 kernel；k=1 全行 → argmax；
  `B≥256 且 k≥32 且 top-p` → FI joint kernel；其余 → fork 原链。
- 五类强制回退（回退腿**零设备同步**）：per-request generators / processed logprobs /
  batch-invariant / reduce-sample / async-exponential。config 读不到时按"五类全命中"保守回退。
- 开关：`VLLM_HUST_FI_SAMPLING`（默认 `0`，未置 = 不 import 宿主/内核、不 patch、零差异）。

## 2. 证据链（全部可溯源）

| 项 | 结果 | 出处 |
| --- | --- | --- |
| CPU 守护 | `pytest -q` **226 passed**；`ruff check .` 全绿 | 本次 §3 前置（`docs/evidence/fi-sampling-s3/raw/extension-ladder.txt`） |
| 新基线 e2e（api1，六腿交替） | Δmedian TPOT **−0.052%** < 噪声带 **0.368%** → **噪声带不可判** | `docs/evidence/w2b-fi-sampling/EVIDENCE-refresh-2026-09-10.md` §3、`bench/results/fisampling_refresh_e2e.json` |
| FI 路由生效 | ON 腿 `fi_api1` **1300/1300 ×3 腿**；OFF 腿 `fi_sampling` 出现 **0** 次 | 同上 §3.4 + `bench/results/logs_fisampling_refresh/serve_api1_*.log` |
| joint 可达性 | 512-ctx 形态实测 **`max_B=504`**，直方图 `{'fork':136,'fi_joint':64}` | 同上 §4（`serve_joint_on_j.log`） |
| **§3 启用验证（本次）** | `check` compatible+enabled；serve 启动成功；`fi_sampling ... ACTIVE` ×1；`fi_api1` 75/75；`/health` 200；chat 冒烟 200×2；0 TypeError/OOM | `docs/evidence/fi-sampling-s3/section3-enablement.md` + `raw/` |
| 宿主漂移事故与修复 | as-shipped 在新基线曾 **100% 静默走 fork**（上游删 `enable_async_exponential` → fail-closed）；已修复入库 | `EVIDENCE-refresh-2026-09-10.md` §2、commit `879e0ff` |

## 3. §3 启用验证记录（本次，2026-09-11）

完整记录：`docs/evidence/fi-sampling-s3/section3-enablement.md`（原始日志在 `.../raw/`）。

- 启动：`vllm-hust-ext run -- vllm serve <Qwen2.5-14B-Instruct> --max-model-len 4096
  --gpu-memory-utilization 0.85 --port 8441 --generation-config vllm
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[32,64,128]}'`（卡 7，flock）。
- 三能力共存门控证据行：cascade `cascade plugin loaded (gate=1, graph_gate=1)` ×2、
  fia-demask `de-mask applied` ×1、fi-sampling `fi_sampling sampling path is ACTIVE` ×1。
- 路由：`route=fi_api1 B=1`（20 条 trace）+ 直方图 **`{'fi_api1': 75}`**、`route=fork` = 0。
- `k=1 argmax` 与 joint 分支本次冒烟未触发（无截断腿、B=1），其行为以 §2 的路由/可达性证据为准。
- **关键偏差（必须随结论读）**：`run --dry-run` **不注入** `VLLM_HUST_FI_SAMPLING`——fi-sampling
  **不是 manifest carrier**（catalog 明确"当前刻意不注册"）。故本次启用 env 由环境面提供，
  §3 验证的是「能力在 manager 启动的真实 serve 中门控生效」，**不是**「manager 注入 enable env」。

## 4. 两条必随形态事实

1. **无 e2e 增益——落在噪声带**。新基线 api1 六腿交替：Δmedian **−0.052%**（74.961→74.922 ms/步），
   噪声带 **0.368%**；逐腿配对符号在三条配对上翻转。机理自洽：B=64 时采样仅占 TPOT ≈1%
   （W2 单算子"省 0.9 ms/步"投影上限 1.2%）。**"已生效"≠"有收益"**——路由 1300/1300 命中，
   但该形态下省不出可测的时间。
2. **joint 采样在常规 4k 场景是死分支**。单卡 4k 上下文 KV 容量把 `B` 卡在 ~122 < 256 门槛，
   joint cell 永不可达；只有把上下文压到 ~512（`max_B=504`）才命中。故 joint 分支是"按 W2 判据
   把已实测赢的区间接上"的安全保留，而非常规部署的性能来源。

## 5. 风险评估

| 风险 | 等级 | 说明 / 缓解 |
| --- | --- | --- |
| 宿主耦合面漂移 | 中 | 只依赖 `AscendTopKTopPSampler` 名称 + `forward_native` 签名 + 3 个只读 config 旋钮。本 baseline 只逐项复核过 1 处（async-exponential，`879e0ff`）；其余（类名/签名/`logprobs_mode`/`vllm.envs` 名称）**尚未逐项重核**（见 §8 遗留 1）。**升级宿主时按 `docs/release.md` §4 清单复核。** |
| 第二套采样语义 | 中 | FI 的 RNG 流与 fork 链不同 → 同 prompt 下 on/off **token 序列不逐位对齐**（分布等价，非逐位等价）。默认 `raw_logprobs` 下不影响正确性；任何逐位复现验收须以 off 腿为基线。 |
| k=1 的 argmax tie | 低 | 默认 k=1 走 argmax；存在精确 tie 时可能选不同下标。位级一致需求可设 `VLLM_HUST_FI_SAMPLING_K1_ARGMAX=0` 回退 fork 链（未做 tie 专项对拍）。 |
| 默认关闭安全性 | 低（已证） | 未置 `VLLM_HUST_FI_SAMPLING` 时 `load()` 在 import 前返回——不 patch、不导入 `triton`、日志零输出；OFF 腿 3/3 零 `fi_sampling` 痕迹。 |
| manifest 未注册 | 中 | 无法通过 `vllm-hust-ext` enable 注入开关（本次 §3 偏差）；也是 catalog `enablement.allowed` 的直接缺口（§6）。 |

## 6. qualified 翻转建议（**建议性质**，供团队 review）

**现状（catalog `org.vllm-hust.fi-sampling`）**：`maturity=preview`、`availability=preview`、
`enablement.allowed=false`（blocker：未注册 manifest）、`tested_effects=neutral/inconclusive`。

**catalog `qualified` 的硬门槛**（`vllm_hust_ext/catalog.py`）：
`functional=passed ∧ recovery=passed ∧ installation非空 ∧ availability=available ∧ enablement.allowed=true`。
本条目 `functional/recovery` 在**kernel 层**已 passed（191 PASS/0 FAIL、族测全绿），缺的是：
(a) `enablement.allowed`（fi-sampling 未注册 manifest）；(b) `installation`（无不可变安装目标）。

**建议（推荐 A）**：

- **A（推荐）不翻 `qualified`**：保留 `preview` / `not-recommended-for-tested-cell`。
  理由：qualified 的唯一实质增量是"可启用"，而 enablement 在已验证的形态下**买不到时延**
  （§4 事实 1），却引入"第二套采样语义"的默认可达面。诚实状态即 preview——
  按 catalog 政策，性能不达标的正确扩展本就应留在 catalog 并记 `not-recommended-for-tested-cell`。
  只需把条目的 `tested_effects`/`recommendation` 文本刷新为新基线口径（−0.052% 噪声带、
  joint `max_B=504`），并把旧 blocker 里"e2e benefit is unproven"细化为"e2e benefit not observed
  at the tested cell"。
- **B（备选，若团队要"可启用"）**：先按 fia-demask 先例
  （`docs/design/fia-decode-demask.md` §8：单能力独立开关）注册 **独立 bundle**
  `org.vllm-hust.fi-sampling`（`implementation[]` + `activation.environment={"VLLM_HUST_FI_SAMPLING":"1"}`），
  走一次纯 §3（此回 dry-run 才会注入开关）；**但仍保持 `maturity=preview`**，直到 `installation`
  （不可变 wheel/源码 + sha256）落地——`qualified` 在缺安装目标时不可达。

**无论 A/B**：`qualified` 今天都不可达（缺 `installation`）；对外改 catalog（改 maturity/availability/
enablement、写 `installation`）属"对外提交动作"，**本材料不执行**。

## 7. 交付物与提交

| 项 | 位置 / commit |
| --- | --- |
| 实现（default-off 接线 + 纯逻辑分流） | `src/vllm_ascend_split_batch/fi_sampling_plugin.py`、`fi_sampling_route.py`、`fi_sampling/`（commit `fd255ee`；漂移修复 `879e0ff`） |
| 新基线 e2e 证据 | `docs/evidence/w2b-fi-sampling/EVIDENCE-refresh-2026-09-10.md` + `bench/results/fisampling_refresh_e2e.json`（刷榜素材入库 commit `ac33382`；本分支终态等价 commit `783794a`，额外含 EVIDENCE-refresh 叙述） |
| §3 启用验证记录 | `docs/evidence/fi-sampling-s3/section3-enablement.md` + `raw/`（本次） |
| 本 review 材料 | `docs/evidence/fi-sampling-review.md`（本次） |
| 目录侧 | catalog 条目 `org.vllm-hust.fi-sampling`（工作区 `knowledge/surveys/catalog/`，**本次未改**） |

## 8. 已知边界 / 遗留项

1. **宿主漂移类缺陷面未清**：本次只复核/修复了 1 处旋钮缺失；`AscendTopKTopPSampler` 签名、
   `logprobs_mode` 取值、`vllm.envs` 名称**尚未逐项按新宿主复核**（建议按 `REPORT.md` 复审同款方式
   做一次"宿主面清单核对"）。
2. **单算子基线未在新 triton-ascend 3.2.2 / 图模式下重测**：§4 事实 1 的 1.2% 上界借自旧基线 eager，
   仅为借用上界；解析型收益投影需在新基线重跑单算子。
3. **joint cell 在常规 4k 部署下仍是死分支**：本次只复测了 512-ctx 可达性，未测 4k 形态。
4. **e2e 收益未显形**：形态受限（采样占步时 ~1%），非路由失效（路由 100% 命中）。
5. **k=1 argmax / RNG 语义差异**：见 §5，未做 tie 专项对拍与逐位对拍。
6. **vendor 目录 lint 例外**：`src/vllm_ascend_split_batch/fi_sampling/` 逐字复制自源包，已入 ruff
   `extend-exclude`；升级源包需重新同步并记录 hash。

## 9. 裁决（2026-09-11）

- **日期**：2026-09-11
- **裁决人**：用户（团队拍板）
- **裁决依据**：本材料 §4（两条形态事实）、§5（风险评估）、§6（推荐路径 A）
- **裁决内容（三条）**：

  1. **不因性能翻 `qualified`**：catalog 条目 `org.vllm-hust.fi-sampling` 保持
     `maturity=preview` / `availability=preview` / `recommendation.level=not-recommended-for-tested-cell`。
     理由：唯一实质增量是"可启用"，而已验证形态下 enablement **买不到时延**（api1 六腿 Δmedian
     −0.052% 落在 0.368% 噪声带内不可判），却引入"第二套采样语义"的默认可达面。
  2. **不注册 manifest**：维持 `fi_sampling_plugin` **刻意不注册**现状（无 `implementation[]`/
     `component`、无 `activation.environment` 键），即 `run --dry-run` 继续**不注入**
     `VLLM_HUST_FI_SAMPLING`（该事实已记入 §3 记录 §2.1/§7）。
  3. **不注册独立 bundle**：未选择本材料 §6 的备选路径 B（"可启用"分支），`pyproject.toml`
     与 manifest 均不动。

- **生效动作清单**：

  | # | 动作 | 落点 |
  | --- | --- | --- |
  | 1 | 条目文本刷新为新基线口径（api1 −0.052%/噪声带 0.368% 不可判、FI 路由 1300/1300、joint 512-ctx `max_B=504` 且常规 4k 死分支 B≈122<256、§3 与 review 引用、manager 不注入开关的事实与未来独立 bundle 路径） | `knowledge/surveys/catalog/extension-catalog-hust-mods.json`（条目 `org.vllm-hust.fi-sampling`；**工作区知识库，不在版本控制内**） |
  | 2 | 裁决留痕（本小节） | `docs/evidence/fi-sampling-review.md` §9（本 commit） |
  | 3 | 目录说明同步（状态表 + 路径 B 描述 + 陈旧的 `flashinfer-migration` 路径） | `knowledge/surveys/catalog/README.md` |
  | 4 | `manifest` / `host.version_range` / 其它 bundle / `profiles/` | **未改**（裁决明确不动） |

  > 说明：§7 表中"目录侧 … **本次未改**"描述的是上一 commit（`b84e511`）时的状态；本次按本裁决
  > 完成了条目刷新（动作 1），以本小节为准。

- **未来若要"可启用"（裁决未选，仅留路径）**：按 fia-demask 先例（`docs/design/fia-decode-demask.md`
  §8：单能力独立开关）注册独立 bundle `org.vllm-hust.fi-sampling`
  （`implementation[]` + `activation.environment={"VLLM_HUST_FI_SAMPLING":"1"}`）并重跑一次纯 §3；
  但 catalog `qualified` 在缺 `installation`（不可变安装目标：wheel/源码 + sha256）时**仍不可达**，
  且性能口径不变（本裁决不因性能翻）。
