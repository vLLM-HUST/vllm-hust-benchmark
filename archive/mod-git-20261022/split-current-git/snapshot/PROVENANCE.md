# Provenance

> **核对状态（2026-09-26）**：逐条核对结论见
> [`docs/provenance-verification.md`](docs/provenance-verification.md)（含命令与原始输出）。
> 机械守卫：`tests/test_provenance_hashes.py`（vendored hash / `api.py` 逆变换复核 / 许可声明 /
> 本文件的 commit 引用）。一句话：**14 个归档提交全部可复核**（sha 存在于源仓、作者唯一、
> PR 状态与本文一致）；**vendored 4 个文件全部可复核**（其中 `api.py` 用逆变换复核，见 §4.1）；
> **1 项如实记录的缺口**：14 个提交**均无** DCO 类 trailer（见 §2）。

## 1. 源仓与历史（已核对）

| 项 | 值 | 核对结果 |
|---|---|---|
| core 源仓 | [intellistream/vllm-hust-legacy-20260831](https://github.com/intellistream/vllm-hust-legacy-20260831) | public、**archived**、license **Apache-2.0**、pushed 2026-08-31 |
| ascend 源仓 | [intellistream/vllm-ascend-hust-legacy-20260831](https://github.com/intellistream/vllm-ascend-hust-legacy-20260831) | public、**archived**、license **Apache-2.0**、pushed 2026-08-30 |

| PR | 状态（2026-09-26 复核） | merge commit | head | 提交数（=归档数） |
|---|---|---|---|---|
| core #273 | **closed + merged** | `13f11da44455` | `ab3a536aac3a` | 7 = 7 ✓ |
| core #263 | **closed + NOT merged** | —（仅测试合并值） | `13ee9f7db462` | 2 = 2 ✓ |
| ascend #280 | **open + not merged** | — | `ecb70fb47826` | 2 = 2 ✓ |
| ascend #281 | **open + not merged** | — | `e08f79d014b6` | 1 = 1 ✓ |
| ascend #282 | **open + not merged** | — | `82a8cc7dd770` | 1 = 1 ✓ |
| ascend #283 | **open + not merged** | — | `73c121d4c212` | 1 = 1 ✓ |

**Closed 不等于 merged，open 不等于 accepted。这些是迁移证据，不是发布凭据。**

## 2. 归档提交（14 个，逐条可复核）

`provenance/legacy-patches/` 保存 `git format-patch` 原始导出（**只读**，勿改）。
每个 sha 都已在对应源仓用 `GET /repos/{repo}/commits/{sha}` 复核（HTTP 200）。

| # | patch | commit（前 12） | 作者 | +/- |
|---|---|---|---|---|
| 1 | `core-pr-263/0001-828cf30d84cd` | `828cf30d84cd` | Raing5Days <M202574102@hust.edu.cn> | +268/−11 |
| 2 | `core-pr-263/0002-13ee9f7db462` | `13ee9f7db462` | 同上 | +54/−1 |
| 3 | `core-pr-273/0001-0c5f3714f4be` | `0c5f3714f4be` | 同上 | +268/−11 |
| 4 | `core-pr-273/0002-1ca673fddf05` | `1ca673fddf05` | 同上 | +54/−1 |
| 5 | `core-pr-273/0003-3f019aaec355` | `3f019aaec355` | 同上 | +315/−172 |
| 6 | `core-pr-273/0004-d351c22e18a6` | `d351c22e18a6` | 同上 | +19/−16 |
| 7 | `core-pr-273/0005-1a9eb1a14d35` | `1a9eb1a14d35` | 同上 | +346/−357 |
| 8 | `core-pr-273/0006-860c7a9657f8` | `860c7a9657f8` | 同上 | +825/−77 |
| 9 | `core-pr-273/0007-ab3a536aac3a` | `ab3a536aac3a` | 同上 | +1/−2 |
| 10 | `ascend-pr-280/0001-c46c31b550f1` | `c46c31b550f1` | 同上 | +1059/−30 |
| 11 | `ascend-pr-280/0002-ecb70fb47826` | `ecb70fb47826` | 同上 | +1/−2 |
| 12 | `ascend-pr-281/0001-e08f79d014b6` | `e08f79d014b6` | 同上 | +1317/−1 |
| 13 | `ascend-pr-282/0001-82a8cc7dd770` | `82a8cc7dd770` | 同上 | +3404/−149 |
| 14 | `ascend-pr-283/0001-73c121d4c212` | `73c121d4c212` | 同上 | +4333/−3 |

- 合计 **+12264/−833**，触及 **36 个不同路径**（清单见核对记录 §2）。
- **作者**：14 个提交的 `From:` 均为 `Raing5Days <M202574102@hust.edu.cn>`；源仓侧
  commit author 元数据同名。仓库内主维护者登记见 `MAINTAINERS.md`。
- **trailers**：14 个提交**均无** `Signed-off-by` / `Co-authored-by` / `Change-Id`
  —— 没有 DCO 签署证据。这不阻塞归档（原样留档），但**不能**据此宣称签核链完整。
- **版权头**：core 侧 patch 只改既有文件；ascend 侧 #281/#282/#283 的 patch 自身含
  Apache-2.0 头。本仓抽取代码的许可覆盖由 `tests/test_provenance_hashes.py` 机械守住。

## 3. 可迁移范围的判定

**结论：这批 legacy patch 一律不移植（工作区决策），只作为技术论据与缺陷定位参考。**
本仓当前代码**不是**这些 patch 的搬运，而是按宿主公开面重写；映射关系如下（按模块标注）：

| 本仓模块 | 对应 legacy 面 | 关系 |
|---|---|---|
| `planner.py`（`plan_dual_pad` / `precheck_reason`） | Ascend #281 `worker/dual_pad_utils.py`、`tests/test_dual_pad_planner.py` | **语义参照**，按宿主无关重写；`import_only`，无执行面 |
| `cascade_plugin.py` / `cascade_graph_plugin.py` / `cascade_runner_patch.py` | Ascend #282 的 `attention_v1.py` / `compilation/acl_graph*.py`、core #273 的 `forward_context.py` / `v1/cudagraph_dispatcher.py` | **行为参照**（两段式 cascade、图孪生、cudagraph key）；插件侧重写，宿主源码零改动 |
| `fia_demask_plugin.py`、`rope_fix_plugin.py`、`zerocost_wiring.py`、`mlp_chunk_plugin.py`、`fi_gelu*`、`fi_sampling/` | 无对应 patch | **独立开发**，不是 legacy 搬运 |
| legacy 的 `inplace_split_*` 系列、`csrc/third_party/catlass` 子模块 pin、`examples/**` | Ascend #280/#282/#283 | **未迁移**（inplace 车道未进入本仓） |

> 判定依据：legacy patch 中 **36 个路径有 32 个落在宿主树内**（`vllm/` 与 `vllm_ascend/`），
> 而本仓硬约束是"不修改宿主源码"（见 `AGENTS.md`）；`csrc/third_party/catlass` 是子模块 pin，
> 与插件无关；`examples/**` 是演示脚本。故可迁移面只剩"策略/契约/预检"这类宿主无关部分，
> 已在上表按模块落到本仓，其余留档不迁移。

## 4. Vendored `fi_sampling` host API（`src/vllm_ascend_split_batch/fi_sampling/`）

- 算法锚：flashinfer `main@3a4e7052`（`include/flashinfer/sampling.cuh`、`flashinfer/sampling.py`，
  Apache-2.0，Copyright (c) 2023-2025 FlashInfer team）。
- 直接来源（被搬运字节的来源）：W2 `fi_sampling` 包，2026-09-09 评审通过；
  报告 `knowledge/surveys/sampling/报告-fi_sampling-w2-w3-2026-09-09.md`（191 PASS / 0 FAIL；
  micro-bench V=152064）。
- attribution 说明（含"为何 vendored 文件内无版权头"）：见
  [`src/vllm_ascend_split_batch/fi_sampling/ATTRIBUTION.md`](src/vllm_ascend_split_batch/fi_sampling/ATTRIBUTION.md)。

### 4.1 hash 复核（2026-09-26 实测）

| vendored 文件 | vendored sha256 | 源侧 sha256（记录值） | 复核 |
|---|---|---|---|
| `api.py` | `ca8dc05cfc8906d7ef7992aa6f872e732277c89432b58a6471e6ffd647dc7446` | `902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c` | ✓ **逆变换复核命中** |
| `kernels.py` | `8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e` | 同左 | ✓ 与 vendored 一致 |
| `npu_env.py` | `04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a` | 同左 | ✓ 与 vendored 一致 |
| `pure.py` | `923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08` | 同左 | ✓ 与 vendored 一致 |

**`api.py` 为什么不能直接比对、以及它怎么复核**：

1. `api.py` 是四个文件里**唯一**含 `from fi_sampling ...` 导入的文件（`grep` 实测）⇒
   搬运动了那两行（改成相对导入），所以 vendored 字节**必然**不等于源侧字节，
   源侧 hash 无法用"直接 sha256 源文件"复核。
2. 但改动是**可逆且已知**的 ⇒ 把 vendored `api.py` 的两条相对导入还原成绝对导入后，
   其 sha256 **精确命中**记录值 `902884d6…`（2026-09-26 实测）。这一命中同时证明了两件事：
   **记录值为真**，且 **api.py 的搬运改动只有那两行**（没有夹带其它编辑、没有把 W3 代码带进来）。
3. 可执行版本：`tests/test_provenance_hashes.py::test_api_py_source_hash_is_recoverable`。
4. **活源包已漂移**（2026-09-09 之后加了 W3 renorm/mask 家族；实测 api.py 现为 `4e3ce156…`、
   pure.py 现为 `8a8bba61…`，与 vendored 相差 91/112 行），所以：
   **"与源包 diff 仍有意义"只在对照 W2 冻结快照时成立**；对照今天的活源包会把 W3 的新增
   算成"搬运差异"。复核请用上表的 hash + 逆变换，不要用裸 `diff`。
5. 本表同时记录 **vendored 侧** hash：它是"没人碰过这些字节"的收据（由上面的测试机械守住）。

> 订正记录：本轮初判曾写 `api.py` 源侧 hash 为 `NOT RE-VERIFIABLE`（推断：源已漂移 ⇒ 无从复核）。
> 随后用逆变换实测命中，故改为"可复核"，并把方法写进测试。留此一行以便追溯判断过程。

### 4.2 搬运改动与边界

- 改动：`from fi_sampling.x import ...` → 相对导入；`__init__.py` 为本仓新写（带项目版权头）。
  **无算法改动**。
- `reference.py`、`page.py` 与 test/bench harness **不搬运**；统计等价性证据的权威仍在源包。
- `src/vllm_ascend_split_batch/fi_sampling` 排除在 ruff 行宽强制之外（源包按 ruff 默认行宽，
  非 88），理由同前：字节一致性优先。

## 5. 复核方式（复现）

```bash
# 归档自身的一致性 + 上游状态 + vendored hash + 许可覆盖（只读，联网部分可跳过）
python3 docs/provenance-verification.md 中的命令清单
pytest -q tests/test_provenance_hashes.py     # 机械守卫（离线）
```

- 源侧 hash 若要"直接比对"复核，用**已入仓的冻结快照** `frozen-20260909`
  （`docs/evidence/fi-sampling-frozen-20260909/`）：`cd` 该目录后 `sha256sum -c FROZEN.sha256`
  即得 4 行 OK，**不需要**自己推逆变换。快照 id、逐件记录值与边界见该目录 `README.md`。
- **"打 tag 冻结"已被实测排除**（2026-09-26）：W2 源包是活工作区，且工作区仓的 git 历史里
  没有那一版（`git log --format=%H -- knowledge/surveys/sampling/fi_sampling/api.py` 只返回
  `9e98559`，其内容即漂移后的 `4e3ce156…`）⇒ 快照只能由 §4.1 的逆变换重建，这也是
  `docs/evidence/fi-sampling-frozen-20260909/make_snapshot.py` 的做法（重建命中不了记录值即拒绝写入）。
- §4.1 的逆变换法**仍然有效**，是快照的来源与独立复核路径（
  `tests/test_provenance_hashes.py::test_api_py_source_hash_is_recoverable`）。
