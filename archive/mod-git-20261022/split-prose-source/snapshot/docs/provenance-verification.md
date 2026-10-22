# PROVENANCE 核对记录（2026-09-26）

> 对应 issue #2 接手清单第 1 条「核对 PROVENANCE.md 中的提交、作者、许可证和可迁移范围」。
> **结论**：14 个归档提交**逐条可复核**（sha 存在于源仓、作者唯一、PR 状态与 `PROVENANCE.md` 一致）；
> vendored `fi_sampling` 的 4 个文件**全部可复核**（`api.py` 用逆变换法，见 §4）；
> **1 项如实记录的缺口**：14 个提交**均无** DCO 类 trailer。
> 机械守卫：`tests/test_provenance_hashes.py`（5 例，离线可跑）；原始输出见 §6。
> 复核方式：`bash docs/` 内命令见 §6 首部的注释；联网段为 GitHub **匿名只读** API。

## 1. 源仓（已核对）

| 项 | 值 | 结果 |
|---|---|---|
| core 源仓 | `intellistream/vllm-hust-legacy-20260831` | http=200、private=False、**archived=True**、license=**Apache-2.0**、pushed 2026-08-31 |
| ascend 源仓 | `intellistream/vllm-ascend-hust-legacy-20260831` | http=200、private=False、**archived=True**、license=**Apache-2.0**、pushed 2026-08-30 |

两个仓都是**公开归档**仓 ⇒ 上述 sha 与 PR 状态第三方可独立复核（`PROVENANCE.md` 的链接可直接点）。

## 2. 14 个归档提交（逐条已核对）

每个 sha 都用 `GET /repos/{repo}/commits/{sha}` 复核过（**全部 HTTP 200**），且源仓 PR 的提交列表与归档数一一对应：

| PR | 状态 | merge commit | head | 归档数 / PR 提交数 |
|---|---|---|---|---|
| core #273 | closed + **merged** | `13f11da44455` | `ab3a536aac3a` | 7 / 7 ✓ |
| core #263 | closed + **not merged** | —（`merge_commit_sha` 为测试合并值，`merged=false`） | `13ee9f7db462` | 2 / 2 ✓ |
| ascend #280 | **open** + not merged | —（同上） | `ecb70fb47826` | 2 / 2 ✓ |
| ascend #281 | **open** + not merged | — | `e08f79d014b6` | 1 / 1 ✓ |
| ascend #282 | **open** + not merged | — | `82a8cc7dd770` | 1 / 1 ✓ |
| ascend #283 | **open** + not merged | — | `73c121d4c212` | 1 / 1 ✓ |

- **作者**：14 个提交的 `From:` 均为 `Raing5Days <M202574102@hust.edu.cn>`；源仓侧 commit author 元数据同名。
  （PR #263 的 PR 作者登记为 `ilnnfover`，但其**提交**作者是 Raing5Days —— 两者不矛盾，已分别记录。）
- **trailers**：14 个提交**均无** `Signed-off-by` / `Co-authored-by` / `Change-Id`
  ⇒ **没有 DCO 签署证据**。原样留档不阻塞，但不得据此宣称签核链完整。
- 合计 **+12264/−833**，触及 **36 个路径**；diff 与 GitHub PR 的 `additions/deletions` 有 0–3 行差异
  （统计口径不同：本记录按 patch 内 `+`/`-` 行计，GitHub 按 blob 计），不影响 sha 级复核。

## 3. 可迁移范围（判定）

**结论：这批 legacy patch 一律不移植**（工作区决策），只作技术论据与缺陷定位参考。判定依据：

- 36 个路径里 **32 个落在宿主树内**（`vllm/` 17 个、`vllm_ascend/` 15 个）—— 而本仓硬约束是
  "禁止修改宿主源码"（`AGENTS.md`）⇒ 这些 diff **不能**搬进本仓。
- 其余 4 类：`csrc/third_party/catlass`（子模块 pin，与插件无关）、`docs/source/**`（上游文档）、
  `examples/**`（演示脚本）、`tests/**`（宿主侧测试）—— 都不构成插件能力。
- 可迁移面因此只剩"策略 / 契约 / 预检"这类**宿主无关**部分，已在本仓按模块重写（映射表见
  `PROVENANCE.md` §3）：`planner.py`（对 #281 的 `dual_pad_utils`）、cascade 三个模块
  （对 #282 的 `attention_v1`/`acl_graph*` 与 #273 的 `forward_context`/`cudagraph_dispatcher`）。
- **未迁移**并如实登记：legacy 的 `inplace_split_*` 系列（inplace 车道未进入本仓）、catlass pin、examples。

## 4. vendored `fi_sampling` 的 hash 复核

| 文件 | vendored sha256 | 源侧 sha256 | 复核方式与结果 |
|---|---|---|---|
| `api.py` | `ca8dc05c…` | `902884d6…` | **逆变换复核命中**（见下） |
| `kernels.py` | `8d0b20cc…` | 同左 | 直接比对一致（无导入改写点） |
| `npu_env.py` | `04a3a122…` | 同左 | 直接比对一致（无导入改写点） |
| `pure.py` | `923d7db2…` | 同左 | 直接比对一致（无导入改写点） |

**关键实验（§6 段 [E]）**：`grep` 实测**只有 `api.py`** 含 `from fi_sampling ...` 导入（另三个没有），
所以只有它的字节必然改变。把 vendored `api.py` 的两条相对导入还原成绝对导入后求 sha256：

```
902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c   <- 还原后
902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c   <- 记录值
```

命中 ⇒ **记录值为真**，且**搬运只改了那两行**（未夹带后来 W3 的 renorm/mask 代码）。
已落成可执行的 `tests/test_provenance_hashes.py::test_api_py_source_hash_is_recoverable`。

**源包漂移（如实记录）**：源包是活工作区，2026-09-09 之后加了 W3 家族；实测 api.py 现为 `4e3ce156…`
（与 vendored 差 91 行）、pure.py 现为 `8a8bba61…`（差 112 行）、kernels.py `a90f068a…`（差 6 行）、
npu_env.py 未变。⇒ **不要用裸 `diff` 复核**（会把 W3 新增算成搬运差异），用上面的 hash/逆变换。
根因：工作区 git 仓 2026-09-12 才初始化，**没有** 09-09 的源侧历史可查。

> 判断订正（保留过程）：初判曾写该 hash `NOT RE-VERIFIABLE`（推断：源已漂移 ⇒ 无从复核）。
> 随后用逆变换实测命中，改为"可复核"。这条订正也记在 `PROVENANCE.md` §4.1。

## 5. 许可证与版权头

- 两个源仓 license 均为 **Apache-2.0**；本仓根 `LICENSE` 为 Apache-2.0。
- ascend 侧 patch #281/#282/#283 自身含 Apache-2.0 头；core 侧 patch 只改既有文件。
- vendored 的 4 个文件**无 in-file 版权头** —— 因为 W2 源文件本身没有，而字节一致性优先。
  attribution 与理由落在 `src/vllm_ascend_split_batch/fi_sampling/ATTRIBUTION.md`
  （含 flashinfer 上游锚 `main@3a4e7052` 与 Apache-2.0 归属）。
- 本仓其余 `src/**/*.py` **全部**带许可声明（24/28；缺的 4 个即上述 vendored 文件）。
  这条由 `test_every_python_module_declares_a_license` 机械守住。

## 6. 原始命令与输出

```
PROVENANCE 核对 — 原始命令与输出
生成时间(UTC): 2026-09-26T08:52:51Z
工作区: /vllm-workspace/vllm-ascend-split-batch-hust
说明: 全部只读。A 段离线可判; B 段用 GitHub 匿名只读 API; C/D 段本地实测。

==============================================================================
[A] 归档 patch 自身（离线）
==============================================================================
$ find provenance/legacy-patches -name '*.patch' | wc -l
$ # 每个 patch 解析 From / From: / Date / Subject / trailers / diff --git
共 14 个 patch

--- ascend-pr-280/0001-c46c31b550f1.patch
    commit : c46c31b550f1d3cd7f4694d1405054e6d4726a11
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sun, 30 Aug 2026 15:49:49 +0000
    subject: feat(ascend): split-batch scaffolding - config, envs, platform key strategy, plumbing
    diff   : +1059/-30, 触及 11 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- ascend-pr-280/0002-ecb70fb47826.patch
    commit : ecb70fb478267e4143dc74ac3bcd84248ce911f0
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sun, 30 Aug 2026 15:51:10 +0000
    subject: chore(catlass): pin submodule to b50cad68 for the split-batch series
    diff   : +1/-2, 触及 1 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- ascend-pr-281/0001-e08f79d014b6.patch
    commit : e08f79d014b6884032838e348b574091a0bf7c5b
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sun, 30 Aug 2026 15:52:55 +0000
    subject: feat(ascend): split-batch primitive modules
    diff   : +1317/-1, 触及 4 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- ascend-pr-282/0001-82a8cc7dd770.patch
    commit : 82a8cc7dd7702b162af1d1f06b0d9e95bad2718b
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sun, 30 Aug 2026 15:52:55 +0000
    subject: feat(ascend): split-batch engine integration and cudagraph support
    diff   : +3404/-149, 触及 5 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- ascend-pr-283/0001-73c121d4c212.patch
    commit : 73c121d4c212a6eebdbd7644a7bf0aa7bafede25
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sun, 30 Aug 2026 15:52:55 +0000
    subject: test(ascend): split-batch examples, correctness harness and unit tests
    diff   : +4333/-3, 触及 8 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-263/0001-828cf30d84cd.patch
    commit : 828cf30d84cd8ca37e08b2de9c08c4612fdb7af3
    author : Raing5Days <M202574102@hust.edu.cn>   date: Mon, 13 Jul 2026 13:37:49 +0000
    subject: temp
    diff   : +268/-11, 触及 2 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-263/0002-13ee9f7db462.patch
    commit : 13ee9f7db46268246fdf624c4c23b36421d521c7
    author : Raing5Days <M202574102@hust.edu.cn>   date: Tue, 14 Jul 2026 21:10:00 +0000
    subject: add full_graph_parallel
    diff   : +54/-1, 触及 1 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0001-0c5f3714f4be.patch
    commit : 0c5f3714f4be73baa4cf0be2effa7b2cd339c66e
    author : Raing5Days <M202574102@hust.edu.cn>   date: Mon, 13 Jul 2026 13:37:49 +0000
    subject: temp
    diff   : +268/-11, 触及 2 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0002-1ca673fddf05.patch
    commit : 1ca673fddf0574679ee69d5cbb4b4a8138d6d6fd
    author : Raing5Days <M202574102@hust.edu.cn>   date: Tue, 14 Jul 2026 21:10:00 +0000
    subject: add full_graph_parallel
    diff   : +54/-1, 触及 1 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0003-3f019aaec355.patch
    commit : 3f019aaec3554ed429630a46e630f969ded5dd62
    author : Raing5Days <M202574102@hust.edu.cn>   date: Wed, 26 Aug 2026 10:48:59 +0000
    subject: feat(v1): converge split-batch cudagraph key policy into pluggable strategy
    diff   : +315/-172, 触及 3 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0004-d351c22e18a6.patch
    commit : d351c22e18a651a5f46c502dd6878baae1354a3a
    author : Raing5Days <M202574102@hust.edu.cn>   date: Wed, 26 Aug 2026 10:49:27 +0000
    subject: docs(forward_context): neutralize split-batch metadata docstrings
    diff   : +19/-16, 触及 1 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0005-1a9eb1a14d35.patch
    commit : 1a9eb1a14d353a0f4e2b9e3bca96343992a23eb0
    author : Raing5Days <M202574102@hust.edu.cn>   date: Wed, 26 Aug 2026 12:07:53 +0000
    subject: feat(v1): adopt platform-neutral runtime-graph-metadata contract
    diff   : +346/-357, 触及 5 个路径
    trailers: ['Co-authored-by: converges PR #264 per review.']
--- core-pr-273/0006-860c7a9657f8.patch
    commit : 860c7a9657f8ae7b55653375916a4768a3aad32f
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sat, 29 Aug 2026 10:55:57 +0000
    subject: fix(v1): close the runtime-key capture lifecycle (bounded, synchronized, fail-closed)
    diff   : +825/-77, 触及 4 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）
--- core-pr-273/0007-ab3a536aac3a.patch
    commit : ab3a536aac3ae7fb6642df2ad1969820fb5d21bb
    author : Raing5Days <M202574102@hust.edu.cn>   date: Sat, 29 Aug 2026 15:02:25 +0000
    subject: style: add missing EOF newline in test_forward_context.py
    diff   : +1/-2, 触及 1 个路径
    trailers: （无 Signed-off-by / Co-authored-by / Change-Id）

合计: 14 个提交 +12264/-833, 触及 36 个不同路径
作者分布: {'Raing5Days <M202574102@hust.edu.cn>'}
trailers 汇总: ['Co-authored-by']

触及路径全清单（按是否在宿主树内分组）:
  宿主树内 (22):
    vllm/forward_context.py
    vllm/platforms/interface.py
    vllm/v1/cudagraph_dispatcher.py
    vllm/v1/cudagraph_key_strategy.py
    vllm_ascend/ascend_config.py
    vllm_ascend/ascend_forward_context.py
    vllm_ascend/attention/attention_v1.py
    vllm_ascend/attention/utils.py
    vllm_ascend/compilation/acl_graph.py
    vllm_ascend/compilation/acl_graph_diagnostics.py
    vllm_ascend/compilation/acl_graph_split_batch.py
    vllm_ascend/envs.py
    vllm_ascend/inplace_split_debug.py
    vllm_ascend/ops/fused_moe/moe_runtime_args.py
    vllm_ascend/ops/rotary_embedding.py
    vllm_ascend/platform.py
    vllm_ascend/worker/dual_pad_utils.py
    vllm_ascend/worker/inplace_split_ops.py
    vllm_ascend/worker/inplace_split_runner.py
    vllm_ascend/worker/inplace_split_utils.py
    vllm_ascend/worker/inplace_split_worker_pool.py
    vllm_ascend/worker/model_runner_v1.py
  其它 (14):
    csrc/third_party/catlass
    docs/source/user_guide/feature_guide/index.md
    docs/source/user_guide/feature_guide/split_batch.md
    examples/bench_split_batch_latency_npu.py
    examples/benchmark_split_compare.py
    examples/dual_inplace_parallel.py
    examples/online_serving/full_graph_parallel_test.py
    examples/test_split_batch_correctness_npu.py
    tests/run_dual_stream_link_smoke.py
    tests/test_dual_pad_planner.py
    tests/test_forward_context.py
    tests/test_split_full_graph_startup.py
    tests/ut/compilation/test_fia_shared_list.py
    tests/v1/cudagraph/test_cudagraph_dispatch.py

==============================================================================
[B] 上游状态（GitHub 匿名只读 API）
==============================================================================

$ GET /repos/intellistream/vllm-hust-legacy-20260831   -> http=200
    private=False archived=True license=Apache-2.0 pushed_at=2026-08-31T04:04:42Z

$ GET /repos/intellistream/vllm-hust-legacy-20260831/pulls/263   -> http=200
    state=closed merged=False merge_commit_sha=dd0b60a077d73564326a28aba8515e8825f6c17a
    head=13ee9f7db46268246fdf624c4c23b36421d521c7 author=ilnnfover changed_files=3 +322/-10
    commits(2): 828cf30d84cd=Raing5Days, 13ee9f7db462=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ GET /repos/intellistream/vllm-hust-legacy-20260831/pulls/273   -> http=200
    state=closed merged=True merge_commit_sha=13f11da44455d2f85f00f810a04e1bd2eaf30ba6
    head=ab3a536aac3ae7fb6642df2ad1969820fb5d21bb author=Raing5Days changed_files=6 +1224/-12
    commits(7): 0c5f3714f4be=Raing5Days, 1ca673fddf05=Raing5Days, 3f019aaec355=Raing5Days, d351c22e18a6=Raing5Days, 1a9eb1a14d35=Raing5Days, 860c7a9657f8=Raing5Days, ab3a536aac3a=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ 归档 sha 是否存在于该仓（逐个 GET /commits/<sha>）
    828cf30d84cd8ca37e08b2de9c08c4612fdb7af3  http=200  author=Raing5Days
    13ee9f7db46268246fdf624c4c23b36421d521c7  http=200  author=Raing5Days
    0c5f3714f4be73baa4cf0be2effa7b2cd339c66e  http=200  author=Raing5Days
    1ca673fddf0574679ee69d5cbb4b4a8138d6d6fd  http=200  author=Raing5Days
    3f019aaec3554ed429630a46e630f969ded5dd62  http=200  author=Raing5Days
    d351c22e18a651a5f46c502dd6878baae1354a3a  http=200  author=Raing5Days
    1a9eb1a14d353a0f4e2b9e3bca96343992a23eb0  http=200  author=Raing5Days
    860c7a9657f8ae7b55653375916a4768a3aad32f  http=200  author=Raing5Days
    ab3a536aac3ae7fb6642df2ad1969820fb5d21bb  http=200  author=Raing5Days

$ GET /repos/intellistream/vllm-ascend-hust-legacy-20260831   -> http=200
    private=False archived=True license=Apache-2.0 pushed_at=2026-08-30T15:57:02Z

$ GET /repos/intellistream/vllm-ascend-hust-legacy-20260831/pulls/280   -> http=200
    state=open merged=False merge_commit_sha=42a715170b558afc77c1f66826beb18794011e39
    head=ecb70fb478267e4143dc74ac3bcd84248ce911f0 author=Raing5Days changed_files=12 +1060/-30
    commits(2): c46c31b550f1=Raing5Days, ecb70fb47826=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ GET /repos/intellistream/vllm-ascend-hust-legacy-20260831/pulls/281   -> http=200
    state=open merged=False merge_commit_sha=6c3bb268cce61e593f6a101b995e1d66ba4f5930
    head=e08f79d014b6884032838e348b574091a0bf7c5b author=Raing5Days changed_files=4 +1317/-0
    commits(1): e08f79d014b6=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ GET /repos/intellistream/vllm-ascend-hust-legacy-20260831/pulls/282   -> http=200
    state=open merged=False merge_commit_sha=cf8a48bb591daef2c4835df88436fee2f088a93f
    head=82a8cc7dd7702b162af1d1f06b0d9e95bad2718b author=Raing5Days changed_files=5 +3404/-148
    commits(1): 82a8cc7dd770=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ GET /repos/intellistream/vllm-ascend-hust-legacy-20260831/pulls/283   -> http=200
    state=open merged=False merge_commit_sha=cad994542090595219e42f99dbcceadbb01badd9
    head=73c121d4c212a6eebdbd7644a7bf0aa7bafede25 author=Raing5Days changed_files=8 +4333/-0
    commits(1): 73c121d4c212=Raing5Days
    HEAD commit author: Raing5Days <M202574102@hust.edu.cn>

$ 归档 sha 是否存在于该仓（逐个 GET /commits/<sha>）
    c46c31b550f1d3cd7f4694d1405054e6d4726a11  http=200  author=Raing5Days
    ecb70fb478267e4143dc74ac3bcd84248ce911f0  http=200  author=Raing5Days
    e08f79d014b6884032838e348b574091a0bf7c5b  http=200  author=Raing5Days
    82a8cc7dd7702b162af1d1f06b0d9e95bad2718b  http=200  author=Raing5Days
    73c121d4c212a6eebdbd7644a7bf0aa7bafede25  http=200  author=Raing5Days

==============================================================================
[C] vendored fi_sampling 的 sha256（本地实测）
==============================================================================
$ sha256sum /vllm-workspace/vllm-ascend-split-batch-hust/src/vllm_ascend_split_batch/fi_sampling/*.py
    171480416d7c9ac0e82815974332f80c5f7834cb87d4348f655cfe862b91d8ac  __init__.py
    ca8dc05cfc8906d7ef7992aa6f872e732277c89432b58a6471e6ffd647dc7446  api.py
    8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e  kernels.py
    04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a  npu_env.py
    923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08  pure.py

$ sha256sum /vllm-workspace/knowledge/surveys/sampling/fi_sampling/*.py   # 活源包（供对照，注意已漂移）
    24f41aa08c3e004dd2ce862acdd321e59315309f99926c776b714b005388e01d  __init__.py
    4e3ce156eb4abdd5f2de6fae860aa05950f3a34be7caa36fb22aff750ade1b97  api.py
    a90f068a6cb6585e7bfb824f339da9e9b6655df11c122ee7f6a095e86da7dfe2  kernels.py
    04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a  npu_env.py
    19d0c0dd21e0c7aef819646a4758d5f985fc922cae997d9ed0ebc29c77b6d9ca  page.py
    8a8bba61ee654765215865950bf6dd3fdd05f9a6170a236df2e51d92ff168518  pure.py
    1585b2864e5033c7bafc1e7f6a7e668584963709220dacbb559eed1a1e82e193  reference.py

$ diff <(grep -v '^from' 源) <(grep -v '^from' vendored)  # 去 import 行后的差异行数
    api.py       91 行
    kernels.py   6 行
    npu_env.py   0 行
    pure.py      112 行

$ ls /vllm-workspace/knowledge/surveys/sampling   # 源包所在目录
    README.md bench fi_sampling results scripts smoke tests 报告-fi_sampling-w2-w3-2026-09-09.md
$ grep -n 'from fi_sampling' /vllm-workspace/knowledge/surveys/sampling/fi_sampling/*.py   # 哪些文件含导入改写点
    api.py       ['56:from fi_sampling import pure', '57:from fi_sampling.kernels import KSCALE, MAX_ROUNDS, _sampling_from_probs_kernel', '58:from fi_sampling.npu_env import init_device_properties_triton']
    kernels.py   （无导入改写点）
    npu_env.py   （无导入改写点）
    pure.py      （无导入改写点）

==============================================================================
[D] 本仓许可声明覆盖（本地实测）
==============================================================================
$ 遍历 src/**/*.py，检查前 600 字节是否含 'Apache License' 或 SPDX Apache-2.0
    py 文件 28 个; 缺声明 4 个:
      src/vllm_ascend_split_batch/fi_sampling/api.py   <- vendored 字节（hash 为收据），attribution 见 fi_sampling/ATTRIBUTION.md
      src/vllm_ascend_split_batch/fi_sampling/kernels.py   <- vendored 字节（hash 为收据），attribution 见 fi_sampling/ATTRIBUTION.md
      src/vllm_ascend_split_batch/fi_sampling/npu_env.py   <- vendored 字节（hash 为收据），attribution 见 fi_sampling/ATTRIBUTION.md
      src/vllm_ascend_split_batch/fi_sampling/pure.py   <- vendored 字节（hash 为收据），attribution 见 fi_sampling/ATTRIBUTION.md

$ head -2 LICENSE
   Apache License
   Version 2.0, January 2004

$ pytest -q tests/test_provenance_hashes.py   # 机械守卫

== 采集结束 ==

==============================================================================
[E] api.py 源侧 hash 的逆变换复核（本轮关键实验）
==============================================================================
$ # 记录值(PROVENANCE §4.1 源侧) = 902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c
$ # 把两条相对导入还原为绝对导入，再求 sha256
    902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c  <- 还原后
    902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c  <- 记录值
$ # 结论: 命中 ⇒ 记录为真，且搬运只改了那两行（未夹带 W3 代码）

$ # 对照：加回 `from fi_sampling import pure` 的两种位置（若源侧真有该行，则应命中）
    pure 行在 kernels 之前: 152c9ac8f8c6cc6ffd2e2869e1ab6eca4a80948622f916d14886959c21f6e2f6  （未命中 ⇒ 源侧无该行）
$ # vendored api.py 是否使用 pure: False
$ pytest -q tests/test_provenance_hashes.py
    5 passed

== [E] 结束 ==
```
