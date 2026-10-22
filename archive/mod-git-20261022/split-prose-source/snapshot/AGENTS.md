# AGENTS.md — vllm-ascend-split-batch-hust

vllm-ascend 宿主的 default-off 插件壳:cascade 两段式 decode(eager+图模式)
与 split-batch 规划契约。

**算子落点按形态分(2026-10-13 明确)**:
- **CCE / Ascend C / CATLASS(编译成 `.so`)**:在**仓库外** `ascend_kernel`
  wheel(`vllm-hust-cascade-kernel`),本壳经 `[kernels]` extra 单向软依赖,
  不在本仓引入 C/C++ kernel 构建。
- **Triton / 纯 Python kernel 载体**:**在仓内**。形态=default-off 插件,
  靠 `vllm.general_plugins` 入口点注入;`triton` 必须 lazy import(关闭时零
  import)。先例:W3.1 `fi_gelu`(gelu_and_mul triton 融合)、
  `fi_sampling/kernels.py`。宿主只读的 `vllm_ascend/ops/triton/` 不得改。

## 硬约束

1. 宿主 `/vllm-workspace/vllm-hust`、`/vllm-workspace/vllm-ascend-hust` **只读**;
   宿主能力只走官方公开面(general_plugins entry / `--additional-config` /
   `--worker-cls` / 官方 CLI+env)。不注册 `vllm.platform_plugins`。
   (旧基线 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend` 已于 2026-09-10 删除。)
2. default-off:新能力必须 env-gated 且默认与原生行为零差异。
3. bundle 保持 `import_only` 直到三项证据齐(default-off 冒烟/正确性/性能);
   `host.version_range` 禁止写 `>=0`。
4. `provenance/` 只读;提取代码保留 Huawei 版权头与溯源。
5. **不要以 `/vllm-workspace` 为 CWD 跑任何 python/vllm 命令**(目录会遮蔽
   真实 vllm 包)。
6. **注释**:代码只写必须写的注释,不写解释性段落/改动沿革/"为什么这样改";需要解释的写进提交信息或 `docs/`。文档只写约束与判据,不写沿革。
7. **可比性口径**:两组数字可比只看**算子形状事实**(M/K/N、专家 counts、相位、callsite)+**权重标签**(model/precision/权重格式);不同则只比方向。
8. **cohort 与归属(2026-09-28/29 上游裁定,长期核心约束)**——详见
   [docs/cohort-and-ownership.md](docs/cohort-and-ownership.md):
   - 中央团队只保留"分类与发布边界";**下列四项由本仓 owner 负责**:
     ① independent cohort definition;② kernel-wheel licensing/distribution;
     ③ NPU evidence;④ performance claims。
   - 任何正确性/性能声明**必须带 cohort 边界**(2026-10-13 基准切换):
     **0.28 镜像栈**(vllm `0.28.1.post1.dev143+gf18cf803c` + vllm-ascend
     `0.25.1rc2.dev125+hust` 构建, CANN 9.1.0, 2×910B2 TP=2) + **Qwen3.5-35B-A3B**
     hybrid TP2 + chunked prefill on / APC off / `cudagraph_mode=NONE`。
     **不得跨域外推**:Qwen2.5/0.25 的历史证据(独立专项)不适用于该基准,反之亦然。
   - MTP 是**边界不是支持**:任何 `speculative_config` fail-closed;该配置下
     "开 cascade"跑的是原生路径,不构成 cascade 效果。
   - "进统一 cohort"需先有 k+1 verification rows 的 dual-pad/graph bucket 合同
     + 该配置真实 replay correctness;**当前未立项**,未经裁定不得绕过上述守卫。

## 算子交付检查点

- 本仓是交付目标，不是实验附件。开工记录分支、起点 HEAD、本轮路径和已有 dirty 文件；任务句与动作约束见 `/root/AGENTS.md`「算子任务执行合同」。
- 分支口径：`main` 只承接"可发布且 Extension CI 三条腿全绿"的改动，开发默认留在 `版本/方向` 分支。判据是这次改动动的是哪一侧：只影响"能不能验证"的（CI/测试基建、只读文档与证据、让红腿转绿的修复）可直推 `main`；影响"被验证的东西是什么"的（载体实现、注册面 entry points、默认值或行为变更、host proposal、任何改 kernel/数值的提交）必须走分支后合入。合入前先同步 `origin/main`；`main` 已启用可信 benchmark-only 必需检查；全部 CI 通过后走 PR 合并，不绕过保护。
- 已推送的 `版本/方向` 分支用 `merge origin/main` 同步，不改写历史：分支 commit sha 被 [docs/evidence/](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/) 与 L2 回执引用，`rebase` 会让这些引用指空。
- 算子默认执行 L0 → plain L3 + compare → light L4；不强制 L1/L2 screen。端到端问题显式跑完整 screen；无 screen 的采集条件以 `/root/opt-ops/AGENTS.md` 为准。历史报告保留，不为新流程重采旧数据。
- 最小实现可测后、完整验证或明确阻塞后，各检查一次是否应提交。实现、测试、接入与回滚材料留本仓；结果原件、报告和 trace 统一归档 benchmark，本仓保留固定引用、解析入口和哈希，见 [EVIDENCE.md](EVIDENCE.md)。
- 数值失败保持 default-off，失败复现和证据同样归档 benchmark；不得将失败项从报告删除，或将局部收益写成已验收。Git 提交与优化验收是两个状态。
- 先跑测试，再审查 `git diff` 与 staged diff，仅 `git add <本轮明确路径>`。不提交无关 dirty 文件，不改正在执行的脚本。
- commit 后用 `opt delivery --base <起点HEAD> --path <本轮路径> ... --check` 核验；工作台未部署时可用 `python3 tools/operator_delivery.py` 的相同参数。无范围不算通过，push 另需授权及远端 hash 核验。
- `/root/opt-ops` 的流程修改必须有本仓可校验补丁及测试；运行树仅为部署副本。

## 常用命令

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
pytest -q && ruff check .        # CPU 层门槛,提交前必过
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
```

## 知识库

规范、架构、发布、踩坑、进度全部在 [docs/README.md](docs/README.md),
按其阅读顺序走;改行为必须同步对应权威文档(分工表在 docs/README.md)。
