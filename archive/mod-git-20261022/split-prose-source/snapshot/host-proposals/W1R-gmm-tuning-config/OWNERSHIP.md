# W1R GMM `tuning_config` / counts dump — 归属记录（tier-2 宿主提案）

> 与 `proposal_meta.json`（机器可读）互为镜像。**当前状态（2026-10-18T07:5xZ）：P0 全门复核通过
> （`gen_patch --check --host` rc=0、仓内 `pytest -q` 614 passed、`ruff check .` clean）；
> blocker `fixture-unreachable-fullgraph-decode` 在 **FULL 路径上仍然成立**，但用户已授权 P1.0a 改走
> **cudagraph=NONE 诊断口径**（该路径无 capture，hook 可取真实 counts）→ 本层已重新加回
> `apply-runtime-patches.sh` MANIFEST 并部署到运行树（default-off）。
> NONE 窗**不作测量窗**（见 §5.2 的硬约束）。

## 1. 这是什么

一份**默认关闭的宿主提案补丁**，给宿主 `vllm_ascend/ops/fused_moe/routed_experts.py`
的两处 GMM 调用加两个 env 门控能力：

| env | 作用 | 未设时 |
|---|---|---|
| `VLLM_HUST_GMM_TUNING_CONFIG` | 单值（∈ {1,2,8,32}）→ `tuning_config=[value]` 传给 `apply_gmm1`/`apply_gmm2` | 零 import / 零 IO / 零 shape 读取 / 无设备差异（每次 GMM 仍有两次早退 Python 调用） |
| `VLLM_HUST_GMM_COUNTS_DUMP=<path>` | 记录 group_list counts + 形状/provenance 采样 | 同上，且不建文件 |

`dump` 只记录、不改变计算输出（hook 恒返回 `None`，op 结果原样返回）。
**dump 产物永远不是计时产物**；capture 内一律跳过（`sample_source` 记录原因）。

## 2. 为什么是 tier-2（PURPOSE v3 §8 档2）

1. 宿主公有面**没有** `tuning_config` 开关；
2. 两处调用点在宿主**私有方法体**内，插件复制方法体被本仓契约禁止；
3. 全局 monkeypatch `torch_npu.npu_grouped_matmul` 非 `HOST_CONTRACT.md` 已列面，且波及所有 GMM 调用方。

## 3. 哈希与基线

| 项 | 值 |
|---|---|
| 宿主基线 sha256 | `b17e5a105d59fc280f244bdc4036b4cb41efd1d5a2ff1accca4b56620cc91ab7` |
| 基线复现 | `git -C /root/vllm-workspace/vllm-ascend-hust cat-file -p 74f0c0a27:vllm_ascend/ops/fused_moe/routed_experts.py \| sha256sum` |
| `patch.diff` sha256 | `a86cee731138bc817ef59a1e6b99d16582a28791d7fac06ad0f4c6168f64b438` |
| patched（应用后）sha256 | `26fcacab198e64cce9117843892242d4abfd9a80500f3abf6c9af29fb2cc8dc4` |
| `host.version_range` | `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`（点钉） |

## 4. 责任人 / 生命周期 / 后续动作

- 责任人：主线会话 `028/D3-gdn-high-concurrency-longctx`（当前 Writer：SwarmWorker）。
- 上游状态：**未提交**（proposal snapshot）。不得表述为「上游已接受/已合入」。
- 何时可删：①上游接受并合入等价能力；②宿主升级使调用点签名/语义变化且补丁无法对齐（先重做 P0）。
- 后续动作：
  1. 提交上游 issue/PR 并回填 `proposal_meta.json.upstream_status.issue_or_pr`；
  2. **P1.0a（count dump 短跑）走 NONE 诊断口径**（2026-10-18T07:5xZ 用户授权）：
     `opt caliber switch mmlu-pro-gdn-high-concurrency-longctx-cudagraph-none-diagnostic --reason …`
     → 起服 `opt bench serve --profile light` + `--compilation-config {"cudagraph_mode":"NONE"}`
     + `VLLM_HUST_GMM_COUNTS_DUMP=<path>` → 跑 `run-prof-load.sh 32 mmlu_pro_smoke 16` →
     `opt bench stop` → 立刻切回 FULL active 口径。**全程不调用 `opt bench profile`**、
     不形成 L4 测量窗、不产出 opstat/l1/l2、不更新 CURRENT；
  3. 若上游拒绝：候选记 blocked（载体不可达），不得改判 rejected。

## 5. Blocker 与重新启用条件

- **blocker id**：`fixture-unreachable-fullgraph-decode`
- **内容**：active FULL graph 下 decode（M=128）在 capture 内执行、replay 不跑 Python；
  按 capture 防护，hook 必须在任何 device 读取前跳过 ⇒ **拿不到 M=128 真实 per-expert counts**；
  prefill（eager）样本可得，但不构成完整 fixture。
- **因此**：P0 通过 ≠ 可实验；**不消耗已授权的 NPU 预算**；不用 uniform/skew 合成夹具。
- **重新启用条件（任一 blocker 解除后）**：
  1. 出现 capture-safe 的真实 route-count 机制（路由侧在 capture 外落 counts，或 device 侧 reduce 后只读小张量）；
  2. 用户明确授权新 scenario/口径（该窗绝不作测量窗）并修订 caliber/预算；
  3. 用户明确授权合成夹具方案（须改夹具语义并另立预算）。
- **重新启用的操作步骤**：先 `gen_patch.py --check --host <blob>` 确认宿主仍为基线 →
  仓内 CPU 门 → 把 `w1r-gmm-tuning` 层重新加回 `apply-runtime-patches.sh` 的 MANIFEST →
  apply 并核对 patched sha 与 `patch -R --dry-run` → 才可启动 P1.0a。

### 5.1 已执行的重新启用记录（2026-10-18T07:5xZ，条件 2）

| 步骤 | 结果 |
|---|---|
| `gen_patch.py --check` | rc=0，`patch.diff` 未过期（sha `a86cee73…`） |
| `gen_patch.py --check --host <运行树文件>` | rc=0（宿主仍为基线 `b17e5a10…`，无漂移） |
| 仓内 CPU 门 | `pytest -q` **614 passed / 4 skipped**；`ruff check .` **clean** |
| MANIFEST 重新加入 | `w1r-gmm-tuning|ops/fused_moe/routed_experts.py|20261018T0517_…{orig,patch}` |
| 部署 | `apply-runtime-patches.sh` → `[applied]`；运行树 sha `b17e5a10…` → **`26fcacab…`**（= 期望 patched）；接口面校验 PASS（4 条） |
| 可逆 | `patch -p1 -R --dry-run` rc=0 |
| 幂等 | 再次 `--check` → `[skip] w1r-gmm-tuning: 已应用` |
| 回执 | `records/probes/20261018T075800Z_w1r_p0_recheck_deploy.json` |

### 5.2 NONE 诊断路径的硬约束（不得放宽）

1. NONE 窗只用于取 **真实 per-expert counts**（L0 夹具），**不作测量窗**、不更新 CURRENT、
   不产出 opstat/l1/l2、不调用 `opt bench profile`；
2. **禁止**与 FULL 口径做绝对数值相减/相除，**禁止**用 NONE counts 推断 FULL 图内路由分布；
3. 起服前 `opt caliber switch …-cudagraph-none-diagnostic`，动作后**必须** `opt bench stop` +
   切回 FULL active 口径；切回前禁止启动任何性能阶段；
4. FULL 下的 decode（M=128）counts 仍然不可得——该限制不因本次启用而改变；
5. 若 counts 缺形状/元数据不全，或证明改变输出 → 硬停并如实记 `blocked`。

## 6. 复现/校验命令

```bash
cd /root/vllm-workspace/vllm-ascend-split-batch-hust
# 生成器只对「未打补丁的宿主 / 镜像 blob」做 --check；部署态先 rollback 再核（见 docs/release.md §4.2）
python3 host-proposals/W1R-gmm-tuning-config/gen_patch.py --check \
  --host /vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py
python -m pytest tests/test_w1r_gmm_tuning_patch.py -q
```
