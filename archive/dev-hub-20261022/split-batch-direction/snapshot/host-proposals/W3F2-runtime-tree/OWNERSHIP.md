# W3F2 / W3F3 runtime-tree patches —— 归属记录

## 最终归类（2026-10-20 用户裁定：镜像定死，不走上游提案/issue）

W3F2 按 `tree-patch-rehoming` stop_condition ① 登记为**长期例外**：
改动必须落在宿主私有面（定义点+rebind+runner-hook），宿主是不可修改镜像且无
merged-consumption seam，插件载体已判 blocked（`F2-plugin-carrier-migration`），
上游提案被用户裁定不做 → **运行树分层补丁（tools/runtime-tree/ 重放路径）即承载形态**，
`tree-seam-assertions.json` + apply 前置校验持续把守「换镜像必 DRIFT」风险。
PURPOSE §8 的例外追加记录属判据级改动，待用户批准后落文。

## 判定（2026-10-20 无卡轮，逐条回答「能否只用 general_plugins / --additional-config / --worker-cls / CLI+env 表达」）

| 层 | 内容 | 判定 | 归档 |
|---|---|---|---|
| W3F2 定义点+rebind+runner-hook | 权重加载后、dynamo 编译/图捕获前对 `QwenGatedDeltaNetAttention` 做一次性 GDN in_proj 合并；env `VLLM_HUST_GDN_MERGE_INPROJ=1` 门控，default-off | 宿主无 seam、镜像定死、不走上游提案 → **长期例外：运行树补丁承载** | 本目录 patches/ 三件原件 |
| W3F3 shared-gate | shared scalar-gate pack（env `VLLM_HUST_SHARED_GATE_PACK`），定义点 `shared_experts.py` + runner-hook | 方法本体已 rejected（候选 `F3-shared-gate-pack`）→ **不提案**，补丁仅作历史证据 | patches/ 内 W3F3 两件，无测试义务 |

## 风险边界

更新的上游树（vllm-ascend-hust f939ad767）已不含 `prepare_input_before_routed_experts`
接口对；换镜像时 W3F2/W3F3 必然 DRIFT——这是预期防线，不是缺陷：
apply/seam 校验会以 rc=1 拒绝起服，届时按新镜像重新归类。

## 原件与校验

原件 = L2 `/root/包体恢复/records/tree-patches/2026101*T*_W3F*-runtime-tree.*`，
本目录 patches/ 为逐字节副本（sha256 见 tests/test_w3f2_runtime_patch_replay.py 内联清单）。
测试以 orig 为基做 `patch --dry-run` 重放，保证「原件可作用于其快照基线」；
分层链判定与幂等重放见 tools/runtime-tree/apply-runtime-patches.sh。
