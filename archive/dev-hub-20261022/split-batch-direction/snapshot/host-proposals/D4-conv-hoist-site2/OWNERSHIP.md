# D4 conv-weight 转置第二落点外提（runtime-tree 长期例外快照）

- 路由: 028/D4-const-hoist | 候选: d4-conv-transpose-second-site
- 领地: /vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py（W3F2 同文件族, PURPOSE §8 例外已批）
- 载波: VLLM_HUST_D4_CONV_PACK=1（default-off; pack 挂层自身 process_weights_after_loading,
  经 model_runner_v1 W3F2 块在 VLLM_HUST_GDN_MERGE_INPROJ=1 时调用）
- 判据: L3 8/8 identical（1021-140043 vs 1021-140814）; light L4 Transpose 960→0
  （1021-140924-opstat）; 单变量逐项不变（MatMulV2/GroupedMatmul/MoeInitRoutingV3）
- 回滚: patches/*.orig 覆盖回运行树即可（无其他文件依赖）
- 施加记录: 20261021T1415Z_applied.note.md; 原件守卫: tests/test_d4_site2_runtime_patch_replay.py
