# ConcatD 零拷贝解包 — decode T=1 视图路径（runtime-tree 快照）
- 路由 028/D4-const-hoist | env VLLM_HUST_GDN_UNPACK_VIEWS=1 default-off
- 回滚: 恢复链 = 先回 D4 site2 patch 前的 orig（patches/*.orig），或整体覆盖 patches/*.orig + 逐段反向
- 守卫: tests/test_concatd_views_runtime_patch_replay.py
- 相邻已拒: gdn-qkv-unpack-single-pass（单拷贝）——本候选为零拷贝不同机制, 用户 2026-10-21 授权重开
