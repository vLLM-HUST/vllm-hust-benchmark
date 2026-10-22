# comm mechanism trace —— 归属记录

## 最终归类（2026-10-20 用户裁定：保留，不退役）

comm trace 线（`src/vllm_ascend_split_batch/comm_trace.py`、`comm_trace_worker.py`、
`tools/collect_comm_mechanism.sh`、`tools/audit_comm_mechanism.py`、
`tools/attribute_comm_free.py`、`tools/check_comm_profiler_stop.py` 及其测试）**保留**：

- 它是 `comm-overlap-free-attribution` 三个关键问题（allreduce 图层调用点、
  payload 16/825/1489/1587/2048 相位调用点、free 是否落在 device graph 边界外 +
  overlap 缺口）目前**唯一**的取证通道；宿主默认 wrapper 不给
  `record_shapes/with_stack/Text+Db`，答不了 allreduce 归因。
- 测试是纯 CPU 的编排防回归（preflight / audit / offline-audit），不占卡，保留。

## 现状（validation/20261019/receipt.json：`mechanism_status = blocked`）

- 063113 窗口：rank1 pinned buffer 初始化失败（EL0004），profiler 未启动；
  serve 启动额度用尽，未重试。
- 离线修复：torch_npu `PythonTracer::clear` 无 GIL → SIGSEGV；最小补丁
  `host-proposals/comm-trace-stop-gil/patch.diff` 离线验证 6/6 exit0，
  但完整 libtorch_npu 未编完（cap 用尽）→ `runtime_repair_verified: false`。
- `required_artifacts` 全 MISSING；三个关键问题 UNANSWERED。

## 解阻塞路径（未排期；属 §0.2 额度项）

1. 编完打过补丁的 torch_npu（补丁已验证）。
2. 重跑 `tools/collect_comm_mechanism.sh`（占卡 + 1 次 serve 启动窗口）。

## 退役触发条件

用户裁定放弃 comm-free 归因方向时，按 cascade 先例退役：删 src/tools/tests，
保留反回归 tripwire，并同步 HOST_CONTRACT/README。
