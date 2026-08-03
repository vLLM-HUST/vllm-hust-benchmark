# Docker NPU 作业归属契约

poy-180 的 GitHub Actions runner 使用 `poy-180-21rc-npu0` 至 `poy-180-21rc-npu3` 四个名称。通过宿主机 Docker
socket 启动的容器必须显式携带 runner 归属，否则 watchdog 会把其中的 NPU 进程视为无主进程。

统一使用 `scripts/run-watchdog-owned-npu-container.py` 启动这类作业：

```bash
python scripts/run-watchdog-owned-npu-container.py \
  --name "benchmark-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
  --image quay.io/ascend/vllm-ascend:21rc \
  --volume "${GITHUB_WORKSPACE}:/workspace:ro" \
  -- python /workspace/run_benchmark.py
```

脚本从 `RUNNER_NAME` 解析物理卡，只把对应的 `/dev/davinciN` 映射为容器内 `/dev/davinci0`，并固定
`ASCEND_RT_VISIBLE_DEVICES=0`。创建后会读取 Docker inspect，核验 runner label、物理卡 label、设备映射和逻辑设备环境，再启动容器。

作业不得自行覆盖以下字段：

- `org.vllm-hust.runner`
- `org.vllm-hust.npu-physical`
- `org.vllm-hust.npu-logical`
- `ASCEND_RT_VISIBLE_DEVICES`
- `ASCEND_VISIBLE_DEVICES`

### Volume / mount 约束

Volume 入口若不约束，可以通过 `--volume /dev:/dev` 或 `--volume /:/host` 绕过单卡容器的设备隔离。启动脚本和
`validate_container_inspect` 双层校验所有 volume：

- **宿主机源路径黑名单**：禁止 bind-mount
  `/`、`/dev`、`/dev/davinci*`、`/sys`、`/proc`、`/var/run/docker.sock`、`/usr/local/Ascend`、`/etc/dcmi*`
  等路径，这些路径会暴露宿主机设备节点、NPU 驱动 sysfs 或 Docker socket，从而绕过隔离。
- **容器目标路径白名单**：只允许挂载到 `/workspace`、`/tmp`、`/data`、`/root/.cache`、`/home`、`/opt/models` 及其子目录。
- **禁止危险选项**：`shared`/`slave` propagation、privileged 模式、`SYS_ADMIN`/`SYS_PTRACE`/`SYS_MODULE`
  capabilities 均被拒绝。
- **Docker inspect 双重校验**：`validate_container_inspect` 同时检查 `Mounts` 数组和旧式 `HostConfig.Binds`
  列表，确保运行时配置与构建时一致。

`job.container` 目前无法通过该脚本完成创建前校验。在 GitHub Actions 能可靠写入 上述 label 和设备映射以前，不得把 `job.container` 作业调度到
`linux-aarch64-a2b3-pool`。

代码合并前仍需在真实 runner 上分别验证成功、失败和取消路径，并确认两轮 watchdog 扫描后正确标记的容器仍存活；错误 label 或错误设备映射的告警记录在 #125。
