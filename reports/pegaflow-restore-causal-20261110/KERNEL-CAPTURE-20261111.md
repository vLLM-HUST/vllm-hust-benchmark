# 宿主机内核调用短时采集

目标是区分恢复慢尾中的驱动函数内部耗时与调用之间的调度空档。
当前容器看不到可用 tracefs，perf 受限；下面脚本为管理员准备，已通过 Bash 语法检查，尚未在宿主机执行。

本轮慢宿主机为 `host-192-168-0-2`，工作区 `hust-zsx-91`，物理7卡。
共享路径可见时使用：

```sh
bash /data/shared_upload/hust-zsx/restore-causal-20261110/capture-restore-kernel-20261111.sh --check
bash /data/shared_upload/hust-zsx/restore-causal-20261110/capture-restore-kernel-20261111.sh --capture /path/to/new-private-output 20
```

若宿主机没有挂载该共享路径，先从[包 A](raw/resume-20261111-a.tar.gz)取出脚本。
`--check` 只读检查 AArch64、已挂载 tracefs 和四个目标符号。
`--capture` 需要宿主机管理员的内核探针权限；本脚本不挂载 tracefs或调整系统权限。

需在显示 `Capture ready` 后，立即在容器里运行一次原始慢恢复。
可以使用包 A 中的原生探针及其冻结 command/config，也可使用包 C 的独立 C 重现器；
保持原数据量和描述符数，采集窗口至少覆盖已确认变慢的区间。若这次恢复未出现慢化，应标为正常对照。

四对进入/返回事件：

1. `trs_hw_sq_send_task`
2. `hal_kernel_trs_chan_send`
3. `trs_chan_fill_sqe`
4. `trs_chan_flush_sqe_cache`

脚本创建独立 trace instance 和唯一事件组，默认 20 秒、上限 30 秒；采集结束清理其创建的实例和探针。
输出包括内核/启动身份、已执行脚本、原始调用事件、kretprobe 命中/丢失统计。
探针覆盖这些函数的宿主机调用，可能包含其他卡及进程；原始指针和其他租户信息先保留在管理员私有输出目录，
提供分析前按目标上下文筛选，公开归档使用去标识的派生结果。

分析时应按线程和嵌套关系配对进入/返回事件，先检查丢失、未配对、跨窗口和探针对吞吐的影响。
只凭同一线程相邻两行的时间差，不能认定是函数耗时；返回值也不应统一解释为错误码。

若外层持续几十毫秒、子层对应地持续，继续向耗时子层查；若外层调用短但长时间没有被调用，
再由管理员针对已识别的工作线程补一段 `sched_switch/sched_wakeup`，将运行、可运行等待和睡眠分开。
现阶段无需先做模型完整推理或改变网络配置。
