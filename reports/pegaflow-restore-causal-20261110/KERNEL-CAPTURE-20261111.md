# 宿主机内核调用短时采集

目标是区分恢复慢尾中的驱动函数内部耗时与调用之间的调度空档。 当前容器看不到可用 tracefs，perf 受限；下面脚本为管理员准备，已通过 Bash 语法检查，尚未在宿主机执行。

本轮慢宿主机为 `host-192-168-0-2`，工作区 `hust-zsx-91`，物理7卡。 共享路径可见时使用：

```sh
bash /data/shared_upload/hust-zsx/restore-causal-20261110/capture-restore-kernel-20261111-v2.sh --check
bash /data/shared_upload/hust-zsx/restore-causal-20261110/capture-restore-kernel-20261111-v2.sh --capture /path/to/new-private-output 20
```

若宿主机没有挂载该共享路径，先取得版本化的 [v2 脚本](analysis/capture-restore-kernel-20261111-v2.sh)。包 A 中保留准备阶段的 v1。
`--check` 只读检查 AArch64、已挂载 tracefs 和四个目标符号。 `--capture` 需要宿主机管理员的内核探针权限；本脚本不挂载 tracefs或调整系统权限。

需在显示 `Capture ready` 后，立即在容器里运行一次原始慢恢复。 可以使用包 A 中的原生探针及其冻结 command/config，也可使用包 C 的独立 C 重现器；
保持原数据量和描述符数，采集窗口至少覆盖已确认变慢的区间。若这次恢复未出现慢化，应标为正常对照。

四对进入/返回事件：

1. `trs_hw_sq_send_task`
1. `hal_kernel_trs_chan_send`
1. `trs_chan_fill_sqe`
1. `trs_chan_flush_sqe_cache`

脚本创建独立 trace instance 和唯一事件组，默认 20 秒、上限 30 秒；采集结束清理其创建的实例和探针。 输出包括内核/启动身份、已执行脚本、原始调用事件、kretprobe
命中/丢失统计。 v2 还保存事件格式、trace 选项、逐 CPU 的采集前后缓冲统计，并记录通道提交函数的第二个参数。 该参数须与本实验 `kernel-resources.json` /
`/proc/trs_chan` 的实际通道 ID 核对； 公开接口原型提供解释依据，但仍须通过这个运行时对应关系验证当前二进制。
探针覆盖这些函数的宿主机调用，可能包含其他卡及进程；原始指针和其他租户信息先保留在管理员私有输出目录， 提供分析前按目标上下文筛选，公开归档使用去标识的派生结果。

分析时应按线程和嵌套关系配对进入/返回事件，先检查丢失、未配对、跨窗口和探针对吞吐的影响。 只凭同一线程相邻两行的时间差，不能认定是函数耗时；返回值也不应统一解释为错误码。

若外层持续几十毫秒、子层对应地持续，继续向耗时子层查。这里测得的是函数边界之间的经过时间， 包括线程睡眠和被抢占，不能直接称为 CPU 执行时间。若外层调用短但长时间没有被调用，
再由管理员针对已识别的工作线程补一段 `sched_switch/sched_wakeup`，将运行、可运行等待和睡眠分开。 现阶段无需先做模型完整推理或改变网络配置。

## 日志分析

```sh
python analysis/analyze_kernel_calls.py /path/to/capture --channel OWNED_CHANNEL_ID --output kernel-summary.json
```

`OWNED_CHANNEL_ID` 必须取自本次原生探针创建的通道，不沿用另一轮分配的 ID。 省略 `--channel` 时汇总整个采集窗，不能把其他卡或其他租户的最长调用归到目标恢复。
跨多个通道的外层调用会从单通道统计中排除并计数。

分析器按线程和嵌套关系配对，支持 CPU 迁移，输出各函数经过时间分位数、已跟踪子调用以外的时间、
同一上下文相邻调用的空档及最大样本。子调用以外的时间仍包含未跟踪工作和调度；同上下文空档也可能是其他上下文工作， 两者均不等于 CPU 空闲时间。

先检查 `pairing_quality`、`loss_checks` 和 `complete_loss_free_pairing`。 不完整配对、嵌套损坏、显式丢失、kretprobe
missed、缓冲 overrun、未读取尾部及命中数不一致都会降低完整性。 已配对的部分样本可以辅助检查，但不能由其分位数宣称整个恢复窗的覆盖。
输出将线程与指针替换为局部编号；原始私有日志继续由管理员保管。

本地已通过 11 项合成日志测试，覆盖迁移、并发、丢失、截断和多通道误归属；这些测试不是宿主机实测。 运行测试：

```sh
python -m unittest discover -s analysis -p test_kernel_calls.py
```

格式和丢失检查依据 Linux 的 [kprobe 事件说明](https://www.kernel.org/doc/html/latest/trace/kprobetrace.html) 与
[ftrace 缓冲统计](https://www.kernel.org/doc/html/latest/trace/ftrace.html)。 通道提交原型参考固定版本的
[chan_rxtx.c](https://github.com/hicann/driver/blob/471fa6b09b556cabdb7ed917a8dd2ec25ed0bc6b/src/sdk_driver/trsdrv/trsbase/chan/chan_rxtx.c)。
