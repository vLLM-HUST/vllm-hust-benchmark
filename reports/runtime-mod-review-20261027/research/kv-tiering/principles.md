# KV Tiering：设备、CPU 与文件系统之间的缓存分层

日期：2026-10-26。源码固定于
`vLLM-HUST/vllm-hust-kv-tiering@82c4fe863e0ad2281e56ef0b5333319ec5f19dbd`。本轮是关键源码阅读，未运行插件或硬件实验。

## 它做什么

方向是保留算过的 KV：设备容量紧张时，借助 CPU 内存和文件系统保存历史块，后续命中后取回，避免重复 prefill。它不做语义相似替换，也不是选择主动绕过已有缓存重算的 arrival
policy。

概念路径：设备 KV ↔ CPU primary buffer ↔ filesystem secondary tier。模型仍在设备上使用恢复后的状态；不是直接让 attention
在磁盘上计算。具体保存、加载调度与缓存匹配依赖宿主 offloading 系统，不是本插件完全重写。

## 实际贡献和接入

`HustTieringOffloadingSpec` 继承宿主 `TieringOffloadingSpec`，注册 `hust_fs`，普通路径沿用宿主 worker。插件的实质增量包括：

- filesystem tier：通过 CPU buffer 读写缓存块；
- segment 文件与 SQLite 索引：多个块追加进大文件，记录 key → segment/offset/length，减少一块一文件的碎片化；
- 连续位置的批量 `preadv` 读取，以及 legacy 单块文件格式；
- 后台 load/store 队列，读优先线程与写优先线程分别偏好自己的队列，完成后供宿主轮询；
- 可选 Ascend 布局与同步复制适配。

[spec](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/src/vllm_hust_kv_tiering/spec.py)；[文件系统 tier](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/src/vllm_hust_kv_tiering/fs/manager.py)；[segment store](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/src/vllm_hust_kv_tiering/fs/segment_store.py)。

## 不能把 feature 清单当作全部已接通

仓库包含 `SessionLifecycleManager`、TTL、CPU demotion 和 residency tracking 组件。但是本轮在插件源码中未找到活跃 spec/FS
manager 对该 lifecycle manager 的实例化调用；不能仅因文件存在，就断言默认服务已启用这些策略。宿主是否通过额外路径接入需要单独核实。

此外，FS 任务异步排队不等于整个设备↔内存↔磁盘路径都可与推理无阻塞重叠。

## Ascend 当前的关键限制

显式 opt-in 的 `ascend_copy_backend: torch_sync` 路径使用：

```text
NPU synchronize → tensor.copy_(non_blocking=False) → NPU synchronize
```

所以它目前是先保证复制正确性的同步实现，不是 HiCache 那种高效传输重叠能力的等价交付。插件 owned connector 还为分离 K/V 与 Mamba state views
构建布局、保留宿主 block/group/sub-block 映射；不支持的布局会拒绝。

作者记录了真实 910B2、partial-page、two-group 的 NPU→CPU→NPU 往返与 padding 保持。但 README
仍明确：**完整模型正确性、实际缓存复用、性能和恢复验收未完成**。不能由 buffer round trip 推出完整服务可用，更不能宣称实际推理加速。

[同步 worker](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/src/vllm_hust_kv_tiering/ascend_worker.py)；[当前资格边界](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/README.md)。

## 当前理解

它属于我们刚讨论的“扩大 KV 保留容量，命中后搬回而不是重算”方向。范围更准确地说是 **宿主 offloading 框架上的文件系统缓存后端，加实验性 Ascend
适配**，不能把它等同于已经成熟的完整多级缓存服务。

收益条件仍需真实负载验收：缓存命中率、块恢复正确性、传输与计算重叠、CPU/存储带宽、同步停顿及全局吞吐。暂不替 Fletcher 作采用或 PASS 决定。

## 传输链路追踪：同步/异步具体落在哪里

补充核对了插件声明的 Frontier core `d0f22d2bda562156e4dbf433ce645e1769b4f804` 中 OffloadingConnector 与
OffloadingConnectorWorker；未沿所有宿主 tier-manager 状态转换作完整审计。

### CPU ↔ filesystem

- `lookup` 不立即等待查盘：未有结果时返回 RETRY；`on_schedule_end` 把本轮查询 flush 给后台线程，下轮收集 HIT/MISS。
- `submit_load` 在调用线程查询 SQLite location、整理任务，再把实际块读取放进 load 队列。
- `submit_store` 在调用线程 prepare/reserve segment 写入计划，再入 store 队列。因而不能声称提交路径完全无阻塞：索引、锁与任务准备仍同步进行。
- 后台线程执行普通阻塞 `preadv`/`pwritev` 等 I/O；不是 io_uring 或设备 DMA 异步完成机制。批量 load 直接把文件字节读入 CPU primary
  buffer 的 memoryview 切片。
- 任务聚合为 job completion，通过 finished 队列由宿主轮询；写入完成回调成功后提交块索引。`drain_jobs` 则显式等待整个池空闲。

### Ascend NPU ↔ CPU

`submit_store` / `submit_load` 直接调用 `_copy`，在调用线程执行：预先校验所有 span → `torch.npu.synchronize(device)` →
逐 span `copy_(non_blocking=False)` → 再次 synchronize → 记录 TransferResult → 返回。没有独立 copy stream/event
的异步流水；`wait(job_ids)` 是 no-op，因为 submit 返回时已拷完。

底层仍可能使用设备 DMA 实现 tensor copy，但接口与当前调度语义是同步，不能称为异步 DMA overlap。同步针对该 worker
的设备，不是停止整个集群；会让当前调用链等待该设备工作完成，影响真实吞吐的幅度尚未测量。

CPU buffer 来自宿主 `SharedOffloadRegion`。FS tier 看到 memoryview，Ascend worker 看到对应 tensor
view；AscendLayoutWorker 按同一宿主块映射，在 CPU page 内按 byte offset 拼放分离 K/V 或 recurrent state 的
views。不是文件直接读入 NPU。

### 宿主调用与回执

```text
OffloadingConnector.start_load_kv
  → OffloadingConnectorWorker.start_kv_transfers
    → 先提交上一轮延后的 stores
    → 再提交 metadata.load_jobs
      → 当前 Ascend worker 的同步 submit_load/store
```

`get_finished` 中先 `prepare_store_kv` 将本轮 store 暂存，到下一步开始再真正提交，宿主注释意图是避开 sampling
传输。延后一轮不是异步复制；Ascend 提交后仍阻塞。preemption 路径会提交暂存 store 并 wait jobs_to_flush，保证必要的块复用 fence。

宿主收集 completed job IDs；load 完成给出 finished_recving，让 scheduler 能恢复等待 KV 的请求。这里回执协议与数据复制是否异步是两件事。

该 connector 的 `wait_for_layer_load` 和 `save_kv_layer` 都是空实现，因此不能把它描述成逐层事件等待的“算第 N 层，同时搬第 N+1
层”机制。是否还能在其他层面形成请求间重叠，要看完整宿主调度，不能由函数名推定。

[宿主 connector](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py)；[宿主 worker](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py)；[FS 线程池](https://github.com/vLLM-HUST/vllm-hust-kv-tiering/blob/82c4fe863e0ad2281e56ef0b5333319ec5f19dbd/src/vllm_hust_kv_tiering/fs/thread_pool.py)。

## 共同评审 comment（2026-10-26）

**设备搬运侧都是同步搬运，对于计算的干扰非常大，生产不可用。**

本判断针对本轮审阅的 Ascend `torch_sync` 路径。同步点已由源码核实，干扰程度与生产采用结论属于共同评审判断，非本轮性能实测；不据此评价未审阅的 CUDA/XPU 路径。
