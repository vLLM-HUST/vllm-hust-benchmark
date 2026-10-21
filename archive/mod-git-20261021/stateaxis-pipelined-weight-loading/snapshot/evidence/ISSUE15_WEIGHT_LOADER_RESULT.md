# Issue #15：identity-bound pipelined weight loader 收口

> **文档角色：已冻结的 point-in-time 结果，不是当前状态入口。** 当前能力分类只看
> [`NATIVE_ENGINE_STATUS.md`](../../NATIVE_ENGINE_STATUS.md)；旧 R8 与 2026-08-14 acceptance
> audit 继续作为历史证据保留，不因本次新实验而改写。

## 裁决

Issue #15 的原验收范围现已完成。Qwen2.5-14B 的受控 cold 与 warm page-cache
实验均完成 control/candidate 各 3 次 fresh process；所有 12 次 serving run 的 arena、
逐 pack identity、请求 token 和 lifecycle 均 exact。另一个真实 MiniMax-M2.5 layer-0
EP8 artifact 在 8 张物理 910B2 上完成 1+1：32 个文件、8 个 rank 的源 digest、D2H
arena digest 和跨 rank 事务性 Ready 全部通过。

Qwen cold process-to-Ready 中位数由 344.168 s 降至 53.895 s，下降 84.340%；
warm 由 324.809 s 降至 36.392 s，下降 88.796%。cold TPOT P50 变化 +0.043%，
warm 变化 -0.685%，都通过不超过 1% 回退门禁。该结果满足预注册的 cold Ready
至少下降 20% 条件。

候选仍是 execution-artifact 显式绑定的 opt-in 路径，未绑定 load plan 时保持
`serial_pageable_sync`。这是有意的 fail-closed 部署边界：不同 carrier 必须先证明
pinned allocation、异步 copy 与 `libcrypto.so.3` closure；本结果不把一台机器上的
验收自动改成所有环境的隐式默认。

## Qwen 3+3

两臂完全匹配同一个 48+1 pack Qwen2.5-14B artifact（29,540,560,896 bytes）、
请求、oracle、官方 `.23` 镜像和计时边界。control 使用 scalar SHA、pageable buffer
和同步 H2D；candidate 使用 ARM SHA2-backed OpenSSL SHA-256、两个 64 MiB pinned
slot、64 MiB chunk 和 event-guarded async H2D。两臂都对实际复制的每个字节重算
SHA-256；digest cache 只用于 artifact gate，不能代替 loaded-byte 校验。

| Cold metric（3 次中位数） | Control | Candidate | 变化 |
|---|---:|---:|---:|
| process-to-Ready | 344.168 s | 53.895 s | -84.340% |
| loader total | 336.850 s | 49.526 s | -85.297% |
| TTFT P50 / P95 | 470.125 / 608.429 ms | 468.916 / 607.688 ms | -0.257% / -0.122% |
| TPOT P50 / P95 | 44.151 / 48.640 ms | 44.170 / 48.642 ms | +0.043% / +0.004% |
| output token/s | 550.091 | 550.224 | +0.024% |
| peak HBM | 44,687 MiB | 44,687 MiB | 0 MiB |
| Ready CV | 0.395% | 0.138% | — |

| Warm metric（3 次中位数） | Control | Candidate | 变化 |
|---|---:|---:|---:|
| process-to-Ready | 324.809 s | 36.392 s | -88.796% |
| loader total | 317.324 s | 32.180 s | -89.859% |
| TTFT P50 / P95 | 472.151 / 610.484 ms | 469.063 / 607.027 ms | -0.654% / -0.566% |
| TPOT P50 / P95 | 43.998 / 48.479 ms | 43.696 / 48.181 ms | -0.685% / -0.613% |
| output token/s | 550.911 | 554.604 | +0.671% |
| peak HBM | 44,688 MiB | 44,688 MiB | 0 MiB |
| Ready CV | 0.770% | 0.434% | — |

受控 cold 使用实验私有 inode；每次 fresh process 前执行 `POSIX_FADV_DONTNEED`，
并以 `mincore` 证明 49 个 pack 的 resident page 都为零。warm 则先逐字节 SHA，再以
`mincore` 证明所有页面 resident。这里没有使用全局 drop-caches，也没有驱逐未知资源。

独立 resource 1+1（不参与 3+3 性能裁决）显示 control/candidate peak RSS 为
3,388,833,792 / 907,362,304 bytes，CPU/wall 为 94.88% / 73.49%，swap 都为零；
candidate 固定 pinned staging 为 134,217,728 bytes。正式 3+3 的 steady-state 指标
才是 TPOT/吞吐回退裁决来源。

## 真实 EP8 验收

MiniMax cell 使用 `Eco-Tech/MiniMax-M2.5-w8a8-QuaRot` revision
`9118343f43b8177306a312fcc0c8b766012796b8` 的 layer-0 prepacked experts：8 rank、
每 rank 4 文件、总计 3,628,597,248 bytes。私有树的每个 inode 都在复制时完成全字节
SHA、fsync，并绑定 canonical path、dev、inode、size、mtime_ns 与 nlink；两臂前的
885,888 个文件页均被证明 nonresident。

| EP8 cold 1+1 | Control | Candidate |
|---|---:|---:|
| 进程耗时 | 20.666 s | 9.352 s |
| 最慢 rank total | 18.855 s | 7.019 s |
| peak RSS | 1,850,101,760 B | 1,175,568,384 B |
| peak pageable / rank | 301,989,888 B | 16,777,216 B |
| pinned staging（8 rank 总和） | 0 B | 268,435,456 B |
| max HBM at Ready | 871,972,864 B | 871,632,896 B |
| exact + transactional Ready | pass | pass |

candidate 的进程耗时点估计下降 54.748%，但该 cell 只验收真实 8-rank loader 的
正确性、资源上界和事务屏障；它不是完整 MiniMax 模型 serving，也不产生 MiniMax
端到端 speedup 声明。

## Fault 与身份闭环

host tests 对 short read、bit flip、digest mismatch、duplicate/overlap/OOB span、
stale size/plan、pinned allocation、async copy、rank failure、cancel 和 event timeout
逐项 fail closed；单 rank 或单 span 完成不会产生半 Ready。Qwen worker、producer
closure、load plan、execution artifact 和 pack root 都以 SHA-256 绑定；EP8 同时绑定
manifest、私有 inode tree 与每 rank arena view。

机器可读汇总为
[`issue15_weight_loader_20260815_r5_result.json`](../../../benchmarks/issue15_weight_loader_20260815_r5_result.json)，
精简 per-run raw 为
[`issue15_weight_loader_20260815_r5_raw.jsonl`](../../../benchmarks/issue15_weight_loader_20260815_r5_raw.jsonl)，
发布状态为
[`issue15_weight_loader_20260815_r5_status.json`](../../../benchmarks/issue15_weight_loader_20260815_r5_status.json)。
它们记录并哈希本机 append-only `results/` 原始目录；失败的 r4 gate 和第一次 EP8
cache-domain 失败也继续原样保留。所有本实验容器均已清理，诊断用单文件副本已删除，
没有停止或修改非实验容器。
