# cohort 边界与归属（长期核心约束）

> **本文件是"我们能声称什么、谁负责什么"的权威源。**
> 约束已写进 [AGENTS.md](../AGENTS.md) 硬约束 6；此处给依据、边界与判定口径。
> 来源：上游 @ShuhaoZhangTony 的两条裁定（本仓 issue，逐字原文见
> `knowledge/handoffs/receipts/20260929-upstream-cohort/facts.md`，工作区）：
> - issue #2 comment `5867233862`（2026-09-28，**Frontier 合同审计结论**）
> - issue #4 comment `5891258209` / `5891262039`（2026-09-29，**owner handoff**）
>
> **2026-10-13 owner 裁定（基准切换）**：本仓基准已从 Qwen2.5/0.25 切到 **0.28 镜像 + Qwen3.5-35B-A3B**
> （"我们不会再做 Qwen2.5"）。§1 的上游原文是**历史收据，逐字保留不动**；适用域以 §2 为准。
> Qwen2.5/0.25 的全部证据降级为**历史独立专项**（保留、不再作为本仓主张基线）。

## 1. 上游裁定说了什么（两条，逐字可核）

**① 审计结论（#2，审计对象 = 本仓 `f77dc2214727ab1448874f1aee787a67328f1433`）**

> "当前 … 明确把任何 `speculative_config` 判为 `speculative_decode_conflict`，README 说明这只是 guard、
> 不是 MTP 支持，启用时 cascade 不产生加速；HOST_CONTRACT 也要求拒绝 speculative decoding。
> 统一配置必须保留 native MTP2，不能关闭或替换，所以本轮未绕过 precheck，**也没有生成形式上成功
> 但实际退化为 Native 的点**。… 要进入统一 cohort，需要先设计支持 k+1 verification rows 的
> dual-pad/graph bucket 合同，并补 native MTP2 + chunked prefill、APC、async、FULL_AND_PIECEWISE、
> Qwen3.5 hybrid TP2 的 fail-closed 与真实 replay correctness。**现有 Qwen2.5 数据继续作为独立
> 专项证据，不应外推到该配置。**"

⇒ 两句可直接引用的结论：**未发现作弊式结果**；**不得外推**。

**② 归属 handoff（#4）**

> "owner handoff: 0.1.3 is executable and should remain visible, but its Cascade path deliberately
> does not exercise MTP steps and targets the 0.25 host line. Independent cohort definition,
> kernel-wheel licensing/distribution, NPU evidence, and performance claims remain with the project
> owners; the central team will only preserve the classification and publication boundary."

⇒ 中央团队**只保留"分类与发布边界"**；下列四项**回到本仓 owner**：

| # | 归属项 | 本仓现状（可核） | 缺口 |
|---|---|---|---|
| ① | independent cohort definition | 宿主域 [release.md](release.md) §0.1/§0.3；能力域 [support-matrix.md](support-matrix.md) §5.1 | 缺一页正式 cohort 声明（本文件 §2 即其初版） |
| ② | kernel-wheel licensing / distribution | 许可齐全（wheel 内含 CANN OSL 2.0 协议文本、METADATA 已订正、根 `LICENSE`/`NOTICE`）；已按 GitHub Release 附件分发（算子仓 `v2026.9.27`，附件 = 本地构建字节） | **算子仓仍在个人账号 `Raing5Days/vllm-hust-cascade-kernel`**，org 归属未定 |
| ③ | NPU evidence | 0.28 基准：[evidence/host-028-2026-10-11.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/host-028-2026-10-11.md)、[evidence/d4-gate-fp32-cohort.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/d4-gate-fp32-cohort.md)；历史：[evidence/cascade/](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/cascade/README.md)（147 件）+ 目标栈验证程序 | **MTP2 域仍未验证**（保持 fail-closed）；历史 cascade 证据只覆盖 0.25 线 + Qwen2.5 |
| ④ | performance claims | [evidence/cascade/section3-performance.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/cascade/section3-performance.md) §13.5 | 对外引用必须带 §3 的 cohort 边界 |

## 2. 本仓 cohort（我们能声称的适用域）

**覆盖（声明兼容且已实测）—— 2026-10-13 现行基准**：

| 维度 | 取值 | 依据 |
|---|---|---|
| 宿主 `vllm-ascend` | `>=0.25.1rc2.dev125,<0.25.2`（0.28 镜像的 hust 构建，版本串仍停在 0.25.1rc2 窗口） | manifest `host.version_range`；[evidence/host-028-2026-10-11.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/host-028-2026-10-11.md) |
| vllm core | `0.28.1.post1.dev143+gf18cf803c`（0.28 镜像默认栈配对） | 同上 |
| 底座 / 设备 | Ubuntu 22.04.5 + CANN 9.1.0 + Python 3.12.13；2×910B2（**TP=2**） | 同上 |
| 模型 | **Qwen3.5-35B-A3B**（40 层：30 GDN + 10 MoE），hybrid **TP2** | 同上 + [evidence/d4-gate-fp32-cohort.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/d4-gate-fp32-cohort.md) |
| 请求形态 | chunked prefill **on**；APC **off**；max_model_len 8192 | D4 serve 固定协议 |
| 调度 / 图 | `cudagraph_mode=NONE`（本栈 `FULL_AND_PIECEWISE` 在 `capture_model` 阶段崩，见 [pitfalls.md](pitfalls.md) §1.3） | d4-gate-fp32-cohort.md |
| 精度档 | bf16 | 同上 |

**不覆盖（不得据此声称）**：

- **native MTP2 / 任何形式的投机解码** —— 一律 fail-closed（不变）；
- 未跑过的 `async` 组合；
- 0.28 镜像之外的其它宿主 build（除 host-028 已实测的一支）；
- **历史 Qwen2.5 / 0.25 域**的结论不再自动适用于本基准，反之亦然。

**历史 / 独立专项（保留，勿删）**：Qwen2.5-14B-Instruct（行为类）与 Qwen2.5-Coder-14B（性能替身）在
0.25 线上的全套证据（[evidence/cascade/](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/cascade/README.md)）是**独立专项**；按原口径引用时
必须同时标注"历史域"，且**不得与 0.28/Qwen3.5 的数字互推**。

## 3. 引用纪律（写结论时逐条对照）

1. **任何性能/正确性数字必须带 cohort 边界**（至少写：宿主线、模型、请求形态、APC/图配置）。
   缺一项即视为不可引用的孤立数字。
2. **禁止跨域外推**：Qwen2.5/0.25 的历史数字不得外推到 0.28/Qwen3.5 基准，反之亦然。
3. **MTP 只能写成"边界/拒绝"**，不得写成"支持"。可引用 README
   [「Speculative decoding boundary」](../README.md) 与 `HOST_CONTRACT.md:11`。
4. **要再改 cohort** 须由 owner 裁定并在本文件留档；2026-10-13 的基准切换即按此执行
   （[docs/evidence/host-028-2026-10-11.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/host-028-2026-10-11.md)、[docs/evidence/d4-gate-fp32-cohort.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/d4-gate-fp32-cohort.md) 为本次切换的实测依据）。
5. MTP / k+1 形态的"进统一 cohort"前置**当前未立项**（与工作区台账 OPEN-05 的既有裁定一致：
   维持 fail-closed 边界）。立项需项目侧明确裁定；**不得通过"绕过 precheck"来造出形式上的成功点**。

## 4. 变更本文件的条件

下列任一发生即须同步本文件与其引用处（`AGENTS.md` 硬约束 6、`support-matrix.md` §5.1、README）：

- `host.version_range` 变更（放宽/收窄）；
- 上游对 cohort 或归属有新裁定；
- **本仓基准切换**（2026-10-13：Qwen2.5/0.25 → 0.28/Qwen3.5，已执行）；
- 四项归属中任一的状态变化（如算子仓转 org、补出 0.28/Qwen3.5 域证据）；
- MTP / k+1 形态的合同或实现发生变化。
