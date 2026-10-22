# W2b — fi_sampling 接入 vLLM-Ascend fork 采样路径(default-off)与 e2e TPOT 证据

- 日期:2026-09-09
- 落点:插件仓库 `vllm-ascend-split-batch-hust`(分支 `feat/cascade-attention-plug`)
- 被集成物:W2 已 review(approve)的 `knowledge/surveys/sampling/fi_sampling`
  (flashinfer `main@3a4e7052` 采样主路径 → triton-ascend 3.2.1,910B2 实测,
  统计等价 191 PASS / 0 FAIL)
- 宿主:`/vllm-workspace/vllm` 0.23.0(empty)+ `/vllm-workspace/vllm-ascend`
  0.23.0rc1;**两棵宿主源码树零改动**
  (**历史口径**:这两棵参考树已于 2026-09-10 删除;本报告全部数字绑定
  0.23.0rc1/CANN 9.0.1 口径,**不跨宿主继承**——在新基线翻 active 前需复测)
- 环境:Python 3.12.13 / torch 2.10.0+cpu + torch_npu 2.10.0.post2 / CANN 9.0.1 /
  910B2(单卡,`ASCEND_RT_VISIBLE_DEVICES=7`) / triton-ascend 3.2.1
- **manifest 保持 `import_only`**(本包只攒 e2e 证据,未翻 active)

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 接入机制 | `vllm.general_plugins` entry `fi-sampling` → `fi_sampling_plugin.load()`,运行期类级替换 `vllm_ascend.sample.sampler.AscendTopKTopPSampler` 为子类(与 fork `_310p/sample/sampler.py:96` 同款先例);未注册 `vllm.platform_plugins` |
| default-off | env `VLLM_HUST_FI_SAMPLING` 缺省 = 不 import 宿主/内核、不 patch(子进程断言 `vllm_ascend.sample.sampler` 与 `triton` 均未进入 `sys.modules`) |
| 分流 | 无截断 → FI api1 kernel;**大 B(B≥256)+ k≥32 + top-p → FI joint**(block_v=4096);k=1 → argmax;其余 → fork 原链 |
| 回退 | 五类强制回退全部落地(per-request generators / processed_logprobs 模式 / batch-invariant / reduce-sample / async-exponential),回退腿**零设备同步** |
| e2e TPOT(无截断,主收益区) | on/off 两轮中位数差 **−0.14% / +0.02%**(median/mean),即**在噪声带内无变化**——原因见 §4,采样并非该形态的瓶颈 |
| e2e TPOT(joint 形态) | **已作废(评审 F2)**:off 146.55 ms → on 162.17 ms(median,+10.7%)产自**排序修复前构建**,on 腿含每步多余 D2H 同步,该对比不可用于任何归因——详见 §3.3 勘误 |
| 关键形态事实 | 4k 上下文单卡 KV-bound 时 B_max≈122,joint cell(B≥256)**不可达**;需把上下文压到 512 才能到 B≥256。**真实收益在无截断路径;joint 分支在常规部署形态下是安全保留的"死分支"** |
| 质量门 | `pytest -q` fi_sampling 包 **44 例全绿**(全量 77 passed / 7 failed,失败为 cascade 新宿主锚点漂移,非本包,见 §8)+ `ruff check .` 全绿;NPU 冒烟 6/6 PASS(旧环境口径) |

## 1. 落点文件清单

### 1.1 插件实现(自有落点,宿主零改动)

| 文件 | 作用 |
|---|---|
| `src/vllm_ascend_split_batch/fi_sampling_plugin.py` | `vllm.general_plugins` entry;env 门控、类级替换、设备胶水(惰性导入内核、每步 seed 推进、单条 fail-open warning、可选路由 trace) |
| `src/vllm_ascend_split_batch/fi_sampling_route.py` | **纯逻辑**(不 import torch):五类回退判定 + 分流判据 + bench 来源注释 |
| `src/vllm_ascend_split_batch/fi_sampling/` | vendor 的 `fi_sampling` 宿主面(`__init__.py` 新写;`api.py`/`kernels.py`/`npu_env.py`/`pure.py` 逐字复制,仅把 `from fi_sampling.x` 改为相对导入) |
| `pyproject.toml` | 新增 entry `fi-sampling`;`[tool.ruff] extend-exclude` 排除 vendor 目录(逐字复制需保持与源包可 diff,源包按 ruff 默认行长而非 88 排版) |
| `tests/test_fi_sampling_route.py` | 纯逻辑单测:每类回退 ≥1 例 + 每个分流 cell + 阈值来源钉死 + `needs_top_k_read` |
| `tests/test_fi_sampling_plugin.py` | 集成契约单测:default-off 零替换/零导入(子进程)、替换为子类、各回退腿逐字委托、回退腿零同步、FI 腿参数与 seed 推进、kernel 失败 fail-open |

### 1.2 证据(本目录)

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 本报告 |
| `smoke_npu_fi_sampling.py` / `logs/logs_smoke_npu.txt` | NPU 冒烟(6 项,含 default-off 零替换、三类路由、generator 回退) |
| `run_e2e_matrix.sh` | e2e 矩阵驱动(每个 mode × repeat 独立 server 生命周期) |
| `run_e2e_tpot.sh` | 单次 e2e 跑法(早期版本,保留作命令记录) |
| `summarize_e2e.py` | 从落盘 JSON 汇总 TPOT(报告数字全部由它提取) |
| `results/serve_*.json` | **6 份参与汇总**:`serve_api1_{off,on}_{r1,r2}.json`(最终构建,api1 两轮)+ `serve_joint_{off,on}.json`(**修复前构建,已降级**,见 §3.3 勘误);另 `results/superseded/` 存修正前的首对 api1(2 份,不参与汇总) |
| `logs/w2b_serve_*.log` | server 日志(含 `fi_sampling ... ACTIVE`、路由 trace/histogram、非默认启动参数) |
| `logs/w2b_bench_*.log`、`logs/logs_bench_api1_*.txt` | 客户端压测日志 |
| `logs/micro_host_top_k.txt` | 每步 top-k 主机读取成本微测 |
| `logs/w2b_probe*.log` | B≥256 可达性探针(4k/1k/512 上下文;joint cell 在 512 上下文形态实际命中) |

## 2. 最终行为矩阵

### 2.1 开关与 env

| 变量 | 默认 | 语义 |
|---|---|---|
| `VLLM_HUST_FI_SAMPLING` | `0` | 总开关。`0` = 不 import 宿主/内核、不 patch(零差异) |
| `VLLM_HUST_FI_SAMPLING_SEED` | 未设 | 固定基准 seed;未设时取 `torch.initial_seed()`(随 `--seed` 可复现),每次调用推进 |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH` | `256` | joint cell 的 B 下限(W2 bench 实测拐点) |
| `VLLM_HUST_FI_SAMPLING_JOINT_MIN_K` | `32` | joint cell 的 k 下限(W2 bench:k=50 赢、k=1 输) |
| `VLLM_HUST_FI_SAMPLING_K1_ARGMAX` | `1` | `1`=k=1 走 argmax;`0`=k=1 留在 fork 链(位级 tie 保真逃生口) |
| `VLLM_HUST_FI_SAMPLING_TRACE` | `0` | 每步路由 trace + 周期性 histogram(证据用) |

### 2.2 分流矩阵(替换类内部,`decide_route`)

| 条件(自上而下) | 路由 | 依据 |
|---|---|---|
| 命中任一强制回退 | **fork 原链** | 见 §2.3 |
| 无 top-k 且无 top-p | **FI api1**(`sampling_from_probs`, block_v=1024) | W2 bench 表 A:ours 全 B 赢 1.12×→12.83× |
| top-k 全行 == 1 | **argmax** | W2 bench 表 C:k=1 ours 全 B 输(1.43×–6.35×),应直接 argmax |
| B ≥ 256 且 k_min ≥ 32 且 top-p 存在 | **FI joint**(`top_k_top_p_sampling_from_probs`, block_v=4096) | W2 bench 表 B:B≥256 调优后 1.28×/1.79× |
| 其余(k-only/p-only 大 B、小 B 重截断等) | **fork 原链** | W2 bench 未测该区间,保守留 fork |

判据参数集中 `fi_sampling_route.py` 顶部,注释引用 bench 来源;可用 env 覆盖以便 e2e A/B。

### 2.3 回退矩阵(命中任一即回退,且**不付任何设备同步**)

| # | 回退类 | 判定条件(代码锚点) |
|---|---|---|
| ① | per-request generators | `generators` 非空——FI 的 generator 只取 `initial_seed()` 且**不随调用推进**(与 FI 语义不同,W2 review 陷阱) |
| ② | processed logprobs 路径 | `logprobs_mode ∈ {"processed_logits","processed_logprobs"}`——FI 不返回截断后 logits/logprobs |
| ③ | batch-invariant 采样 | `vllm.envs.VLLM_BATCH_INVARIANT` |
| ④ | reduce-sample | `get_ascend_config().enable_reduce_sample` |
| ⑤ | async-exponential | `get_ascend_config().enable_async_exponential` |

另有两条 fail-open(非"回退类",但同样不出错):内核 vendor 包导入失败 → 单条 warning + 全程 fork 链;
内核调用抛异常 → 单条 warning + 该次及后续回 fork 链。vllm-ascend config 读不到时按"五类全命中"保守回退。

**顺序修正(本包由 e2e 证据倒逼)**:回退判定必须排在"读每步 top-k 主机值"之前,否则回退腿会白付一次
D2H 同步。初版顺序有误,§4 的第一对 api1 数据(+4.9% median TPOT)直接暴露了它,已修正并补守护测试
(`test_fallback_cells_never_read_top_k`、`test_small_batch_top_k_cell_is_sync_free`)。

## 3. e2e TPOT 证据

### 3.1 口径

- `vllm bench serve --backend openai-chat`,random 数据集,`--ignore-eos`,固定输入/输出长度,
  `--percentile-metrics ttft,tpot,itl --metric-percentiles 50,95,99`;每个 mode 独立 server 生命周期。
- 服务:`vllm serve /data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct --enforce-eager
  --max-num-seqs 512 --generation-config vllm`,单卡(卡 7),CWD=`/tmp`。
- **`--generation-config vllm` 是必需的**:该模型 `generation_config.json` 会覆盖服务端默认
  (`top_k=20, top_p=0.8`),不关掉的话"无截断"腿实际是重截断腿(首轮即踩到,日志
  `Default vLLM sampling parameters have been overridden ...`)。
- 每个 on 腿都核对 `fi_sampling sampling path is ACTIVE` 出现次数(off 腿必须为 0)。

### 3.2 api1 形态(无 top-k/top-p = vLLM 默认随机采样,主收益区)

形态:4k 上下文,1024 in / 256 out,`--num-prompts 256 --max-concurrency 64`,两轮独立生命周期。

| 轮次 | 腿 | median TPOT (ms) | mean TPOT (ms) | p95 | p99 | 输出吞吐 (tok/s) | 成功/失败 |
|---|---|---:|---:|---:|---:|---:|---|
| r1 | off | 113.48 | 111.66 | 114.67 | 115.83 | 535.77 | 256/0 |
| r1 | on | 111.49 | 110.19 | 113.70 | 114.97 | 542.94 | 256/0 |
| r2 | off | 111.87 | 110.13 | 113.08 | 114.50 | 542.86 | 256/0 |
| r2 | on | 113.54 | 111.66 | 114.47 | 115.61 | 536.01 | 256/0 |

两轮中位数(median over repeats):

| 指标 | off | on | 变化 |
|---|---:|---:|---:|
| median TPOT | 112.67 | 112.52 | **−0.14%** |
| mean TPOT | 110.90 | 110.92 | **+0.02%** |
| p95 TPOT | 113.88 | 114.08 | +0.18% |
| p99 TPOT | 115.16 | 115.29 | +0.11% |

**路由证明**:on 腿 `fi_api1` 命中 1300/1300 次(histogram `{'fi_api1': 1300}`,max_B=64),
off 腿 0 条 trace。即"无截断走 FI api1"在真实服务下 **100% 生效**。

**结论**:在该形态下 on/off 差异落在噪声带内(±0.2%),**没有可测收益**。这不是路由没生效(证据显示
100% 生效),而是采样的端到端占比太小:B=64 时 W2 单算子数据是 fork 链 ~1.27 ms vs ours ~0.35 ms,
即节省约 0.9 ms/步;而实测 TPOT ≈ 112 ms/步,采样只占 **≈1%**,理论收益上限 <1%,被运行波动吞掉。
**收益要显形,需要采样在 TPOT 中占比更大的形态**(更大 B:W2 数据显示 B=1024 时 fork 链 25.4 ms、
ours 2.0 ms,节省 23 ms/步,占比会显著上升)。这一点在本包的单卡 14B 上受 KV 容量限制无法达到
(见 §3.4)。

### 3.3 joint 形态(k=50 / p=0.95,大并发)

> **⚠️ 勘误(评审 F2,2026-09-09)**:下表两腿数据产自**排序修复(§4)之前的构建**——当时
> `forward_native` 每步先读 top-k 主机值再判定回退,on 腿(k=50,B≤122 全走 fork)每步白付一次
> D2H 同步。因此 162.17 vs 146.55 的差值**部分来自该已修复的实现缺陷,而非纯运行噪声**;
> 本表对比**作废**,不得用于任何性能归因。保留于此仅为存档完整。
> joint 腿不在最终构建上重跑:§3.4 已证实 joint cell 在真实部署形态(4k 上下文)不可达,
> 重跑只能复测一条永不命中的路径;其**路由可达性**结论(全 fork / B_max≈122)由 trace 与
> probe 证据支撑,不受本次降级影响。若未来启用短上下文形态,需按最终构建重测后另立报告。

形态:4k 上下文,1024 in / 256 out,`--num-prompts 512 --max-concurrency 256`,各 1 轮。

| 腿 | median TPOT (ms) | mean TPOT | p95 | p99 | 输出吞吐 (tok/s) | 成功/失败 |
|---|---:|---:|---:|---:|---:|---|
| off | 146.55 | 147.56 | 180.99 | 215.89 | 706.69 | 512/0 |
| on | 162.17 | 160.10 | 197.77 | 234.94 | 655.16 | 512/0 |

**路由事实(不受勘误影响)**:on 腿 histogram 为 `{'fork': 1500}`(1500/1500 全部 fork),
trace 显示 `route=fork B=8 k=50..50`——joint kernel 确实从未被触发:B 受 KV 容量限制只到
122(< 256 门槛),joint cell 不可达。

### 3.4 关键形态事实:joint cell 的可达性边界(实测)

同一台单卡 910B2 上,逐步压缩上下文以观察可达 batch 上限(`VLLM_HUST_FI_SAMPLING_TRACE=1` 的
`max_B` 与 histogram,证据 `logs/w2b_probe{2,3,4}.log`):

| 服务形态 | `--max-model-len` | 实测 max_B | joint cell | 证据 |
|---|---:|---:|---|---|
| 4k 上下文,concurrency 256 | 4096 | **122** | 不可达(全 fork) | probe2 |
| 1k 上下文,concurrency 512 | 1024 | **230** | 不可达(全 fork) | probe3 |
| 512 上下文,concurrency 512 | 512 | **512** | **可达**:`{'fork': 111, 'fi_joint': 64}` | probe4 |

即:**B≥256 需要把上下文压到 ~512 token**;常规 4k 部署下 KV 容量把 B 卡在 ~122,joint cell 永不可达。
这直接决定了本包的收益结构:

- **可拿的收益在无截断路径**(任意 B 都命中,无需大 B);
- **joint 分支需要"大 B + 短上下文"这种特殊形态**才可能显形,常规长上下文服务里它是死分支;
- 因此 joint 分支的价值是"按 W2 判据把已实测赢的区间接上",而非本形态下的性能来源。

## 4. 由证据倒逼出的实现修正(重要)

第一对 api1 数据(初版实现,`results/serve_api1_off.json` / `serve_api1_on.json`)是:

| 腿 | median TPOT | 说明 |
|---|---:|---|
| off | 115.18 | — |
| on | 120.77 | **+4.9%** |

该轮 on 腿**全部回退到 fork 链**(当时 `generation_config.json` 未被关闭,实际 served top_k=20),
也就是说:没有任何 FI kernel 运行,TPOT 却退化了 4.9%。根因是初版 `forward_native` 先做"每步读
top-k 主机值"再判定回退,导致回退腿每步白付一次 D2H 同步。修正后(回退先行、读值仅在大 B top-k cell
发生)重测,得到 §3.2 的 −0.14%/+0.02%。

单步读取成本微测(`logs/micro_host_top_k.txt`):B=64 → 0.068 ms、B=122 → 0.072 ms、B=512 →
0.098 ms。裸拷贝成本远小于观察到的 4.9%(≈5.6 ms/步),差额是同步对流水线的破坏——这正是"回退腿
必须零同步"这一设计约束的实测依据。

## 5. 验证门槛结果

| 门槛 | 命令 | 结果 |
|---|---|---|
| CPU 单测 | `cd /vllm-workspace/vllm-ascend-split-batch-hust && python -m pytest -q` | 全量 **77 passed / 7 failed**(失败均为 cascade 锚点漂移,属新环境适配项,与本包无关);**本包 fi_sampling 44 例全绿**(含 F1 修复 5 例 + env 容错 4 例) |
| lint | `ruff check .`(line-length 88) | **All checks passed** |
| NPU 冒烟 | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 python docs/evidence/w2b-fi-sampling/smoke_npu_fi_sampling.py` | **6/6 PASS**(`logs/logs_smoke_npu.txt`) |
| e2e TPOT | 实际执行序列:① `bash run_e2e_matrix.sh 7 joint 1`(05:30,**修复前构建**,产物已降级见 §3.3)→ ② `bash run_e2e_matrix.sh 7 api1 2`(06:26–06:41,最终构建,§3.2 的 4 份 JSON);首轮 api1(05:18,修复前)移入 `results/superseded/`。汇总:`python3 summarize_e2e.py` | §3.2(api1,有效)+ §3.3(joint,作废) |

NPU 冒烟覆盖:default-off 零替换(load() False + 类未变)、on 替换为子类、无截断路由 B=8、
joint 路由 B=256、k=1 路由 == argmax(B=256)、per-request generator → fork 链。

## 6. 遗留项与后续建议

1. **e2e 收益未显形(主遗留)**:无截断路径在 B=64 下采样仅占 TPOT ≈1%,理论收益 <1%,实测落在噪声带。
   要给出"e2e 正收益"结论,需要在采样占比更大的形态上复测:
   - 大 B(≥512)短上下文(如 512 ctx)以抬高 B;或
   - 更大词表/更小模型(采样占比相对上升);或
   - 逐算子层面已由 W2 证实的 12.83× 用"采样耗时占 TPOT 比例"换算成 e2e 上限,作为解析证据补充。
2. **joint 分支是死分支(形态受限)**:4k 上下文单卡 B_max≈122 < 256 门槛。若要让 joint 分支显形,
   需要短上下文大并发形态,或把 `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH` 下调——但下调前必须补
   B∈[64,256) 区间的单算子/e2e 证据(W2 bench 显示该区间 ours 输 1.33×–3.83×)。
3. **k=1 的 argmax 语义**:默认走 argmax,与 fork 链在**存在精确 tie** 时可能选不同下标;需要位级
   一致时设 `VLLM_HUST_FI_SAMPLING_K1_ARGMAX=0`(回退 fork 链)。当前未做 tie 专项对拍。
4. **RNG 语义差异**:FI 路径每次调用推进 seed(避免每步同流),但与 fork 链的
   `torch.Generator` 流不同 → 同 prompt 下 on/off 的**具体 token 序列不可逐位对齐**(分布等价,
   非逐位等价)。默认 `raw_logprobs` 模式下不影响正确性,但任何依赖逐位复现的验收需 off 腿做基线。
5. **vendor 目录 lint 例外**:`src/vllm_ascend_split_batch/fi_sampling/` 逐字复制自源包(便于 diff 溯源),
   已加入 ruff `extend-exclude`;升级源包时需重新同步并记录 hash。
6. **manifest 未翻 active**:证据链仍缺"e2e 正收益"(第 1 条),按 release.md §0 三项证据门槛,
   当前不满足翻 active 条件。`import_only` 保持。
7. **上游源包仍在变动**:本包 vendor 期间 `knowledge/surveys/sampling/fi_sampling/kernels.py`
   被并发更新(2026-09-09 04:53,review errata:补 kSCALE 溯源说明)。已按最新版本重同步;
   后续若源包再变需重新核对(算法未变,仅注释)。

## 7. 数字复核记录

- §3 全部 TPOT/吞吐/成功数由 `summarize_e2e.py` 从 `results/serve_*.json` 读取后誊入,
  无手工记忆值;两轮中位数用 `statistics.median`。
- 路由命中数(`fi_api1 1300/1300`、`fork 1500/1500`、`max_B`)来自 server 日志中插件打印的
  `[fi-sampling histogram ...]` 与 `[fi-sampling trace ...]`,可用
  `grep -o "histogram calls=.*" logs/w2b_serve_*.log` 复现。
- default-off / on 的激活判定用 `grep -c "fi_sampling sampling path is ACTIVE" logs/w2b_serve_*.log`
  (off 腿 = 0,on 腿 = 1),写在 `run_e2e_matrix.sh` 里作为跑测断言。
- 单步 top-k 读取成本由 `logs/micro_host_top_k.txt`(200 次平均,20 次预热)。

## 8. 评审修订记录(F1/F2,2026-09-09)

评审结论 needs_changes 的两项 major 修复如下;F3–F10(medium/minor)留待后续批次抽查。

### F1(major)宿主 import 失败崩进程 → 真 fail-open

- **缺陷**:`fi_sampling_plugin.install()` 把 `import vllm_ascend.sample.sampler` 写在
  try 块外;宿主模块缺失/改名(fork 演进)时 `load()` 直接抛异常,而它是 `vllm.general_plugins`
  入口——异常会打崩 vLLM 的插件加载器,整个引擎进程启动失败。同类隐患:`_fallback_flags()`
  在 try 外调用 `_batch_invariant()`。
- **修复**:两处 import/探测全部收进 try;任何失败 → 单条 warning + 返回 False /
  按"五类回退全命中"路由 fork 链。
- **新增测试(5 例)**:`test_install_is_fail_open_when_host_module_is_missing`、
  `test_load_is_fail_open_when_host_module_is_missing`、
  `test_load_process_survives_a_missing_host_module`(子进程级:宿主模块被毒化后
  `load()` 返回 False、进程退出码 0)、`test_fallback_flags_fail_closed_when_batch_invariant_probe_raises`、
  `test_fallback_flags_fail_closed_when_ascend_config_raises`。

### F2(major)joint 腿证据产自修复前构建 → 降级 + 报告更正

- **事实**:joint 两腿 JSON(05:30/05:43)先于 §4 排序修复落地,on 腿(全 fork 路由)每步
  含一次多余 D2H 同步;原报告把 +10.7% 归因为"运行噪声"不成立——部分差值来自该缺陷。
- **处置(降级,不重跑)**:§0 与 §3.3 已标注**作废**;§3.3 的路由可达性事实(全 fork、
  B_max≈122、joint cell 4k 形态不可达)由 trace/probe 证据支撑,继续有效。不在最终构建上
  重跑 joint 腿:该 cell 在真实部署形态永不命中,重跑无信息量;若未来启用短上下文形态,
  按最终构建重测后另立报告。
- **报告更正**:§1.2 JSON 份数口径(6 份参与 + 2 份 superseded,其中 joint 2 份已降级);
  §5 e2e 复现命令改为实际执行序列(`joint 1` 修复前构建 + `api1 2` 最终构建),原
  `run_e2e_matrix.sh 7 {api1,joint} 2` 的写法暗示的一次性干净跑测与事实不符。
- **复核**:§3.2 api1 全部数字已用 `summarize_e2e.py` 对 4 份最终构建 JSON 重放,逐位一致。

### 复审追加(同日,5 项残留全部落实)

复审在确认 F1/F2 主体到位后指出 5 项残留,处置如下:

1. (major)`load()` 的 env 门 `int()` 解析无防护——`VLLM_HUST_FI_SAMPLING=true` 即崩。
   **修复**:新增容错 `_env_int`(非法值 → 每变量一条 warning + 默认值),`_env_flag`/
   `_joint_min_batch`/`_joint_min_k`/`_k1_argmax` 全部改走它;默认关闭方向(fail-safe)。
2. (major)`install()` 尾部阈值解析在 try 外且类替换已落盘 → 半安装态。**修复**:全部旋钮
   改为**替换前**解析(结构化消除半安装窗口),logger 使用预解析值。
3. (minor)`forward_native` 每步旋钮/seed 解析无防护。**修复**:同 1(`_env_int` 容错,
   warning 按变量去重避免逐步刷屏);`ENV_SEED` 非法值回退 `torch.initial_seed()`。
4. (nit)`torch.softmax`/argmax 在 fail-open try 外——**保留不修**:这是 torch 本体调用,
   若它失败,fork 链(`super().forward_native`)同样依赖 torch,救援无意义;评审亦判定
   "救援价值有限"。
5. (minor)§0 质量门残留旧计数"73 passed"与 §5 矛盾。**修复**:已改为 fi_sampling 包
   44 例全绿 + 全量 77/7 口径。
6. 新增 4 例 env 容错测试(garbage enable 子进程存活 / 阈值回退默认 / 带垃圾旋钮的
   install 完整成功 / garbage seed 推进)。

7 个 cascade 测试失败经复审确认与本次改动面无耦合,属既有宿主锚点漂移(MAIN 队列 #2/#4)。
