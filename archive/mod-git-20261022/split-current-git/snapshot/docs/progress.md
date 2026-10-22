# 进度日志与计划

时间线基于 git log 事实;计划一节是开放问题,动手前先在团队对齐。

## 1. 时间线

### 阶段 0:split-batch legacy → 契约提取

- 历史 split-batch patch(ascend PR 280-283 / core PR 263/273)效果一般,
  按 AGENTS.md 决策**不移植 legacy patch**,原始实现存档于 `provenance/`(只读)。
- 提取 host-independent 组件为 inert 契约描述符(merge PR #3,
  `2f55b42`):纯 planner(`plan_dual_pad`)+ manifest 描述符,
  `import_only`,等待宿主提供 `vllm.graph.runtime-key.v1` 等 seam
  (提案见 HOST_CONTRACT.md)。

### 阶段 1:cascade attention 插件(eager, default-off)

- `7bddf66` default-off `vllm.general_plugins` 插件 + `256eed9` CPU 单测。
- eager 两段式:stage-1 共享前缀摊平读 + stage-2 suffix + LSE merge,
  Tier0(bf16)/Tier1(fp32,依赖 ascend_kernel wheel)双精度层。

### 阶段 2:图模式(graph twin)

- `8f8df73` 图模式两段式 cascade(default-off):twin graph 按
  `BatchDescriptor` 侧表存取、update-before-replay、bucket-static workspace。
- 系列修复:图池中间量生命周期(09264a1/e9fd67a/b99963c)、twin miss
  fail-open(672a255)、redundant update 跳过(afd487d/39f7616)、
  microbatching 互斥(7484a27/134d172)。

### 阶段 3:自适应 gate(W2)

- `2f041d6` 按 (B, shared) 桶启动期 micro-bench,亏区自动回落 FULL。
- 稳定性线:探针块表隔离(9899f00,aicore 踩内存修复)→ 子进程隔离
  (b479508)→ BNSD 探针 + OOM skip(7858229/904fa93/a3cbbf4)→
  dispatch 级消费(f423a58,HEAD)。

### 阶段 4:工程化与发布准备(2026-09,本轮)

- 对照 bidkv 打包发布指南(vllm-hust-docs 仓库)完成打包验证:wheel 含
  manifest 与双 entry point、`extension_bundles` 发现路径打通。
- 修复 manifest `activation.environment` 占位符隐患(注入语义化),新增
  版本一致性与值域守护测试。
- 建立 [release.md](release.md) 发布流程与宿主升级核对清单。
- 建立本 docs/ 知识库与仓库级 AGENTS.md。
- kernel wheel 软依赖 + fail-open(W4 收尾,2026-09):`b1e7e61` 声明
  `kernels` extra(原钉 `ascend-kernel==2026.3.9`,0.1.1 起钉 `2026.9.26` 并由 Release 附件提供);wheel 探针守卫——
  缺失/注册失败时 cascade 整体禁用 + 单条 warning(探针/eager gate/twin
  capture/gate bench 子进程四处守卫),单测
  `tests/test_cascade_fail_open.py`(含删 wheel 子进程等价场景);
  manifest 保持 `import_only`,翻 active 三项证据归档于工作区
  `knowledge/evidence/cascade/`(仓库外,不入库)。

## 2. 现状(2026-09-09)

- 分支 `feat/cascade-attention-plug`;W4 后 cascade 两个 carrier 已翻 **active**
  (`8751839`,证据复审通过),planner 保持 `import_only`;能力仍 default-off。
- **基线切换(2026-09-09)**:工作区默认环境换为 conda `hust`——vllm-hust v1
  (`0.28.1.post1.dev143`)+ vllm-ascend-hust main(`0.25.1rc2.dev125+hust`)++
  CANN 9.1.0。全部锚点在新宿主重核存活(runner 五方法经 `GPUModelRunner`
  基类继承,位于 `vllm/worker` 侧;`acl_graph.py` 四符号同模块;spec_decode/
  eplb 只缺 `scipy`+`decorator` 依赖,已补)。manifest `host.version_range`
  重钉为实测点 `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`
  (packaging 有序比较符拒 local,故用点 `==`);`extension check` 由
  INCOMPATIBLE 转 **compatible+degraded**。
- **cascade 降级回 `import_only`**(2026-09-09):W4 的 active 翻转基于
  0.23.0rc1 三项证据,不跨宿主继承;按纪律降级,回 active 路径 = 在本基线
  重跑 release.md 三项证据(default-off 零回归冒烟 / 正确性对齐 / 性能对比)。
  kernel wheel 已按 CANN 9.1 重编并全绿(S1 逐 bit、30/30、54/54、56/56、
  fastpath、S4 10/10),证据见 cascade-merge-op `73bf96a`。
- e2e 证据锚点见根 README(16k 段 −21%~−38% 等,kernel README §6 有单算子锚点);
  测量报告在 `knowledge/evidence/c3-legacy/`(工作区)。**换环境后性能数字降级为参考。**
- **新基线三项证据重验(2026-09-09)**:证据① default-off 冒烟 **PASS**;证据②
  正确性 **部分**——图模式车道首次未 shim 跑通(原为引擎无法启动),确定性 0/64、
  自然答案 63/64 通过,但强制续写形态发散 21–22/64 高于历史 ≤8/64 口径(字面形态
  1/64 通过;发散臂集合修复前后一致,非修复引入);证据③ 性能 **部分**——拓扑复现
  (4k 亏 +12.3%/+6.7%、8k −3.5%/−13.5%、16k −20.2%/−31.6%,区间均分离),但
  `(64,4096)` 亏损未被自适应 gate 覆盖(gate 微基准判 `on`)。**故 cascade 保持
  `import_only`,不翻 active**;详情 `knowledge/evidence/cascade/
  EVIDENCE.md`。
- **D1–D4 宿主签名漂移已修复**(`ddc0120`):`_capture_cudagraphs` 缺 `profiler`、
  `_update_full_graph_params_if_needed` 多传 `positions`、`update_graph_params` 多传
  `num_dcp_pcp_tokens`、`_model_forward` 参数约定。所有包装器改 `*args/**kwargs`
  原样转发 + try/except 单次 warning 委托(fail-open 契约兑现);新增
  `tests/test_host_signature_drift.py`(13 例,含变异测试证明有效)。
- **G6 gate 覆盖缺口已修复**(`c4121e2`,2026-09-10):证据③暴露的 `(64,4096)`
  亏损格 bench 判 `on` 不回落。归因:gate 微基准探测用逐请求不相交 block table
  (W0 防故障约束),测的是无 prefix-cache 复用世界,4k 共享前缀在引擎内可被 FULL
  路径经 L2 复用,cascade 流量优势归零;且 `(64,4096)` 与 `(32,8192)` 留档 bench
  margin(+10.7% vs +10.9%)不可分离,单一阈值无解。修复:分档 margin——
  `P≤4096` 桶要求 ≥25%(`SMALL_PREFIX_MARGIN`,env 可调),其余桶维持 2%;
  默认 `MIN_PREFIX=8192` 不 bench 出小桶,默认行为零变化。验证:留档 8 格回放
  8/8;NPU 4 腿重验(预注册判据)——4/4 腿 `(64,4096)→OFF`,p420_b64 亏格收回
  (+6.7%→+2.1%),8k/16k 赢格保留(−14.5%/−32.5%),`(128,4096)` 大 margin 格
  不受影响,0 TypeError;cascade key hit 379→254(Δ≈125,step 级关闭佐证)。
  证据:`knowledge/evidence/cascade/section3-performance.md`
  + `logs/v1-ev3-gatefix/`(含跑前预注册)。**G6 关闭;翻 active 剩余阻塞 = 缺口 A**
  (stand-in 形态正确性口径,等真模型或团队决策)。
- **W3.1 fi_gelu(gelu_and_mul triton 融合)已结题**(`51f5818`,2026-09-10):
  exact-erf 融合 kernel(29 测)+ default-off 接线(`VLLM_HUST_FI_GELU`,patch
  `GeluAndMul.forward_oot`,13 测)+ eager/图/e2e 三段 bench。结论:**能力落地、
  默认关闭、不建议生产启用**——triton eager launch ~110µs 主导(eager 负结果);
  图模式 kernel 可捕获且 replay bit 精确,B=64 单算子 −31%(23.7 vs 34.2µs)但
  e2e A/B 噪声带内不可判(Δmedian −0.311% < spread 1.95%,预注册投影 0.2–0.6%)。
  报告 `docs/evidence/w31-fi-gelu/REPORT.md`;附带发现宿主基线缺陷
  FULL_AND_PIECEWISE 启动崩(见 pitfalls §1.3)。
- CPU 门槛:`pytest -q` **153 passed**(2026-09-10,+29 gelu kernel +13 接线)
  + `ruff check` 干净。
- **cascade 翻 `active` + release.md §3 启用验证通过**(2026-09-11):manifest 两个 cascade
  carrier(`cascade_plugin:load` / `cascade_graph_plugin:install`)由 `import_only` → `active`,
  planner 保持 `import_only`;三项证据在新基线齐备(真模型 6/64 见
  `section2-correctness.md`,gatefix 见 `section3-performance.md`)。
  `extension check` 转 **compatible+configured**,`run --dry-run` 注入
  `VLLM_ASCEND_ENABLE_CASCADE_DECODE/GRAPH=1`;真模型 serve 启动日志 2 条
  `cascade plugin loaded (gate=1, graph_gate=1, kernel_wheel=ok)`,cascade twin
  `capture body SUCCESS` ×144、0 TypeError,`/health` 200、chat 冒烟 200。发现:cascade
  图车道在 `--gpu-memory-utilization 0.85` 下需把 `cudagraph_capture_sizes` 收到
  `[32,64,128]`(默认全网格 twin capture 触发 NPU OOM → capture 级 fail-open)。CPU 门槛
  `pytest -q` **190 passed** + `ruff check` 干净。详情
  `knowledge/evidence/cascade/section4-active-enablement.md`。

### 阶段 5:fi_sampling 采样接入(W2b,2026-09-09)

- 落点:`fi_sampling_plugin.py`(`vllm.general_plugins` entry `fi-sampling`,
  类级替换 `AscendTopKTopPSampler`)+ `fi_sampling_route.py`(纯逻辑分流/回退)
  + vendor 目录 `fi_sampling/`;开关 `VLLM_HUST_FI_SAMPLING`(默认 0)。
- 分流:无截断 → FI api1;k=1 → argmax;B≥256 且 k≥32 且 top-p → FI joint;
  其余 → fork 链。五类强制回退(per-request generators / processed_logprobs /
  batch-invariant / reduce-sample / async-exponential)全部落地且**零设备同步**。
- e2e 证据(见 `docs/evidence/w2b-fi-sampling/REPORT.md`):无截断 on/off 两轮
  median TPOT 差 **−0.14%**(噪声带内;采样仅占 TPOT ≈1%,路由 1300/1300 已确认
  生效);joint cell 在 4k 上下文单卡形态 B_max≈122 < 256 门槛而**不可达**
  (需 ~512 上下文),即 joint 分支是安全保留的死分支。**e2e 正收益未显形,
  manifest 不翻 active。**
- 修正:初版"先读每步 top-k 主机值再判回退"导致回退腿 +4.9% median TPOT,
  已改为回退先行(由 e2e 证据倒逼,见 REPORT §4)。
- **评审修复(2026-09-09,needs_changes 的 F1/F2)**:
  - F1(宿主 import 失败崩进程 → 真 fail-open):`install()` 的宿主模块 import 与
    `_fallback_flags()` 的 batch-invariant 探测收进 try,失败 → 单条 warning +
    回 fork 链;新增 5 例 fail-open 测试(含子进程级证明)。
  - F1 追加(复审 5 项残留全部落实):全部 env 整型解析改容错 `_env_int`(非法值 →
    单条 warning + 默认值,覆盖 enable/阈值/k1/seed),`install()` 旋钮解析前移
    消除半安装窗口;新增 4 例 env 容错测试。fi_sampling 包 44 例全绿。
    `torch.softmax` 不纳入 fail-open(评审 nit:fork 链同样依赖 torch,救援无意义)。
  - F2(joint 腿证据产自修复前构建 → 降级):§3.3 对比作废(不支持"纯噪声"旧归因),
    路由可达性事实保留;REPORT §1.2 JSON 份数、§5 复现命令改为实际执行序列;
    api1 数字已用 summarize_e2e.py 对最终构建 JSON 重放复核。详见 REPORT §8。
  - F3–F10(medium/minor)留待后续批次。修复后整体原子提交。

## 3. 计划与开放问题

> 优先级以工作区 AGENTS.md 为准(flashinfer 移植/e2e 为主线,本仓库是承载壳)。

1. **翻 `active` 的证据门槛**:default-off 零回归冒烟、正确性对齐、性能对比
   三项归档后,按 release.md 第 3 节走启用验证,再改 `implementation.status`。
2. ~~**kernel wheel extras 联动**~~(已完成,见阶段 4 W4 条目):`kernels`
   extra 已声明,fail-open 守卫与单测已落地。
3. ~~**宿主 seam 上游化**~~(**2026-09-11 关闭**:不做上游 PR/上游化,见工作区
   AGENTS.md 硬性约束与 §1 第 6 条):split-batch 四协议**永久保持**插件侧弱契约
   (`protocols[].version_range = null`,理由见 release.md §0),不再推动 vllm-hust
   提供 typed contract。
4. **发布渠道**:证据齐后走 `uv publish` 到 pypi 或内部索引;发布前补
   release.md 第 2 节的隔离安装冒烟自动化。
5. **fi_sampling e2e 收益显形**(W2b 遗留):无截断路径在 B=64 下采样仅占
   TPOT ≈1%,需在采样占比更大的形态复测(大 B 短上下文 / 更小模型 /
   用单算子比例解析外推);joint cell 需 ~512 上下文才能达 B≥256,若要在常规
   形态显形须下调 `VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH`,但下调前须补
   B∈[64,256) 区间证据(W2 bench 显示该区间 ours 输 1.33×–3.83×)。

每条动手前:更新本文件状态,证据落到 `docs/evidence/`(新建)或本文件。
(2026-09-11:**不写 PR 描述**——本仓不做上游 PR,改动只本地落地。)

## 2026-09-11 fia-demask（decode FIA 去掩码）

- 新 carrier `fia_demask_plugin:load`（**import_only**，开关 `VLLM_HUST_FIA_DEMASK=1`，
  observe `VLLM_HUST_FIA_DEMASK_TRACE=1`），op 面 wrapper（FIA v1 + .out + workspace 查询），
  纯 decode 谓词（TND + sparse_mode=3 + pre/next=INT_MAX + 方形 int8 mask + Q_S=1/请求），
  fail-open 检查 / fail-closed 安装，default-off 零差异。
- 三级验收全绿：CPU 35 tests；逐 token 等价 8/8；e2e decode 账 B32 **+3.10%** / B64 **+4.61%** TPOT
  （r1 预热离群对称，热轮腿内 0.2%）。证据：
  `profiles/qwen14b-instruct-hotspot-20260910/probe-fia/E2E-mask-removal.md`。

## 2026-09-11 fia-demask 翻 active（独立 bundle）

- `org.vllm-hust.fia-demask` carrier 翻 active + §3 启用验证全绿：check/enabled/
  dry-run 注入 `VLLM_HUST_FIA_DEMASK=1` / 真模型 serve（与 cascade enable 态共存，
  capture_sizes 封顶）→ `de-mask applied` + `ACTIVE` 证据行 → /health → 冒烟 OK，
  0 fail-open。补账：六格 TPOT +3.3~+10.1%（geomean +6.43%）、e2e 吞吐 +3.52%。
- 证据：`profiles/.../probe-fia/E2E-mask-removal.md` §3.1、`serve_demask_s3.log`、
  review 材料 `docs/evidence/fia-demask-review.md`、~~上游素材~~ `docs/upstream/fia-decode-demask.md`
  （**2026-09-11 冻结**：不做上游 PR，该素材只作技术论据留档，不再维护）。

## 2026-09-11 rope-fix（fork RoPE 三缺陷的插件侧 carrier + 反向漂移守卫）

- 新 carrier `rope_fix_plugin:load`（**import_only**，开关 `VLLM_HUST_ROPE_FIX=1`，
  bundle `org.vllm-hust.rope-fix`）：把 fork 三处真实缺陷做成插件侧一等公民——
  ① llama3-scaling 接线到 Ascend rope kernel（`op_registry_oot["Llama3RotaryEmbedding"]`
  创建 + 10 参签名子类）、② mrope 补 `triton_mrope` 第 9 参 `is_neox_style`
  （镜像 `forward_triton` 的 AST 守卫）、③ YaRN `truncate` 缺省对齐 vLLM/HF（`True`）。
  机制：包裹 `register_ascend_customop`（原函数先跑、返回后覆写 registry 条目；
  包装体同时重绑 `worker.py` 等直接引用点），默认关时零 import/零 patch/零日志。
- CPU：`pytest -q` **254 passed**、`ruff check .` 零告警（新增 28 例：17 carrier +
  10 守卫 + 1 manifest；含真宿主注册路径集成例与子进程 import 纯净性例）。
- **守卫双向自测**（宿主树零改动，`/tmp` 副本注入变异）：三条反向断言在"上游已修"
  变异下报 `upstream fixed defect ①②③ → drop the corresponding override`，在
  "锚点漂移"变异下报 `anchor drifted → re-audit HOST_CONTRACT`。
- **NPU 探针 carrier 形态复现**（卡 7 + flock，**不用 PYTHONPATH 覆盖**）：
  四变体 gen/kernel/`kernel_reachable`/e2e **全 PASS**、`kernel_reachable` 齐
  "生产类接线到 triton"、总偏差 **3.125e-2**（cache bf16 底噪）= fork-fix 参考值；
  env 未置的对照腿逐字复现修前三断点（llama3 落 `CustomOp.forward_oot`、
  mrope `TypeError: triton_mrope() missing 1 required positional argument`、
  yarn 公式分歧 5.09375）。
- 证据：`docs/evidence/rope-fix/REPORT.md` + `comparison.md`（四腿对比表）+
  `guard-selftest/`；设计说明 `docs/design/rope-variant-defects.md`；
  宿主 seam 与锚点 `HOST_CONTRACT.md`「rope_fix component」；
  宿主升级核对项 `docs/release.md` §4。manifest 保持 `import_only`（三证据未齐）。

### 复审收尾（同日，9 条 F 全部落实）

- **F1（310P fail-open 缺口）**：①的 `forward_oot` 不再硬编码 910 类，改为在
  `__init__` 解析并缓存 `op_registry_oot["RotaryEmbedding"]`（本 build 实际用于基础
  rope 的实现），310P 档因此跟随 `AscendRotaryEmbedding310` 自己的
  `npu_apply_rotary_pos_emb` 路径；键上非类时退回模块级 `AscendRotaryEmbedding`。
  ⚠️ 本机是 910B2，**310P 真机未实测**（构造级 mock 覆盖），进入 310P 验证域时升级复核。
- **F3（`triton_mrope` 静默 None）**：kernel 绑定改取 fork 模块自身的
  `triton_mrope`（与调用点同源）；为 `None`（无 triton）时**不安装 ②** + 单条 warning，
  避免把 load 期 fail-open 变成模型 forward 期的 `'NoneType' object is not callable`。
- **F4（半安装状态）**：包装体已装后覆写抛错 → **回滚包装体**，`stats()` 与日志措辞与
  真实状态一致。
- **F2**：补"注册先于 `load()`"（worker 子进程真实次序）用例 → 三键立即覆写、
  `installed is True`。
- **F8/F9（守卫）**：宿主树不可发现时 **fail 而非 skip**；定位回退改为基于本文件位置的
  确定性路径；补 `worker.py` 仍直接 `from ... import` 且仍调用的源级断言；镜像断言在
  "宿主已补第 9 参"时改报 `upstream fixed defect ②`。
- **F5/F6/F7（证据/文档）**：`compare.py` 的 mrope 行改打印真实判据
  （`包装正常返回`，不再空真 `all([])`）；证据计数统一；`extension_version` 一致性扩到
  三个 bundle manifest。
- 复跑：CPU `pytest -q` **261 passed** + `ruff check .` 零告警；NPU 卡 7（flock）carrier 主腿 +
  atb 腿 + default-off 对照腿全部 rc=0，四变体判定与修前一轮**逐字段相同**（无回归）；
  守卫自测新增 no-host 案例（fail 而非 skip）。
- 复审后仍未做（如实）：310P 真机核对（无该卡）、服务级真模型 A/B（无 checkpoint）、
  entry point 元数据刷新（纪律禁 pip）。

## 2026-09-13 zerocost-wiring §3 启用验证（**阻塞，未翻 active**）

- 新 carrier `zerocost_wiring:load`（**import_only**，bundle `org.vllm-hust.zerocost-wiring`，
  开关 `VLLM_HUST_FI_PREFILL_OUT=1` + `VLLM_HUST_SKIP_COS_SIN=1`；交付面 `e308e78`）。
- §3 阶梯与 default-off 腿全绿：`inspect`/`check`/`status`/`enable`/`run --dry-run` 注入正确；
  default-off 腿 `/health` 200 + 2 chat 200 + 日志零 zerocost 痕迹；收尾 HBM 回 5%。
- **阻塞**：§3 正式 serve（`vllm-hust-ext run -- vllm serve`，本工作区**必然** co-enable
  cascade graph）下 **① 静默 fail-open**（`zero-cost wiring ① refused … a host anchor moved`），
  仅 ② ACTIVE。根因：`zerocost_wiring._host_func` 只支持**一层**插件 wrapper——`graph_gate=1` 的
  `cascade_graph_plugin.install()` 在 `cascade_plugin` 的 pass-through wrapper 之上再叠一层
  （`install.<locals>.forward_fused_infer_attention`），其闭包指向的是**上一层插件 wrapper** 而非宿主文件函数
  ⇒ 解析 `break` → ① 整项回滚。隔离腿（plain `vllm serve` + 两 env，cascade 关）复现 Y1 条件：
  **①+② 均 ACTIVE**。⇒ Y1「cascade wrapper 在场时 ① 仍 ACTIVE」成立但**仅限 decode gate**（`serve2.log`
  实测 `gate=0, graph_gate=0`）。
- **处置**：按 release.md §0/§3 与硬纪律**不翻 active**，manifest 保持 `import_only`；manager 侧 `enable`
  已 `disable` 回退；`_host_func` 修复（穿双层 wrapper）与文档纠偏列为未决 B1/B2。
- CPU：`pytest -q` **272 passed**、`ruff check .` 干净。证据：
  `docs/evidence/zerocost-activation-20260913.md` + `docs/evidence/zerocost-activation-20260913/raw/`。

## 2026-09-13 zerocost-wiring `_host_func` 多 wrapper 修复 → **§3 通过，已翻 active**

- **修法**：`zerocost_wiring._host_func` 的判据从「只跳一层、cell 的 `co_filename == 宿主文件`」改为
  「遍历整个 `__closure__` 图，要求**恰好一个**可达函数其 code 属于宿主文件」；先经 `_unwrap_callable`
  归一化 bound method / partial。0 命中（锚点漂移/断链）或 ≥2 命中（歧义）都 raise ⇒ 沿用既有 fail-open
  契约（一条 warning + 整项回滚）；仍是 `host_func.__code__ = new_code` 原地替换，未 `setattr` 到类上。
  不动 ②、不动 cascade / cascade_graph 插件。
- **单测**：`tests/test_zerocost_wiring.py` 新增 6 项——两层/三层叠加 wrapper 仍能溯源（对照旧逻辑
  在两层即 `RuntimeError`）、双层叠加下端到端 `_zc_fi_out` 到达 adaptor 且恒等拷贝被跳过、
  同文件双候选（歧义）与断链（0 命中）都要拒。CPU：`pytest -q` **277 passed**、`ruff check .` 干净。
- **§3 重跑（真模型，卡 7）**：`inspect`/`check`/`status`/`enable`/`run --dry-run` 全绿
  （`activation_ready=true`、blocker=`null`、states 含 `configured`、enable rc=0）；
  default-off 腿 `/health` 200 + 2 chat 200 + **日志零 zerocost 痕迹**；
  **ON 腿（`vllm-hust-ext run`，co-enable cascade graph，`gate=1, graph_gate=1`）**
  ⇒ **① 与 ② 均 ACTIVE、0 fail-open(refused)、`/health` 200、2 chat 200**。
  日志唯一 ERROR 是主动 kill 引发的关停竞态（两条 chat `200 OK` 之后 `EngineDeadError`），非 zerocost 缺陷。
  收尾：三腿子树 `residual children: <none>`、卡 7 HBM 回 5%、manager enable 态 `disable` 回退到任务前。
- **处置**：三项证据齐备 + §3 门控生效 ⇒ 翻 `implementation[0].status: import_only → active`；
  `test_zerocost_bundle_is_*` 守卫同步改为断言 `active`。
- 证据：`docs/evidence/zerocost-activation-20260913.md`（追加「修复后 ON 腿」一节，保留上一程失败记录）
  + `docs/evidence/zerocost-activation-20260913/raw-fix/`。

## 2026-09-26 cascade 准入与默认关闭隔离（PR #5 评审落地）

- **来源**：外部评审 PR #5（`codex/qwen35-cascade-qualification` @ `c6066c35`，base 为
  `224efa2`）。该 PR 的 diff 在本分支不适用（`git apply --check`：5 文件里 3 个 hunk 冲突；
  `git merge-tree` 3 处内容冲突），故按等价形态在本地实现，不直接合并。
- **默认关闭时的宿主隔离**（PR 的核心诉求）：`load()` 在
  `VLLM_ASCEND_ENABLE_CASCADE_DECODE` 未置 1 时只打启动标记行就返回——不注入
  `envs.env_variables`、不 import 宿主模块、不装 fail-open shim、不探测 kernel wheel、
  不替换 attention/graph 条目。标记行改为
  `cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=not-probed)`：既不谎报
  `unavailable`（没探测就说没探测），又保留"插件确实跑过"的判据。契约落
  `docs/release.md` §6。
- **query 布局准入**（PR 的第二诉求）：`_use_cascade_attention` 新增两条 fail-closed 守卫——
  ① 任一请求的 query row ≠ 1（MTP 校验、chunked prefill）即拒；② `speculative_config`
  非空即拒（k>1 布局既未支持也未标定）。`cascade_runner_patch` 同步新增
  `_spec_decode_active()`，投机配置下跳过 cascade twin 捕获（标准 FULL 图照常服务）。
  **这是边界，不是 MTP 支持**；split-batch/dual-pad 的 precheck 早已同样 fail-closed
  （`planner.precheck_reason` → `speculative_decode_conflict`）。边界说明见根 README
  「Speculative decoding boundary」。
- **CPU 门槛**：`pytest -q` **440 passed**（新增 6 例：关闭态零改动、关闭态不探测 wheel、
  标记行 `not-probed`、多 query 拒绝、投机配置拒绝、twin 捕获跳过；另改写 4 例既有用例以
  对齐"先置 env 再 load"的真实次序）、`ruff check .` 干净。
- **登记（未做）**：`ruff format --check .` 存量不过（25 文件，含本轮改的 4 个；`9951e41`
  上同样不过）⇒ CI 模板原样启用会红，启用前需先做一次纯格式化提交。见
  `docs/release.md` §7。

## 2026-09-26 插件包上 PyPI（0.1.0 → 0.1.1）+ 算子轮改走 Release 附件

- **0.1.0 首发**：组织页 pending publisher（OIDC Trusted Publishing，无 token）→
  `workflow_dispatch` target=`pypi` → PyPI `pypi.org/project/vllm-ascend-split-batch/0.1.0/`。
  回执见 `docs/release.md` §11.6。用**非 editable** 的 site-packages 做管理器发现已实测
  `activation_ready=True` / `blocker=None` ⇒ 此前"管理器仍按 import_only 拒绝启用"闭环。
- **首发前拦下的两个问题**：① `publish.yml` 缺 `vllm-hust-ext` 的 git 安装（该包无 PyPI 发布，
  而 test extra 钉 `==0.2.0.dev0`）⇒ runner 上必然解析失败；② **内网地址泄漏**：
  `docs/evidence/w2b-fi-sampling/logs/` 下 10 个日志含未遮蔽的私网地址（cascade 目录当时已遮蔽，
  这批早先提交的漏了）⇒ 一并遮蔽 + 新增 `tests/test_public_hygiene.py` 把判据机械化。
- **0.1.1**：0.1.0 不可覆盖，而它的 `kernels` extra 钉着任何索引都查不到的 `ascend-kernel==2026.3.9`
  ⇒ 修钉子只能发新版。现钉 `2026.9.26`，来源 = 算子仓 GitHub Release 附件；
  `pip install "vllm-ascend-split-batch[kernels]" --find-links <该目录>` 解析已实测通过。
  回执见 `docs/release.md` §11.7。
- **算子轮补齐**（算子仓 `ca5bd9d`，已推送）：根 `LICENSE` 补 CANN OSL 2.0 全文（定性依据：
  `csrc` 40 处 `#include "catlass/..."` ⇒ `.so` 是该模板树的衍生件，§3.3 要求随附协议）、
  vendored 树补 `LICENSE`、根 `NOTICE` 写明判定链与不含项清单、wheel 内置协议文本、
  METADATA 的 license 从误写的 `BSD 3 License` 改为实际协议名、`config.ini` bump 至 `2026.09.26`。
  **重建等价性实测**：重跑 `./build.sh` 后 `_C.so` / `libascend_kernel.so` 与 `2026.9.16` 轮
  md5 逐字节相同（`add6e6951d253328` / `ca8de2d70fe0504d`）⇒ 只改打包元数据与许可，目标码未变。

## 2026-09-27 0.1.3 已发布：`host.version_range` 放开为有界窗口

- **裁定**（用户）：点钉让插件在除一个 build 之外都无法验证、连"哪里会出问题"都测不出来
  ⇒ 改为**有界兼容窗口** `>=0.25.1rc2.dev125,<0.25.2`；窗口内失败按**宿主侧发现**归因
  （理由与纪律：`docs/release.md` §0.3；归因落点 `knowledge/host-findings/`）。
- **改动**：4 个 manifest 的区间 + UpdatableGraph 重放接缝修复 + 5 个发布前测试/文档提交；
  阈值守卫 3 例（两端点在区间内、0.24/0.26 在外、有序比较符内禁 `+local`）。
- **发布**：PyPI `0.1.3`（Publish run `36301111078`）+ GitHub Release `v0.1.3`
  （Release run `36301712587`，附件 == PyPI 字节）。**零凭据**：`publish.yml` 增 tag 触发
  （推 `v*` ⇒ target=pypi，仍走 OIDC）、`release.yml` 用 run 自带 token 建 Release、
  上传前查 PyPI 已存在则跳过（幂等）。
- **发布字节复验**（目标栈 dev605 装 PyPI 的 0.1.3）：`check` compatible、`run --dry-run` exit 0、
  4 并发 8/8（0.2–0.3 s）、`capture body SUCCESS` 96、`cascade update ran=True` 4、0 TypeError。
- **如实登记**：`v0.1.3` tag 曾从 `c8a1891` 移到 `9a9beb1`（同 tag 名、内容仍 0.1.3，为带上
  Release 工作流）—— 属移动已发布 tag 的先例，后续应避免。

## 2026-09-26 0.1.2 已发布（让 `test` extra 摘钉生效）

- **动机**：0.1.1 及以前的 `test` extra 钉着不在任何索引上的 `vllm-hust-ext==0.2.0.dev0`
  ⇒ 第三方 `pip install "vllm-ascend-split-batch[test]"` 必然失败；PyPI 不可覆盖 ⇒ 只能发新版。
- **动作**：版本 0.1.1 → 0.1.2（7 处：pyproject / 4 个 manifest / 2 个 `__version__`）→ 合入 `main` →
  `workflow_dispatch`（`target=pypi`，OIDC Trusted Publishing）→ run `36255715877` 16 步全 success。
- **判据（本次真的验到了）**：干净 venv 打 `pypi.org/simple`，`pip install "...[test]"` 解析到
  `vllm-ascend-split-batch-0.1.2`；反向对照 `[test]==0.1.1` 仍报
  `No matching distribution found for vllm-hust-ext==0.2.0.dev0`。回执见 `docs/release.md` §11.8。
- **tag/Release**：`v0.1.2`（annotated）；Release 附件 = 从 PyPI 下载的权威字节 + `dist.sha256`。
- **两条环境事实**：① 华为云镜像**滞后**（发布后 simple 页仍只有 0.1.0/0.1.1），判据必须打 pypi.org；
  ② 本容器 `github.com:443` 直连仍超时，Release 附件的浏览器下载路径**未实测**（回环走 API 资产端点）。

## 2026-09-26 续做：把"本机可直接做"的后续项清空（issue #2 §7）

来源：`knowledge/handoffs/HANDOFF-2026-09-26-issue2-open-items.md` §7 的分组清单。本轮做了
四件**纯本机、不占卡、不对外**的事；其余各项的默认动作是"不做/先在测试机做"，仍挂在那儿。

- **测试 extra 摘掉不可解析的钉**（`pyproject.toml`）：0.1.0/0.1.1 的 `test` extra 钉
  `vllm-hust-ext==0.2.0.dev0`，而该包**不在任何索引上** ⇒ 第三方
  `pip install "vllm-ascend-split-batch[test]"` 必然失败（实测原文见 `docs/release.md` §11.7）。
  已删该行 + 落注释说明原因，README「Extension framework」改成"装 extra 不需要它 / 跑
  `tests/test_manifest.py` 需要它（从 git 装）"。守卫
  `tests/test_manifest.py::test_test_extra_does_not_pin_an_unresolvable_extension_manager`
  防回钉。**发版才生效**（0.1.1 元数据不可覆盖）⇒ 未 bump 版本、未发布。
- **配置面边界机械化**（新 `tests/test_extension_config_boundary.py`，3 例）：核实到的两侧事实是
  上游 `configure` 只校验顶层是 object（无 per-bundle schema），而**本仓没有任何载体读
  `configuration`**（`src/**` 无 `.configuration`、无 `VLLM_HUST_EXT_CONFIG`，门控全走 env）。
  所以"配置 schema / 未知配置拒绝"在本仓是**边界声明**而非可加校验。测试固定三件事：
  ① 门控面只有 env（源码级扫描）；② 未知 `configuration` 键下 `extension check` 的输出**逐字
  相同**（对照实验）；③ configure→enable→disable→forget 只写元数据、未知键往返原样保留、
  **从不 import 实现载体**（`sys.modules` 审计）。服务级"卸载后重启仍正常"仍需占卡，未做。
- **发布检查单固化 `host_tree` 全量**（`docs/release.md` §5/§7）：CI 跑
  `-m "not host_tree"`，两个源码级漂移守卫在 CI 是**显式 deselect** ⇒ 发布前必须在本机跑一次
  全量。已把"本机跑全量 `pytest -q`（含 `host_tree`）"写成 §5「构建前」的固定步骤并给出判据；
  自托管 NPU runner job 仍属基础设施决策，未动。
- **W2 `fi_sampling` 源侧冻结快照**（新 `docs/evidence/fi-sampling-frozen-20260909/`）：任务是让
  "源侧 hash 可直接比对"。实测排除了打 tag：源包是活工作区（`api.py` 已漂移为 `4e3ce156…`），
  且工作区仓历史里**只有**漂移后那一版（`9e98559`）⇒ 快照只能由 §4.1 的逆变换重建。
  已产出 `snapshot/{api,kernels,npu_env,pure}.py`（4 件源侧字节，`sha256sum -c FROZEN.sha256`
  → 4 行 OK）+ 幂等的 `make_snapshot.py`（`--check` 校验；重建命中不了记录值就**拒绝写入**）。
  `PROVENANCE.md` §5 与 `ATTRIBUTION.md` 已改指快照；守卫
  `tests/test_provenance_hashes.py` 新增 2 例（入仓字节命中记录值、重建可复现）。
- **门槛**：`pytest -q` **455 passed**（449 → +6）、`ruff check .` / `ruff format --check .` 干净；
  **CI 身份复测**（`sitecustomize` 屏蔽 torch/torch_npu/vllm/vllm_ascend，父子进程同效）
  `pytest -q -m "not host_tree"` = **112 passed / 11 skipped / 14 deselected / 0 failed**。
  全程未用 NPU、未起服务、未占卡；未 push。
