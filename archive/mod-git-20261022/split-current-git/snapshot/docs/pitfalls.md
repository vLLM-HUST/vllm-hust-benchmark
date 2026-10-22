# 踩坑史

按"症状 → 原因 → 正确姿势"组织。新坑请追加到对应小节,注明发现时间。

## 1. 环境类

### 1.1 CWD 遮蔽 vllm 包(⚠️ 唯一高频陷阱)

- **症状**:CWD=`/vllm-workspace` 时 `import vllm` 异常、`vllm serve` 报怪错。
- **原因**:`/vllm-workspace/vllm/`(源码 checkout,无 `__init__.py`)遮蔽真实包。
  该目录已于 2026-09-10 删除,但**规则保留**——工作区顶层仍不应作为 python CWD
  (任何未来的 checkout 都会重新引入同一遮蔽)。
- **姿势**:任何 python/vllm 命令**严禁以 `/vllm-workspace` 为 CWD**;到
  `/tmp`、各仓库根或子目录执行都安全(`vllm-ascend-hust` 带连字符不会遮蔽 `vllm_ascend`)。

### 1.2 triton-ascend 安装覆盖坑

- **症状**:`pip install triton-ascend` 连带拉原生 triton 3.5.0,覆盖 patched 包;
  后续任何依赖 `triton` 的 pip 安装也会重新覆盖。出现 triton import 报错先怀疑这个。
- **姿势**(重装流程,禁 `--no-deps` 之外的方式):

```bash
pip uninstall -y triton triton_ascend
pip install triton-ascend==3.2.1 --no-deps --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
pip install pytest-xdist
```

> (2026-09-10 注:hust 新基线为 **triton-ascend 3.2.2**,恢复命令的版本号以工作区
> AGENTS.md 陷阱 #2 为准,勿按上文 3.2.1 覆盖安装。)

### 1.3 默认 cudagraph 模式 FULL_AND_PIECEWISE 启动即崩(2026-09-10,新基线)

- **症状**:本基线(torch 2.13 + vllm-ascend-hust main)默认
  `cudagraph_mode=FULL_AND_PIECEWISE` 时引擎初始化失败:
  `AssertionError: expected OutputCode, got GraphModuleImpl`
  (`fusion_pass_compile`,torch 2.13 AOT-autograd 缓存打包与 torch_npu 不兼容)。
  与任何插件无关(env 不设、插件 inert 时同样崩,OFF 腿已复证)。
- **姿势**:显式 `--compilation-config '{"cudagraph_mode":"FULL"}'`
  (npugraph_ex + ACL graph,自带 AOT 缓存规避);所有 A/B 腿必须**对称施加**。
- **另注**:`VLLM_DISABLE_COMPILE_CACHE=1` 仍是必须项(共享 AOT 缓存命中硬崩,见证据③口径)。
- **2026-10-13（0.28 镜像）更新**:该镜像上默认捕获网格 / piecewise 档在 `capture_model` 阶段失败
  （`Resource_Error(EE1023) Alloc Stream resource failed: Too many streams are created`）。
  **正解不是退成 `cudagraph_mode=NONE`**——Ascend 上 `NONE ⇒ mode=NONE`（`vllm_ascend/platform.py:1195`）
  会把 inductor 一起关掉（L1 TPOT 250ms vs 正常 ~14ms，17x 退化）。可用配置是**收小捕获网格 + FULL 档**：
  `--compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32]}'`
  （实测 L1 TPOT 14.08ms / L2 3983 tok/s）。已固化进 `bench-ladder.sh` 的 serve 默认。

- **副作用须知**:`triton` 包元数据不存在(`pip show triton` 查不到、`pip check`
  常驻告警),vllm 的 HAS_TRITON 检测基于 import 不受影响。裸调
  `vllm_ascend.ops.triton` kernel 前需先调 `init_device_properties_triton()`。

> **订正(2026-09-16,追加不改上文)**:本条的**根因已定位到行并给出正解**。
> - 根因:`AscendCompiler.compile` 的 `enable_npugraph_ex=False` 分支
>   (`fusion_pass_compile` → `compile_fx` → `aot_autograd`,compiler_interface.py:80/:97)
>   返回普通 `GraphModule`,而 torch 2.13 的 AOTAutograd 缓存写入要求 `OutputCode`。
>   宿主已有 guard(`compiler_interface.py:40`)但**只**在 npugraph_ex 分支被进入。
> - 上文"姿势"(`cudagraph_mode=FULL` 走 npugraph_ex)是**绕过**,且**不适用于验收冻结配置**
>   ——V3.8 表附-2 冻结 `cudagraph mode=piecewise`,正好落在崩的那个分支;`VLLM_DISABLE_COMPILE_CACHE=1`
>   也被表附-4 末行("其他执行变量｜禁止")排除。
> - **正解**:插件侧 carrier `aot_cache_guard_plugin.py`(本仓 `src/vllm_ascend_split_batch/`,
>   默认开、`VLLM_HUST_AOT_CACHE_GUARD=0` 关),只对未加 guard 的分支进入宿主自身 guard;
>   宿主将来自修则该 carrier 变 no-op(漂移守卫 `tests/test_aot_cache_guard_drift.py`)。
> - 证据:`bench/runs/20260916-abcert-b1/FINDINGS.md` §阻塞 1(真机:断言计数 0、
>   `Compiling a graph (1,32768) 13.38s` 成功);跨仓条目已在工作区 `AGENTS.md` 陷阱 9 登记。

### 1.4 模型卷只读

`/data/shared_models`(与 `/data/shared_datasets` 同卷)是只读挂载;拉新模型放
容器内可写目录。当前主力模型:**Qwen3.5-35B-A3B**（0.28 镜像基准，hybrid **TP2**，
40 层:30 GDN + 10 MoE）。历史主力 Qwen2.5-Coder-14B-Instruct（单卡,48 层/hidden 5120/bf16）
随 Qwen2.5/0.25 域**退役为历史专项**（见 [cohort-and-ownership.md](cohort-and-ownership.md) §2）。

### 1.5 `Unknown vLLM environment variable` 警告**不是**载体缺席判据(2026-10-17)

- **症状**:调 `opt bench serve` / 跑窗口后看日志,见到
  `WARNING [envs.py:2296] Unknown vLLM environment variable detected: VLLM_HUST_GATE_FP32`
  与 `... VLLM_HUST_GDN_MERGE_INPROJ`,于是判定"husk 载体没装、`VLLM_HUST_*` 全是空开关"。
- **原因**:该警告只说明这两个键**不在 vllm 自己的 env 白名单**里(vllm 的 `envs.py` 不认插件自定义键),
  与插件装没装无关 —— **载体生效时同样会打这两行**
  (`包体恢复/logs/ladder-serve-1017-081645.log:18-19`,而同一份日志 `:8,174,177` 有全套功能 marker)。
  按它判缺席会把"载体已恢复"误读成"载体缺席"(2026-10-17 早间实际发生过一次)。
- **姿势**(判据三件,缺一件只能说 unknown):
  1. **服务实际 Python** 的 `importlib.util.find_spec("vllm_ascend_split_batch")` 非 None。
     解释器必须与 serve 一致(本机 `/usr/local/python3.12.13/bin/python`);
     `/usr/bin/python3`(3.10)看不到这个包,会假报缺席。
  2. `vllm.general_plugins` 里能枚举到 husk 入口点(本机 10 个,含 `gate-fp32` / `gdn-merge-inproj`)。
  3. **本 serve 日志**里功能 marker 命中:`cascade plugin loaded`(每进程一条)、
     `Gate fp32 cache populated: 40 MoE gates`(TP0/TP1)、`W3F2: GDN in_proj merge applied on 30 layer(s)`(TP0/TP1);
     `gdn_merge_inproj_plugin.py:195` 的 fail-closed 警告是"插件不重复合并",**不是失败**。
- **机器可读**:`tools/bench-ladder.sh carriers [serve_log]` 输出上面三件事的单行 JSON;
  plain screen 会把它嵌进 `screen.json` 的 `carriers`(`opt-ops-carriers.v2`),
  serve 就绪后也会自动打人读摘要。其中 `Unknown vLLM env 警告` 单独记在
  `unreliable_absence_signals` 且 `is_absence_signal=false`。
- **纪律**:载体状态是**记录轴不是门禁** —— 只记录、不阻断 screen/serve、不要求重跑;
  只有跨窗口绝对数值比较要求同载体状态,两臂载体状态不一致才是第二个变量。

#### 1.5.1 插件开关只能从 "serve 侧" 读,不能读调用 shell(2026-10-17 修)

- **症状**:`screen.json.carriers.env` 里 `VLLM_HUST_GATE_FP32`/`VLLM_HUST_GDN_MERGE_INPROJ`
  是空的,但同一份 carriers 的 marker 显示 `Gate fp32 cache populated` / `W3F2 ... applied` —— 自证冲突。
- **原因**:`carriers.env` 原来读的是**调用 screen 的那个 shell** 的 `os.environ`,而
  `export VLLM_HUST_*=1` 只发生在 `cmd_serve` 的进程里; screen 由 `opt bench screen` 另起 shell 调用,
  未必继承。同理 `cohort_write` 曾经排在两个 export **之前**,会记成"未导出"。
- **姿势**:`cmd_serve` 在起服前把 export 后的值原子写到与 serve 日志同 stem 的
  `<serve log>.env.json`(`opt-ops-serve-env.v1`: `source`/`scope`/`serve_pid`/`keys`/`keys_set`/`keys_unset`),
  探针按 `serve_log` 关联读取并记 `env_scope`/`env_source`; 读不到或与日志不匹配 → 值一律 `null`
  + `env_scope=env_source=unknown` + `probe_errors`( **不用空串/`0` 冒充 off** )。
  两个 `export VLLM_HUST_*` 必须排在 `cohort_write` 之前。

#### 1.5.2 fail-closed 不等于"插件没生效",也不等于"运行树补丁没 applied"(2026-10-17 修)

- **症状**:看到 `gdn_merge_inproj_plugin.py:195 ... the GDN merge carrier is fail-closed`,
  同时日志又有 `W3F2: GDN in_proj merge applied on 30 layer(s)`,于是把合并记成"插件做的",
  或反过来判"插件没生效"。
- **原因**:两件事分属两个载体层。仓内 `gdn-merge-inproj` 插件**没有可安全消费合并权重的公开 seam**
  (`AscendGatedDeltaNetAttention.forward` 内联算 `in_proj_qkvz + in_proj_ba`),
  所以它 **fail-closed 拒接合并、模型保持 stock**; 日志里的 `W3F2 ... applied` 是
  **运行树补丁**(`apply-runtime-patches.sh`) 干的。两者共存, 因果不可互换。
  细节见 [../HOST_CONTRACT.md](../HOST_CONTRACT.md) 的 "GDN in_proj merge carrier" 段。
- **姿势**:看 `carriers.attribution` 的**逐载体**字段, 不要看单一布尔:
  `husk.cascade` / `husk.gate_fp32`(`effective`)、`husk.gdn_merge_inproj`
  (`present-but-fail-closed`)、`runtime_tree.w3f2`(`applied`)、`runtime_tree.w3f3`
  (`enabled`/`not-enabled`/`failed`)。归因口径: **运行树 W3F2 的合并不得归因给仓内插件**;
  fail-closed 警告断言的是"没有重复合并",不是失败。

## 2. 宿主与集成类

### 2.1 monkeypatch 面是弱契约

- **症状**:宿主(vllm/vllm-ascend)升级后插件行为怪异或静默失效。
- **原因**:patch 的是宿主**私有方法**(签名无兼容承诺),契约只有
  `host.version_range` 单方面钉死。
- **姿势**:升级宿主前按 [release.md](release.md) 第 4 节清单逐项核对签名,
  差异收敛在 `cascade_runner_patch.py`;核对前不要翻 `host.version_range`。

### 2.2 microbatching 互斥

cascade 两段式与任何 microbatching(`use_ubatching`: DBO 或 `ubatch_size>1`)
互斥,插件侧镜像官方 core gate 强制回落。当前 vllm-ascend 平台层会重置
`enable_dbo` 与 `ubatch_size`,该 gate 在此宿主上是休眠的——但改宿主版本时
要重新确认(关联 2.1)。

### 2.3 kernel 侧危害外溢到插件测试

- 同进程内 `fa_fp32_stage1` 之后跑 kv≥8k 的 FIA v2 TND 块表形态会触发
  fftsplus aicore 0x800000——micro-bench 必须分组进程(gate 的
  `cascade_gate_self.py` 因此用子进程,别改回同进程)。
- FIA TND 计时类探针**必须逐请求传** `actual_seq_kvlen`（传累计值会越界读 →
  垃圾块号 → MTE DDR 越界 0x800000 或静默 NaN；2026-09-08 定案，非布局限制、
  非算子缺陷）。**任何 FIA 计时前先同数据对拍正确性,存活≠正确**。
- gate 探针的 block table 必须逐请求独立分配,shared/reused 块表会踩内存
  (历史 fix 9899f00)。
- 详见 kernel 仓库 README §4-§5。

### 2.4 宿主换图重放机制 ⇒ cascade 孪生卡死(2026-09-27 定位并修)

**症状**:目标栈(`vllm-ascend` `fbe4911bb`,自编 `0.25.1rc2.dev605+…`)上,图孪生开启时
**第一个 decode step 之后请求再也不出 token**(4 并发实测 1/4 完成、其余 >90 s 零吞吐;
引擎 CPU ~60% **自旋**,不是 I/O 等待)。`capture body SUCCESS` 照常 96 次、`cascade plugin
loaded (gate=1, graph_gate=1, kernel_wheel=ok)` 照常、`twin missing` 0 次、0 TypeError
⇒ **所有旧观测点都显示"正常",只有吞吐是 0**。关图孪生(`VLLM_ASCEND_ENABLE_CASCADE_GRAPH=0`)
则 4/4 正常(0.5–0.7 s) ⇒ 问题专属图重放路径。

**机理**:新宿主给 FULL 图换了重放机制 —— 新增 `vllm_ascend/compilation/updatable_graph.py`
的 `UpdatableGraph` 与 `use_updatable_graph`(`vllm_ascend/utils.py:1773`),`ACLGraphWrapper.__call__`
对 `FULL + use_updatable_graph(attn_backend)` 改走 `_updatable_graph_replay(...)`
(`acl_graph.py:297`),**不再**调 `update_full_graph_params`(该函数对可更新图直接
`return`);而插件的重参数化包装挂在 `impl_cls.update_graph_params` 上 ⇒ **一次都不会被调到**
(实测 `replay update` = 0;旧宿主上是 2)。孪生图里每个 stage 任务组前后都有
`ExternalEvent.wait`(捕获期写入),**只有**插件的重参数化会 `record` 这些事件 ⇒
没人 record ⇒ 图内等待永不满足 ⇒ 重放挂死。

**修法**(`cascade_graph_plugin._wrap_updatable_graph_replay`,特征探测 + fail-open):
包装 `ACLGraphWrapper._updatable_graph_replay`,在 **`orig(...)` 之前**调
`_update_cascade_on_updatable_replay`。**顺序是硬约束**:放在 `orig` 之后会挂
(实测:同一个 4 并发负载,更新放后面 ⇒ >90 s 超时;放前面 ⇒ 8/8 成功、0.2–0.4 s)。
宿主自己的 `enable_enpu` 分支也是"先 update 再 replay",插件与之对齐。

**判据(两代宿主都要跑)**:

| 宿主 | 期望 |
|---|---|
| 新机制(≥`fbe4911bb`) | trace 出现 `updatable replay: cascade update ran=True`,且 `cascade key hit` **= 0**(那是旧接缝的观测点,新机制下不再触发) |
| 旧机制(基线 `74f0c0a27`) | trace 出现 `updatable-replay seam absent on this host`,`cascade key hit` > 0,且 `cascade update ran` **= 0**(新接缝必须 no-op) |

**同源坑(诊断期踩到,值得记住)**:
- **`VLLM_ASCEND_CASCADE_MIN_PREFIX` 小于一个 block ⇒ 孪生捕获静默关闭**:
  `cascade_runner_patch.py:266-271` 里 `dummy_prefix = min(MIN_PREFIX, …)`,若 `< block_size`
  就 `return`,**无任何日志**;运行期 gate 仍判 on ⇒ 表现为"图开了但没接管"
  (`cascade twin missing … step replays the standard full-KV graph`)。冒烟想放宽运行期门槛时
  别把它设成 0,用 ≥ 一个 block(本机用 1024)。
- **挂死的引擎无视 SIGTERM**:上述死锁状态下 `kill <pid>` 无效(实测仍在、HBM 位不释放),
  必须 `kill -9`;杀完等 10 s+ 再复查 `npu-smi info -t usages -i <card>`。
- **单请求冒烟会给出假绿**:`VLLM_ASCEND_CASCADE_MIN_REQS` 默认 32(本机冒烟设 2),
  1 个请求时常 < 门槛 ⇒ cascade 未准入 ⇒ 走标准图 ⇒ **0.3 s 返回正常**,而 ≥2 并发才触发孪生。
  验证 cascade 图路径**必须**发 ≥ 门槛的并发请求。

### 2.5 正确性验收必须走离线固定批 harness；HTTP 并发路线会给出假发散(2026-09-27 实测)

**症状**:用 `vllm serve` + 64 个并发 HTTP 请求做 cascade ON/OFF 的 token 序列比对,得到
**21–23/64** 发散 —— 高于 C 口径判据(≤8/64)。看起来像"cascade 数值有问题",但**不是**。

**实测判据(同配置、不同进程各跑一轮)**:

| 对照(HTTP 路线,64 并发) | 发散 |
|---|---|
| ON vs OFF | 21–23/64 |
| ON 重复 | 1/64 |
| **OFF 重复** | **22/64** |

**OFF 基线自身跨进程就不一致** ⇒ 该路线的噪声底(22/64)本身远高于判据,数字不可用。

**机理**:并发请求的**到达顺序**在两次运行间不同 ⇒ 引擎批组成/调度不同 ⇒ bf16 级归约路径不同。
历史验收用的 `llm.generate` **固定批** harness 没有这个自由度(同一栈上离线复跑 OFF 两轮 = **0/64**,
ON 两轮 = 0/64,ON-vs-OFF = **6/64**)。

**纪律**:
- **C 口径(逐 token 发散计数)验收只能用离线固定批 harness**(`docs/evidence/cascade/ev2_run.py`
  + `ev2_compare.py`,每腿独立 `VLLM_CACHE_ROOT` 与 `VLLM_DISABLE_COMPILE_CACHE=1`)。
- HTTP 路线只适合**统计量**(服务可用性、TTFT/TPOT 分位),**其正确性数字不得用于判据**。
- 判据必须带**参考帧对照**:同一栈上 `OFF-eager vs OFF-graph` 给出"纯图机制(与 cascade 无关)的
  发散量级" —— 本机目标栈实测 **7/64**(与历史同值),而 cascade 的 ON-vs-OFF 是 **6/64**
  ⇒ cascade 的发散**不高于**图机制自身的噪声底。

## 3. 打包与发布类

### 3.1 manifest `activation.environment` 填文档字符串(2026-09 已修)

- **症状**:无(处于 `import_only` 时值不生效——这也是它潜伏的原因)。
- **风险**:该字段语义是 **enable 时注入的值**;填 `"0 (default) | 1"` 这类
  说明文字,翻 `active` 后会被原样注入环境变量,直接破坏 default-off。
- **姿势**:只填真实值(enable=注入 `"1"`);`tests/test_manifest.py` 有守护
  测试,值域必须是 `"0"`/`"1"`。

### 3.2 版本双写漂移

版本在 `pyproject.toml` 与 manifest `extension_version` 两处,手改漏一处会导致
`extension inspect` 版本错乱。已有测试守护(`test_extension_version_matches_
distribution_version`),发版两处同步改。

### 3.3 平台 wheel 的假 tag

携带 `.so` 的 wheel 若打成 `py3-none-any`,用户可在任何平台安装、import 才炸。
kernel wheel 由 `NpuExtension` 自动产出 `cp312-cp312-linux_aarch64` 正确 tag;
若将来手工打包,发布前必须核对 tag(见 kernel README §4)。

### 3.4 vllm-hust-ext fail-closed 常见原因

`run` 拒绝启动时依次排查:宿主版本/`api_range` 不匹配;手工设置了 manager
拥有的 `VLLM_EXTENSION_MANIFESTS`/`VLLM_EXTENSION_BUNDLES`(这两个变量归
manager,手工设置即拒绝);已 enable 的扩展被卸载(卸载前先 `disable`/`forget`)。
完整清单见 bidkv 指南 §13(vllm-hust-docs 仓库 `operations/bidkv-packaging-
and-release-guide.md`)。

### 3.5 cascade 图捕获在默认 capture_sizes 下 NPU OOM(2026-09-11)

**症状**:`vllm-hust-ext run`(cascade 已 enable)用默认
`cudagraph_capture_sizes` 起服,cascade twin capture 反复
`Tried to allocate 418.00 MiB ... 224.63 MiB free` NPU-OOM →
`cascade graph capture failed; capturing the standard graph instead`
fail-open 刷屏(143+ 次),引擎永远到不了 ready。

**修复**:启用 capture-size 封顶(§3 验证过的 harness 口径):

```bash
--compilation-config '{"cudagraph_capture_sizes":[32,64,128]}'
```

**定性**:config 面问题,非正确性缺陷——封顶后 144 次 capture(3 桶 × 48 层)
全成功、0 fail-open。§3 启用验证日志:
`knowledge/evidence/cascade/logs/v1-active-enablement/serve-active-enablement.log`。

### 3.6 triton-ascend kernel 多指针参数在 ttir_to_linalg 段 segfault(2026-10-18)

**症状**:`gdn_qkv_norm_kernel` 尝试把三段输出拆成多块独立显存时,两种
kernel 签名都在编译期直接段错误(进程退出码 139,无 Python 栈):
①`unpack_norm_kernel(X, Y, K=None, V=None)`(可选指针参数 + 分支选目标);
②独立 `unpack_norm_separate_kernel(X, Q, K, V)`(4 指针 + `dst = Q if
segment == 0 else K`)。栈顶均在
`triton/backends/ascend/compiler.py:247 ttir_to_linalg`。

**对照**:原 2 指针签名 `unpack_norm_kernel(X, Y)` 加 `multibuffer=True`
编译与图捕获均正常;两种 segfault 形态与 multibuffer 无关(未启用时同样崩)。

**规避**:需要多输出布局时不要给 kernel 加指针参数;保持单输出指针 +
`multibuffer=True`,由 host 侧复制/切 view(本仓 `launch_separate` 即此
形态,仅作诊断工具,默认路径不用)。根因未查(bisheng/MLIR 下游),升级
triton-ascend 后需重验。

**证据**:`/root/包体恢复/records/probes/20261018T164700Z_gdn_separate/
separate-graph.log`、`separate-graph-r2.log`(两份 fault handler 栈);
`multibuffer-graph.json`(2 指针 + multibuffer 正常)。
