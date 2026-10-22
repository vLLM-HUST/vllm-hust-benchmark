# Design note: RoPE 变体三缺陷（fork 缺陷的插件侧 carrier + 漂移守卫）

- 状态：**新能力，`import_only`**（三证据未齐，见 `docs/release.md` §3 与 `docs/progress.md`）
- 开关：`VLLM_HUST_ROPE_FIX=1`（默认 0：不 import `vllm_ascend`/`torch`/`torch_npu`、
  不 patch、不注入、日志零输出）
- 落点：`src/vllm_ascend_split_batch/rope_fix_plugin.py`（+ bundle marker
  `src/vllm_ascend_split_batch/rope_fix/`）
- 入口：`vllm.general_plugins` → `vllm_ascend_split_batch.rope_fix_plugin:load`
  （bundle `org.vllm-hust.rope-fix`，`vllm_hust.extension_bundles` → `...rope_fix`）
- 缺陷来源与数值证据（本仓外，只读）：
  `knowledge/surveys/contrast/norm-rope-act/对照-rope.md`（修前断点）、
  `.../对照-rope.md`（fork 分支 `/tmp/vah-ropefix` 的修法与修后数字）、
  `.../patches_ropefix_2026-09-10/combined-main-to-fix.diff`（参考 diff）。
- 本仓证据：`docs/evidence/rope-fix/`（CPU 守卫 + NPU 探针 carrier 形态复现）。

## 0. 结论速览

宿主基线 `vllm-ascend-hust main @ 74f0c0a27` 上，RoPE 变体族有**三处真实缺陷**
（不是"未实现"，是接错/漏参/缺省分歧），此前只能以"临时 clone
`/tmp/vah-ropefix` + `PYTHONPATH` 覆盖"形态存在。本 carrier 把三处修复做成
**插件侧一等公民**（不动宿主源码、不建 venv、不改环境），并附一个**反向漂移守卫**：

| # | 缺陷 | 宿主锚点（基线行号） | 插件侧修法 |
|---|---|---|---|
| ① | llama3-scaling 未接线：`REGISTERED_ASCEND_OPS` 无 `Llama3RotaryEmbedding` 键 → 落到 `CustomOp.forward_oot` 默认实现（`forward_native`，bf16 逐元素数学），triton rope 不可达 | `vllm_ascend/utils.py:688-817`（dict 在 `:735-765`） | 新增子类 `AscendFixLlama3RotaryEmbedding(Llama3RotaryEmbedding)`，把 `op_registry_oot["Llama3RotaryEmbedding"]` 指向它 |
| ② | mrope 少传参：`AscendMRotaryEmbedding.forward_triton` 调 `triton_mrope` 只给 8 个位置参（host 签名 9 个，末位 `is_neox_style` 无缺省）→ `positions.ndim==2 ∧ mrope_interleaved`（Qwen2.5-VL 形态）必抛 `TypeError` | `vllm_ascend/ops/rotary_embedding.py:513-552`（调用在 `:541-550`） | 子类 `AscendFixMRotaryEmbedding` 镜像 `forward_triton` 并补 `self.is_neox_style`，把 `op_registry_oot["MRotaryEmbedding"]` 指向它 |
| ③ | yarn `truncate` 缺省分歧：fork 缺省 `False`，vLLM/HF 缺省 `True` → Qwen2.5 系 yarn 配置（`rope_scaling` 无 `truncate` 键）cache 与 host/GPU 不等价（19.0% 元素不同） | `vllm_ascend/ops/rotary_embedding.py:281-298`（缺省在 `:297`） | 子类 `AscendFixYaRNRotaryEmbedding` 只改缺省为 `True`（显式传参仍生效），把 `op_registry_oot["YaRNScalingRotaryEmbedding"]` 指向它 |

## 1. 宿主 seam：为什么是"覆写 `op_registry_oot` 条目"这唯一杠杆

（对 basline 逐条读码；`vllm/model_executor/custom_op.py`）

1. `vllm_ascend.utils.register_ascend_customop()`（:688）构造全局
   `REGISTERED_ASCEND_OPS`（:735-765，310P 分支再 `update` 一批），末尾
   `for name, op_cls in REGISTERED_ASCEND_OPS.items(): CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)`
   （:813-814），由 `_ASCEND_CUSTOMOP_IS_REIGISTERED`（:695）保证只跑一次。
2. `CustomOp.register_oot`（:339-351）会
   `assert reg_name not in op_registry_oot, f"Duplicate op name: {reg_name}"`
   ⇒ **插件不能自己注册同名 key**（会直接把引擎 init 打断）。
3. `CustomOp.__new__`（:109-128，`PluggableLayer.__new__` :47-66 同构）在
   **实例化时**读 `op_registry_oot[op_name]` ⇒ **替换该 dict 的条目**既"晚于注册
   断言"又"早于模型加载"，是插件侧唯一可行且有效的杠杆。rope 类走的是
   `PluggableLayer`/`CustomOp` 混合 MRO，但两者共用同一张表，故 ① 的键可以直接创建
   （host 不写这个名字 → 不触发断言）。
4. `Llama3RotaryEmbedding` 的构造签名有 4 个额外位置参
   （`scaling_factor/low_freq_factor/high_freq_factor/orig_max_position`），
   而 `CustomOp.__new__` 返回后 Python 会用**原参数列表**调用
   `type(instance).__init__` ⇒ 插件类必须**逐参复刻** 10 参签名
   （不能直接把 `"Llama3RotaryEmbedding"` 映射到 `AscendRotaryEmbedding`：会
   `TypeError`）。另需 `self.use_mtp`（`AscendRotaryEmbedding.forward_oot` 的
   draft-model 分支会读）与 `_record_cos_sin_cache`（fork 的 cache 记账）。

## 2. 时序：为什么必须"包裹 `register_ascend_customop`"而不是直接 patch

插件 `load()`（`vllm.general_plugins`）在 vllm import 期执行，**早于**引擎 init 的
`register_ascend_customop`：

- 若在 `load()` 里直接覆写 `op_registry_oot["MRotaryEmbedding"]`，宿主随后的
  `register_oot` 会撞上 `Duplicate op name` 断言 → **引擎启动失败**；
- 若什么都不做（等注册后再改），插件没有任何"注册完成"的回调点。

因此 carrier 的做法是：

1. 把 `vllm_ascend.utils.register_ascend_customop` **包一层**：先调用原函数，
   **返回之后**再执行覆写；包装体自身 try/except fail-open（覆写失败只 warn
   一次，注册调用照常返回）。
2. 包装体同时安装到**所有已经持有该函数直接引用的模块**上——
   `vllm_ascend/worker/worker.py` 是 `from vllm_ascend.utils import
   register_ascend_customop`（:95 绑定、:150 调用）的**真实调用点**，只 patch
   `vllm_ascend.utils` 属性会漏掉它；同时在 `vllm_ascend.utils` 属性上留一份，
   供之后才 import 的模块（`from ... import`）取到包装体。
3. 覆盖键的**幂等**：`op_registry_oot[key] is our_class` 时不动；重复调用
   `register_ascend_customop`（宿主早退）也不会二次副作用、不会重复告警。
4. **注册之前一个键都不写**：`CustomOp.register_oot` 的
   `assert reg_name not in op_registry_oot` 意味着**预置任何键都可能打断引擎 init**
   （而且上游若哪天自己把 ① 修好，预置键会与之撞车）。因此三键一律在"注册返回之后"
   施加：②③ 在注册前缺席属正常（不是漂移），静默跳过、不告警；`install()` 只在
   检测到注册**已经**跑过（`_host_registration_ran()`）时才立即施加一次，否则等
   包装体回调。

## 3. 三处覆写的实现取舍

| # | 形态 | 为什么这样取 |
|---|---|---|
| ① | 逐参复刻 10 参 `__init__` + `forward_oot` 委托 **`op_registry_oot["RotaryEmbedding"]`（缺失时退回模块级 `AscendRotaryEmbedding`）** | `CustomOp.__new__` 用宿主原参数实例化，签名必须一致；委托目标**跟随本 build 实际注册的基础 rope 类**（310P 档是 `AscendRotaryEmbedding310`，见 §6.3），而不是硬编码 910 类。委托在 `__init__`（模型加载期，注册之后）解析并缓存在实例上，避免每 forward 查表、保持 ACL graph 可捕获。仅 forward 改路：cache 生成端（`_compute_inv_freq` 三段式重映射）本身已在 fp32 内完成且单次舍入（探针 `recompute_max_abs=0`） |
| ② | **镜像** `forward_triton` 方法体 + 补第 9 参；kernel 从 **fork 模块自身的 `triton_mrope`** 绑定 | fork 在方法**中间**硬编码了 8 参调用，`triton_mrope` 是模块全局、调用点没有任何 hook 可注入第 9 参；子类重述方法体是"零宿主改动"下最小可行载体。为避免"镜像悄悄过期"，`tests/test_rope_fix_drift.py` 做 **AST 等价断言**（镜像体去掉注入实参后必须与宿主方法体逐节点相等）。kernel 绑定取自调用点**同一个** `fork_rope.triton_mrope`（不再另走 `vllm...mrope` import 路径）；该绑定为 `None`（无 triton）时**不安装 ②** 并单条 warning——否则镜像体会在**模型 forward 期**调 `None` 硬失败，而不是 load 期 fail-open |
| ③ | `def __init__(self, *args, truncate: bool = True, **kwargs)` 签名无关转发 | 只改缺省值这一件事；`*args/**kwargs` 让宿主 `get_rope` 的调用形态变化（新增关键字）不会把插件类打成 `TypeError`，而显式 `truncate` 仍原样透传（逃生舱保留） |

②的镜像体**不带 docstring**（docstring 会成为额外的 AST 节点），方法体内的注释
不参与比较。

## 4. default-off 与 fail-open

- **default-off**：`load()` 先判 `is_enabled()`；未置 env 立即返回 `False`，
  `vllm_ascend`/`torch`/`torch_npu`/`triton` 一个都不 import，`op_registry_oot`
  不被触碰，日志零输出 ⇒ 与原生 bit 级一致（子进程 `sys.modules` 纯净性测试 +
  `stats()` 断言，见 `tests/test_rope_fix_plugin.py`）。
- **fail-open（逐键）**：
  - `register_ascend_customop` 不存在 → 单条 warning，返回 `False`，**什么都不做**；
  - 覆写流程任何异常 → 单条 warning，注册调用照常返回；
  - **半安装回滚**：包装体已装、随后（`install()` 里的即时施加）抛错时，包装体被
    回滚还原，`stats()["installed"]` 与 `stats()["wrapped"]` 反映真实状态——否则
    日志说"保持宿主原样"而包装体仍在，下一次注册会静默降档；
  - 目标键上不是"本 carrier 继承的那个 fork 类"（例如 310P 的
    `AscendMRotaryEmbedding310`，或上游已自行接线）→ **不覆写**，单条 warning
    说明"expected X, found Y"，行为保持宿主原样；
  - ②的 kernel 绑定缺失（无 triton）→ **不安装 ②**，单条 warning；
  - 覆写时机若晚于注册（例如注册已跑完才 `load()`）→ `install()` 内检测后立即施加
    一次；否则由包装体在注册返回后施加（注册前一律不写 registry，见 §2.4）。

## 5. 漂移守卫（`tests/test_rope_fix_drift.py`）

守卫分三层，失败信息**二分化**，便于一眼判断"变冗余"还是"锚点漂移"：

| 层 | 断言 | 运行前提 |
|---|---|---|
| **反向缺陷断言**（核心） | ①注册 dict **仍无** `Llama3RotaryEmbedding`；②`forward_triton` 的 `triton_mrope` 调用**仍是 8 个位置参且无 `is_neox_style` 关键字**；③`AscendYaRNRotaryEmbedding.__init__` 的 `truncate` 缺省**仍是 `False`** | 纯 `ast` + 源文件（`importlib.util.find_spec` + 本文件位置确定性定位宿主树，**不 import**），CPU 可跑 |
| **镜像等价断言** | carrier 的 `forward_triton`（去掉注入实参后）与宿主方法体 **AST 逐节点相等**；镜像调用点注入的实参必须是 `self.is_neox_style` | 同上（CPU） |
| **接口断言** | `op_registry_oot` 存在 + `CustomOp.__new__` 仍查它 + `register_oot` 仍拒绝重名；`Llama3RotaryEmbedding.__init__` 10 参签名；`AscendRotaryEmbedding.forward_oot` 签名；`AscendMRotaryEmbedding` MRO 仍设置 `mrope_interleaved/is_neox_style/head_size`（任务书的 `head_dim` 在本基线叫 `head_size`）；`worker.py` 仍直接 `from vllm_ascend.utils import register_ascend_customop` 且仍调用它；fork 报 `HAS_TRITON` 时 `fork_rope.triton_mrope` 必须存在且形参个数 9（末位 `is_neox_style`） | `pytest.importorskip`（需 vllm/vllm-ascend） |

两种失败措辞（`FIXED` / `DRIFT` 常量）：

- `upstream fixed defect ①/②/③ → drop the corresponding override`
  ——锚点还在、缺陷没了：**删掉对应覆写**（留着就是死代码/遮蔽上游修复）；
- `anchor drifted → re-audit HOST_CONTRACT`
  ——seam 被移位/改形/消失：**先重新审 HOST_CONTRACT 的载体小节**，再改 carrier。

镜像等价断言会**区分原因**：宿主调用点已补第 9 参时报 `upstream fixed defect ②`
（覆写已冗余），其余情况报 `anchor drifted`。

**缺证据即红**：宿主树定位失败（既无 editable 路径、也无 `VLLM_ASCEND_HUST_ROOT`、也
不在工作区同层）时模块级 **fail**（不是 skip）——skip 会让"守卫没跑"看起来像"守卫通过"。
CPU 门槛要求的就是这三层断言真的执行。`VLLM_ASCEND_HUST_ROOT` 指向宿主 checkout
可在任意布局下运行。

守卫本身做过**双向自测**（不触碰宿主树：把宿主两文件复制到 `/tmp`，注入"上游已修"
与"锚点漂移"两种变异，用 `VLLM_ASCEND_HUST_ROOT` 指向副本运行）：3 条反向断言 +
镜像断言按预期红，措辞分别为 `upstream fixed defect ①②③` 与 `anchor drifted`
（02 变异同时命中两条——这正是 §5 末尾"区分原因"的作用）。证据：
`docs/evidence/rope-fix/guard-selftest/`。

## 6. 已知偏差与未尽事项

1. **entry point 需要刷新安装元数据**：`pyproject.toml` 里的
   `vllm_hust.extension_bundles` / `vllm.general_plugins` 两行是声明；editable 安装的
   `entry_points.txt` 是安装期快照，新增条目要等下一次 `pip install -e .` 才会被
   `load_general_plugins()` / `vllm-hust-ext` 发现（本任务的纪律禁 pip 动作，
   故未刷新）。CPU 侧用 `test_descriptor_declares_both_entry_point_groups`
   钉住声明，NPU 侧探针用**直接调用 carrier `load()`** 的 runner 触发（见
   `docs/evidence/rope-fix/`）。
2. **未做真模型服务级 e2e A/B**：本机无 llama3 / Qwen2.5-yarn / Qwen2.5-VL
   checkpoint，价值仍以算子级/构造级证据闭环（与 fork 侧报告同口径）。
3. **310P（兼容档）与未来上游变体**：
   - **替换键**（`MRotaryEmbedding` / `YaRNScalingRotaryEmbedding`）：只替换"本 carrier
     继承的那个 fork 类"，兼容档的 `AscendMRotaryEmbedding310` 等实现一律跳过 + 单条
     warning；
   - **①的委托**：不硬编码 910 类，而是解析 `op_registry_oot["RotaryEmbedding"]`
     （即本 build 实际用于基础 rope 的实现）。310P 档该键被换成
     `AscendRotaryEmbedding310`，其 `forward_oot` 走 310P 自己的
     `npu_apply_rotary_pos_emb` 路径 ⇒ 插件把 llama3 接到**该档自己的**实现上，
     而不是把可用的原生路径换成 910 kernel（后者会 `NotImplementedError`/算错）。
     委托在 `__init__` 解析并缓存；键上不是类（异常注册表）时退回模块级
     `AscendRotaryEmbedding`，仍不改行为语义。
     —— 这条**不**依赖 `get_current_hardware_profile()`：跟随注册表比再写一份
     硬件判定更贴近"宿主自己选了什么"。
4. **②的镜像体**必须在宿主改动时同步（守卫会红）；这是"零宿主改动"的代价，
   有意接受。无 triton 的 build 上 ② 不安装（见 §3）——该档 fork 自身的
   `forward_triton` 也不可达。
