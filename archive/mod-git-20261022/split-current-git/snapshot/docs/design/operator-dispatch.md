# 算子调度(选择)机制 — 设计与骨架

> 状态:**骨架 / 预留接口**。代码 `src/vllm_ascend_split_batch/op_selector.py`,
> 测试 `tests/test_op_selector.py`。**未接线任何现网路径**,不注册任何真实算子,
> 不 import torch。本文只定接口与扩展点,落地实现(热点表、真 bench、宿主接线)
> 是后续任务。

## 0. 范围与不变量

| 项 | 取值 |
|---|---|
| 落点 | 插件库 `vllm-ascend-split-batch-hust`(容器内 `src/vllm_ascend_split_batch/`) |
| 依赖 | 仅标准库;无 torch / vllm / triton |
| 开关 | `VLLM_ASCEND_OP_DISPATCH`(缺省 = off),`OpSelector.enabled()` 暴露 |
| 失败语义 | fail-open:选不出 → 返回宿主原生实现名,绝不猜 kernel |
| 审计 | 每次 `select` 落一条 `OpDecision` 到有界 ring buffer |
| 现网影响 | 零。无 entry point、无 monkeypatch、无 import 副作用 |

## 1. 背景与需求

模型落地(Qwen2.5-14B-Instruct)做完热点分析后,会得到一批"**同一算子存在多个
实现**"的点:宿主原生原语、自研 CCE/triton kernel、以及若干融合组合。哪个最快
取决于 shape 与场景(batch/token 数、prefix 共享长度、headdim、dtype、是否
greedy 等),而且这个最优解**只在实测过的格点上可信**。

本任务不实现真正的选择,而是把**接口与扩展点固定下来**,作为"热点驱动算子选择"
的前置件:后续新增一个 kernel 时,只需注册一个 `OpCandidate` + 一条静态/实测表
条目,不必改动选择器本体,也不必改动调用方。

### 1.1 两条内部先例(被收敛的对象)

| 先例 | 文件 | 选择方式 | 决策来源 |
|---|---|---|---|
| `fi_sampling` 分流 | `fi_sampling_route.py` | **规则式**:按 `batch_size / top_k / top_p / logprobs_mode` 立即判定 route | 静态规则(阈值来自 W2 bench,写死在常量里) |
| `cascade` 门 | `cascade_gate.py` | **度量式**:启动时 bench 一格一格测,`decision_for(shared_len, num_tokens)` 查表 | env override > 实测表 > 默认 on |

两者是同一件事的两个极端:一个是"规则先行",一个是"实测先行"。本设计把
二者的决策阶梯**抽象为一个 `OpSelector`**,把各自的特化字段沉淀成 `OpContext`
的公共载荷(见 §3.1)。**收敛是路径,不是立刻重构**(见 §4)。

### 1.2 flashinfer 参照语义

`/tmp/flashinfer-ref` 的 wrapper 是**两阶段**:`plan(info)` 在 run 之前把
"这次问题规格"换算成调度参数(分块、split-kv plan、workspace、padding),`run(...)`
只消费 plan,不再做决策。本设计对应其 **NPU 版前置规划**那一半:在真正下发 kernel
之前,把"用哪个实现"这件事**先算出来、先记录、可审计**;设备侧执行面(plan/run 的
run 半段)不在本次范围内,也不照抄 CUDA 面的 workspace/stream 语义。

## 2. 决策来源三档 + fail-open

选择器本体是一台**固定优先级的阶梯**,从高到低:

| 优先级 | source 常量 | 来源 | 典型生产者 | 对齐先例 |
|---|---|---|---|---|
| 1 | `SOURCE_ENV` = `"env"` | env 覆盖(`VLLM_ASCEND_OP_SELECT_<OP>` 或全局 `VLLM_ASCEND_OP_SELECT`) | 运维/AB 摆动手动强制 | `cascade_gate.override()` |
| 2 | `SOURCE_OVERRIDE` = `"override"` | 进程内 :meth:`OpSelector.override` | 单测 / A-B 脚本 | —(新增,便于测试) |
| 3 | `SOURCE_BENCH` = `"bench"` | 运行时实测表 :meth:`set_bench_table` | 启动 bench(未来) | `cascade_gate.decision_for()` 的实测表 |
| 4 | `SOURCE_STATIC` = `"static"` | 声明式 shape/scene 表 :meth:`set_static_table` | 人肉/论文写死的规则 | `fi_sampling_route.decide_route()` |
| 5 | `SOURCE_COST` = `"cost_hint"` | 候选自身的 `cost_hint`,取最小者 | 粗粒度启发式 | —(兜底启发式) |
| 6 | `SOURCE_FALLBACK` = `"fallback"` | **fail-open**:宿主原生实现名(`default`) | 什么都不知道时 | `cascade_gate` 未 bench 格点默认 on |

**为什么 env 高于进程内 override**:对齐 `cascade_gate` —— 部署必须能在不改代码的
前提下强制某一格点。进程内 override 只服务测试/A-B。

**override 是权威的**:若 env/override 指定了名字而该候选**未注册或
`is_applicable` 为假**,不继续往下查表,直接 fail-open 到宿主(记
`override_unknown:*` / `override_inapplicable:*`)。理由:静默落到**另一个**
kernel 会把配置错误藏起来;要么按你说的跑,要么回宿主。

**表条目失效会穿透**:bench/static 表里指向已注销/不适用候选的条目被跳过,继续
走下一档(`stale` 穿透),因为表是派生数据、可能部分过期。这一点与 override 的处理
刻意不同,已用测试钉住。

## 3. 接口规划

### 3.1 `OpContext` —— 载荷设计(字段来自既有先例,不臆造)

```python
@dataclass(frozen=True)
class OpContext:
    num_tokens: int = 0     # 本步扁平 token 数;= cascade 的 batch bucket、fi 的 batch_size
    shared_len: int = 0     # 共享/prefix 长度;= cascade 实测表的 prefix bucket
    headdim: int = 0        # kernel 形状判别量
    dtype: str = ""         # "bf16" / "fp16" / "fp32" ...
    scene: str = ""         # 自由分类标签:如 "greedy_k1" / "untruncated" / "joint_truncated"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "OpContext": ...
    def as_dict(self) -> dict[str, Any]: ...
    def key(self) -> tuple[Any, ...]:    # 默认表键: (scene, num_tokens, shared_len, headdim, dtype)
```

- `num_tokens` / `shared_len` 直接继承 `cascade_gate.decision_for(shared_len, num_tokens)`
  的两个键,`num_tokens` 同时充当 `fi_sampling_route` 的 `batch_size`;
- `scene` 承担 `fi_sampling_route` 里那一串布尔/枚举规则(`has_top_k`、`top_p`、
  `k1`、`logprobs_mode` 命中的路线)的**分类面**——调用方先把它折叠成一个稳定标签,
  选择器只认标签,不认 op 私有字段;
- `headdim` / `dtype` 是 kernel 形状面最常用的两个判别量。
- op 私有载荷(`has_top_k` 之类)在骨架里是**约定在 `scene` 上**;要加公共字段就
  扩 dataclass(唯一受支持的生长方式,签名变更需同步本文与测试)。
- `select` 同时接受 `OpContext` 或普通 `Mapping`(后者经 `from_mapping`,未知键忽略)。

### 3.2 `OpCandidate` —— 一个可替换实现

```python
@dataclass(frozen=True)
class OpCandidate:
    name: str
    is_applicable: Callable[[OpContext], bool] = _always_applicable  # 纯谓词
    cost_hint: float | None = None    # 越小越优;None = 不作主张(仅能靠表/override 胜出)
    label: str = ""                   # 人读说明

    def applicable(self, ctx: OpContext) -> bool:  # 谓词抛异常 → False(fail-open)
```

### 3.3 `OpDecision` + 审计

```python
@dataclass(frozen=True)
class OpDecision:
    op: str
    chosen: str  # 选中的实现名
    source: str  # env/override/bench/static/cost_hint/fallback
    reason: str  # 机读标签: override / bench_table / static_table / min_cost_hint /
    #            override_unknown:<n> / override_inapplicable:<n> /
    #            no_applicable_candidate
    context: OpContext  # 当时的载荷快照
```

`OpSelector` 内部 `deque(maxlen=audit_capacity)`(默认 `AUDIT_CAPACITY = 256`),
`audit()` / `last_decision()` / `clear_audit()` 只读访问。**每次选择有据可查**:
事后可回答"这一步为什么走了 kernel X"。

### 3.4 `OpSelector` —— 签名清单

```python
class OpSelector:
    def __init__(
        self,
        op: str,
        default: str = DEFAULT_HOST,
        *,
        key_fn: Callable[[OpContext], Hashable] | None = None,
        audit_capacity: int = AUDIT_CAPACITY,
        enabled_env: str = ENV_ENABLE,
        global_env: str = ENV_OVERRIDE,
    ) -> None: ...

    # 元信息 / 门控
    op: str  # property
    default: str  # property,fail-open 落点

    def enabled(self) -> bool: ...  # 读 VLLM_ASCEND_OP_DISPATCH == "1",默认 off
    def override_env_var(self) -> str: ...  # 本 op 的 env 覆盖名

    # 注册表
    def register(
        self, candidate: OpCandidate, *, replace: bool = False
    ) -> OpCandidate: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> OpCandidate | None: ...
    def names(self) -> tuple[str, ...]: ...

    # 决策数据(表 / 覆盖)
    def set_static_table(self, table: Mapping[Hashable, str]) -> None: ...
    def static_table(self) -> dict[Hashable, str]: ...
    def clear_static_table(self) -> None: ...
    def set_bench_table(self, table: Mapping[Hashable, str]) -> None: ...
    def bench_table(self) -> dict[Hashable, str]: ...
    def clear_bench_table(self) -> None: ...
    def override(self, name: str) -> None: ...
    def clear_override(self) -> None: ...
    def active_override(self) -> tuple[str | None, str]: ...  # (name, source)

    # 选择
    def select(self, ctx) -> str: ...  # 选名(记审计)
    def explain(self, ctx) -> OpDecision: ...  # 全记录(记审计)

    # 审计
    def audit(self) -> tuple[OpDecision, ...]: ...
    def last_decision(self) -> OpDecision | None: ...
    def audit_capacity(self) -> int: ...
    def clear_audit(self) -> None: ...
    def reset_for_tests(self) -> None: ...


# 进程级注册表(未来接线的扩展点)
def get_selector(op, default=DEFAULT_HOST, **kwargs) -> OpSelector: ...  # 每 op 单例
def registered_selectors() -> tuple[str, ...]: ...
def reset_registry_for_tests() -> None: ...
```

## 4. 与现网三机制的关系(收敛路径)

```text
                    ┌────────────────────────────────────────────┐
   fi_sampling_route│ 规则式:has_top_k/top_p/logprobs_mode/batch │  静态规则
   (已上线)         │ → decide_route() → ROUTE_*                │
                    └───────────────┬────────────────────────────┘
                                    │  规则细胞 → (scene 标签 + 静态表条目)
                                    ▼
   cascade_gate     ┌────────────────────────────────────────────┐
   (已上线)         │ 度量式:启动 bench → decision_for(shared, T)│  实测表 + env override
                    └───────────────┬────────────────────────────┘
                                    │  benched cell → (key=(T,shared) → 候选名)
                                    ▼
   未来"热点表"     ┌────────────────────────────────────────────┐
   (未做)           │  profile 三表 → 热点 op → 候选注册          │
                    └───────────────┬────────────────────────────┘
                                    ▼
                    ╔════════════════════════════════════════════╗
                    ║  OpSelector: env > override > bench > static ║  ← 本设计
                    ║             > cost_hint > fail-open          ║
                    ╚════════════════════════════════════════════╝
```

收敛**不是**要求把 `fi_sampling_route` / `cascade_gate` 立刻改写成 `OpSelector`
消费者:两者都在线上、都有实测证据、都以 default-off 保证零差异。**路径**是:

1. 新写的热点 op 一律用 `OpSelector`(插件侧新文件,不动既有);
2. 旧两机制若后续要合并,只需把各自的"判据"折成 `scene` 标签 + 表条目,
   `decide_route` / `decision_for` 退化为一个把 `OpContext` 喂进 `select()` 的薄壳;
3. 在那之前,**旧机制一行不改**。

三个机制共享的语义因此被 `OpSelector` 一次性固定:**default-off 门控在调用方、
fail-open 落宿主、决策可审计、override 权威**。

## 5. 热点驱动接入流程草案

```text
(1) 模型落地            Qwen2.5-14B-Instruct 在 910B2 上跑通,固定 profile 口径
        │
(2) profile 三表        算子级耗时 / 调用次数 / 占比  →  hotspots 排序
        │
(3) 热点 op 候选        对每个热点 op,列出可替换实现(宿主原语 / 自研 / 融合组合)
        │               并为每个实现写 is_applicable(ctx) 与 cost_hint
        │
(4) 注册 + 静态表        OpSelector(op).register(OpCandidate(...))
        │                set_static_table({...})  ← 来自论文/规则的先验
        │
(5) 1% 门槛验收         单算子延迟 + 融合前后对比 + e2e TPOT 三件套
        │                  (AGENTS.md 硬性要求;只报单算子加速比 = 无效)
        │                过不了 1% → 候选不入表,保持 fail-open
        │
(6) 实测表 / env 覆盖   过门槛的格点写 bench_table 或静态表;保留 env 强制开关
        │
(7) 翻默认              default-off 冒烟(env 未设 → 零差异)通过后,才由接线方
                        打开 VLLM_ASCEND_OP_DISPATCH
```

每一步都产出可审计物:profile 表、候选谓词、表条目、验收报告。`OpDecision` 的
`source/reason` 让"这次为什么没走优化路径"在线上可直接回答。

## 6. 扩展点清单

| # | 扩展点 | 用途 | 现有落点 |
|---|---|---|---|
| E1 | `OpCandidate.is_applicable(ctx)` | 声明一个实现的适用域 | `register()` |
| E2 | `OpCandidate.cost_hint` | 粗粒度排序的先验 | `register()` |
| E3 | `set_static_table` | 人肉/规则表 | 选择器实例 |
| E4 | `set_bench_table` | 运行时实测表(bench 产物) | 选择器实例 |
| E5 | `override` / env 覆盖 | 强制某格点(A-B / 排障) | `override()` / `VLLM_ASCEND_OP_SELECT*` |
| E6 | `key_fn` | 自定义表键(如只按 `(num_tokens, shared_len)`) | 构造参数 |
| E7 | `audit()` / `last_decision()` | 决策可观测 | 选择器实例 |
| E8 | `get_selector(op)` | 每 op 进程级单例,未来接线入口 | 模块级 |
| E9 | `enabled()` | 主门闸,default-off | env `VLLM_ASCEND_OP_DISPATCH` |

## 7. 与 flashinfer `plan`/`run` 的对照

| flashinfer | 本设计 | 说明 |
|---|---|---|
| wrapper `plan(info)` | `OpSelector.select(ctx)` | 下发前的前置规划:由问题规格得到调度决策 |
| wrapper `run(...)` | (不在范围内) | 设备侧执行面;本次只留选择接口 |
| plan 缓存到 wrapper 实例 | ring buffer 审计 + 表 | 我们额外要求**可审计**(flashinfer 不要求) |
| backend autotune | bench 表 + env 覆盖 | 三档来源对齐 `cascade_gate`,不引入 autotune 运行时 |

**不照抄**的点:CUDA 面的 workspace/stream/plan 缓存/backend 枚举;NPU 侧选择是
**离线可定、在线只查表**的模型(与 `cascade_gate` 一致),而不是在线 autotune。

## 8. 未做什么(明确边界)

- **不实现任何真实调度**:没有任何真实算子被注册;`op_selector.py` 是空壳 + 语义。
- **不接线现网**:无 entry point、无 `load()`、无 monkeypatch;`select()` 目前
  无人调用;`get_selector` 注册表在无人调用时是惰性的。
- **不改既有文件**:仅新增 3 个文件 + `docs/README.md` 一行索引。未触碰
  `fi_sampling_route.py` / `cascade_gate.py` / manifest / `pyproject.toml`。
- **不做 bench**:§5 第 (6) 步的实测表生产者(真跑 kernel 的 bench)是后续任务;
  本骨架只提供 `set_bench_table` 这个注入口。
- **不 import torch / 不碰设备**:测试全 CPU。
- **不翻 manifest**:能力未接线,`import_only` 语义不受影响。
