# 证据：rope-fix carrier（CPU 守卫 + NPU 探针 carrier 形态）

> 载体：`src/vllm_ascend_split_batch/rope_fix_plugin.py`（开关 `VLLM_HUST_ROPE_FIX=1`，默认关）。
> 设计说明：`docs/design/rope-variant-defects.md`；宿主 seam/锚点：`HOST_CONTRACT.md`「rope_fix component」。
> 判定：**CPU 层全绿 + NPU 探针四变体全 PASS（= fork-fix 参考值）+ default-off 对照腿逐字复现修前三断点**。
> ⚠️ 本包**不含**服务级真模型 A/B（无 llama3 / Qwen2.5-yarn / Qwen2.5-VL checkpoint，见 §5）。

## 1. 命令（可直接复跑）

CPU（`pytest`/`ruff` 在仓内跑，其余命令 `cd /tmp`）：

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
python -m pytest -q                 # 261 passed（本 carrier 新增 35 例：23+11+1）
ruff check .                        # All checks passed!

# 守卫双向自测（宿主树零改动：副本注入变异 + VLLM_ASCEND_HUST_ROOT 指向副本）
bash docs/evidence/rope-fix/guard-selftest/run.sh
```

NPU（卡 7，全部包 `flock -w 7200 /tmp/w3-npu.lock`；跑前卡 7 `HBM Usage Rate = 5%`）：

```bash
# 探针副本（保护只读知识库里的参考 logs，不被本次运行覆写）
cp /vllm-workspace/knowledge/surveys/contrast/norm-rope-act/rope_variants_e2e_probe.py \
   /vllm-workspace/knowledge/surveys/contrast/norm-rope-act/common.py /tmp/ropefix-probe-on/
cp -a /tmp/ropefix-probe-on /tmp/ropefix-probe-off

# ① carrier 腿（VLLM_HUST_ROPE_FIX=1；无 PYTHONPATH 覆盖）
flock -w 7200 /tmp/w3-npu.lock -c 'cd /tmp && VLLM_HUST_ROPE_FIX=1 \
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-on --part main'
flock -w 7200 /tmp/w3-npu.lock -c 'cd /tmp && VLLM_HUST_ROPE_FIX=1 \
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-on --part atb'

# ② default-off 对照腿（env 未置）
flock -w 7200 /tmp/w3-npu.lock -c 'cd /tmp && \
  ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-off --part main'

python3 docs/evidence/rope-fix/compare.py --check    # 四腿对比 + 判定，rc=0
```

`logs/run-probe-with-carrier.py` 是本次使用的 runner（探针副本 + carrier 装配的完整副本）：
它先按 vLLM 自己的方式 `vllm.plugins.load_general_plugins()`，再显式调用 carrier 的
`load()`，最后 `runpy` 执行探针的单个 `--part`。**为什么需要 runner**：探针是裸脚本、
从不调用 `load_general_plugins()`；而 `vllm.general_plugins` 的发现依赖安装期
`entry_points.txt` 快照，本次任务禁 pip 动作，故新增的 `rope-fix` 入口不在快照里
（runner 日志里如实打印 `rope-fix entry point discovered: False`）。carrier 的
**调用面**因此是"已安装 editable 包里的 carrier 模块"，而不是 PYTHONPATH 源码覆盖——
这正是本任务要验证的形态。

## 2. 结果：四腿对比（`comparison.md` 全文；此处摘判定）

| 变体 | 修前（prefix 参考） | 本次 default-off 对照腿 | 本次 carrier 腿 | fork-fix 参考 | 判定 |
|---|---|---|---|---|---|
| interleaved | 全 PASS | 同 prefix | 全 PASS | 全 PASS | 不变（回归） |
| llama3 | `kernel_reachable=断点`（`Llama3RotaryEmbedding` / `CustomOp.forward_oot`，O≡G3 bf16 数学、O≠直接 kernel） | **同 prefix** | **PASS**（`AscendFixLlama3RotaryEmbedding` / 接线到 triton，O≡直接 kernel） | PASS（`AscendLlama3RotaryEmbedding`） | ① 修复 ✅ |
| yarn | `PASS*`：对 host 公式 **5.10156**，公式分歧 **5.09375** | **同 prefix** | **PASS**：对 host 公式 **0.03125**，公式分歧 **0** | PASS（同值） | ③ 修复 ✅ |
| mrope | `kernel_reachable=断点`：`TypeError: triton_mrope() missing 1 required positional argument: 'is_neox_style'`（栈：`rotary_embedding.py:562 forward_oot` → `:541 forward_triton`） | **同 prefix（同一条 TypeError）** | **PASS**：包装正常返回、O≡G2 逐位 | PASS | ② 修复 ✅ |

- 四变体 e2e 总偏差：**3.125e-2**（= cache bf16 底噪，等于 fork-fix 参考值；
  修前 yarn 是 5.10 = 公式分歧主导）。
- `oot_rope_keys`：对照腿 `[…,'MRotaryEmbedding','RotaryEmbedding','YaRNScalingRotaryEmbedding']`
  → carrier 腿多出 `'Llama3RotaryEmbedding'`（= ① 的接线证据）。
- carrier 运行期 state（runner 打印）：`installed=True`，`applied={Llama3RotaryEmbedding: created,
  MRotaryEmbedding: replaced AscendMRotaryEmbedding, YaRNScalingRotaryEmbedding: replaced
  AscendYaRNRotaryEmbedding}`，`skipped={}`。
- `atb` 腿（`npu_mrope` 备选，独立进程隔离）：`seq3/vision` 两 combo `vs G2 bitwise=True`、
  哨兵 OK（`logs/carrier-atb.json`）——与 fork-fix 参考一致，②的修复未影响 ATB 备选路径。
- 命名差异（如实记录，非行为差异）：插件子类叫 `AscendFixLlama3/YaRN/MRotaryEmbedding`，
  fork 分支叫 `AscendLlama3RotaryEmbedding` / 原名；`compare.py` 对 `cls` 字段只比较
  "是否 Ascend 系 + 是否接线到 triton"，行为字段逐值相等（`--check` rc=0）。

## 3. 守卫双向自测（`guard-selftest/`）

| 变异 | 期望措辞 | 实测 |
|---|---|---|
| 上游已修（注册 dict 加 `Llama3RotaryEmbedding`、`triton_mrope` 补第 9 参、`truncate` 缺省改 `True`） | `upstream fixed defect ①/②/③ → drop the corresponding override` | 5 条红且**逐条**是 `upstream fixed defect ①②③②`（② 同时命中反向断言与镜像断言；镜像断言已能识别"宿主已补第 9 参"） |
| 锚点漂移（`REGISTERED_ASCEND_OPS` 改名、`AscendMRotaryEmbedding` 改名、`truncate` 形参删除） | `anchor drifted → re-audit HOST_CONTRACT` | 4 条全部按此措辞红 |
| 宿主树不可发现（副本 + `sitecustomize` 屏蔽 `find_spec`，无 `VLLM_ASCEND_HUST_ROOT`） | **fail 而非 skip**（缺证据即红） | 模块级 fail：`vllm-ascend source tree not found; ...` + `1 error during collection`（**无 skip 行**） |

原始输出：`selftest-run.log`（摘要）、`selftest-fixed.txt`、`selftest-drifted.txt`（pytest 全量）。

## 4. CPU 测试清单（新增 35 例：23 carrier + 11 守卫 + 1 manifest）

`tests/test_rope_fix_plugin.py`（23 例，mock 为主 + 1 例真宿主集成）：

- default-off（子进程 `sys.modules` 纯净性：不 import `vllm`/`vllm_ascend`/`torch`/`torch_npu`/`triton`）；
- `is_enabled()` 只认精确 `"1"`；
- **时序（安装早于注册）**：安装后 registry 保持全空——注册**之前**一个键都不写
  （否则会撞 `CustomOp.register_oot` 的 `Duplicate op name` 断言），注册返回后三键齐覆写；
- **时序（注册早于加载，= worker 子进程真实次序）**：先 `register_ascend_customop` 再
  `load()` → 三键被**立即**覆写、`stats()["installed"] is True`；
- 幂等：重复 `load()` / 重复注册 / 清空 once-guard 后再注册，registry 对象与告警条数都不变；
- 包装体重绑所有直接引用点（模拟 `worker.py` 的 `from ... import` 绑定）；
- fail-open：seam 缺失 / 类构造失败 / 覆写流程抛错 → 单条 warning、`load()` 返回 False、
  registry 原样、注册调用照常返回；**半安装回滚**：包装已装后覆写抛错 → 包装体还原、
  `stats()["installed"] is False`、后续注册不再降档；
- 对外来条目（310P 式变体）不覆写、单条 warning、`stats()['skipped']` 说明原因；
- 三个覆写类的行为：① 签名 = 宿主 11 参（self+10）、`use_mtp`/cache 记账/`forward_oot` 委托
  **到注册表里的基础 rope 类**（910 档 → `AscendRotaryEmbedding`；310P 档 →
  `AscendRotaryEmbedding310`；键上非类 → 模块级回退）；
  ③ 缺省 `truncate=True` 且显式 `False` 逃生舱保留；② 镜像体把 `self.is_neox_style` 作为第 9 参传给 kernel；
- **缺 `triton_mrope` 绑定（无 triton）**：② 不安装、单条 warning，①③ 照常；
- 真宿主集成：`vllm_ascend.utils.register_ascend_customop` 真调用后三键指向插件类；
- bundle/manifest 断言 + `pyproject.toml` 两组入口点声明（含 `org.vllm-hust.rope-fix`）。

`tests/test_rope_fix_drift.py`（11 例，见设计说明 §5：3 条反向缺陷 + 1 条镜像等价 +
7 条接口/源级，含 `worker.py` 直接引用断言；宿主树找不到时**模块级 fail 而非 skip**）；
`tests/test_manifest.py` 追加 1 例（rope-fix bundle），并把 `extension_version` 一致性
扩到三个 bundle。

## 5. 残留风险 / 未决点

1. **未做服务级真模型 A/B**：本机无 llama3 / Qwen2.5-yarn / Qwen2.5-VL checkpoint，
   价值止于算子/构造级（与 fork 侧报告同口径）；上游记录同此限制。
2. **entry point 元数据未刷新**：`pyproject.toml` 已声明，但要等下一次
   `pip install -e .`（本任务纪律禁 pip 动作）才会被 `load_general_plugins()` /
   `vllm-hust-ext` 发现；届时 `vllm-hust-ext extension inspect org.vllm-hust.rope-fix`
   可作 §3 启用验证的入口。
3. **②的镜像体是"零宿主改动"的代价**：宿主改 `forward_triton` 方法体即触发
   `anchor drifted`，须同步镜像（有意接受，守卫已覆盖）。
4. **310P / 未来上游变体**：两个替换键只认"本 carrier 继承的那个 fork 类"（跳过 + 单条
   warning）；①的委托改为跟随 `op_registry_oot["RotaryEmbedding"]`（不硬编码 910 类）——
   **310P 本机未实测**（本机是 910B2），该结论由构造级 mock 单测覆盖（310P 式注册表
   → 委托到 310P 类），真机行为待 310P 验证域进入时升级复核。
5. **性能未测**：本包只做接线与数值正确性；① 的收益（bf16 数学 → triton fp32 路径）
   需真 llama3 模型在服务级标定。
