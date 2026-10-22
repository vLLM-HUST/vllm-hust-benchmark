# W2 `fi_sampling` 源侧冻结快照（frozen-20260909）

本目录给出**源包那一版的真实字节**，让溯源复核不再需要自己推导逆变换：
`snapshot/` 里 4 个文件的 sha256 与 `PROVENANCE.md` §4.1 记录的**源侧**值逐位相等，
复核只需一条命令。

## 1. 为什么需要这个快照（2026-09-26 核实的两条事实）

| 事实 | 实测命令与输出 |
|---|---|
| W2 源包是**活工作区**，2026-09-09 后加入 W3 renorm/mask 家族 ⇒ 今天的源码 `diff` 不再是"搬运差异" | `sha256sum knowledge/surveys/sampling/fi_sampling/api.py` → `4e3ce156…`（记录值 `902884d6…`）；`pure.py` → `8a8bba61…`（记录值 `923d7db2…`，相差 91/112 行） |
| 工作区仓 git 历史里**没有** W2 那一版 ⇒ "打 tag 冻结"不可行 | `git log --format=%H -- knowledge/surveys/sampling/fi_sampling/api.py` 只返回 `9e98559`（2026-09-12 提交），该版本内容即漂移后的 `4e3ce156…` |

⇒ 唯一可用的重建路径是 `PROVENANCE.md` §4.1 已证实的方法：搬运只改了 `api.py` 的两条
导入行，其余字节与源侧相同；把 vendored 的相对导入还原成绝对导入即得源侧字节，
其 sha256 必须命中记录值 `902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c`
（该命中同时证明"记录为真"与"搬运没有夹带其它编辑"）。

## 2. 快照内容与记录值

快照 id：`frozen-20260909`（对应 `PROVENANCE.md` §4.1 的表；源包评审 2026-09-09）。

| 快照文件 | 源侧 sha256（= §4.1 记录值） | 对应 vendored 件 |
|---|---|---|
| `snapshot/api.py` | `902884d6cd64c78b8b917f6404f51305a1f23a37760b80a005466b770918a03c` | `src/vllm_ascend_split_batch/fi_sampling/api.py`（搬运改了 2 行导入） |
| `snapshot/kernels.py` | `8d0b20cca2559a2c4d893a96bc6c2398569eb2805258848f54ccde114b17d27e` | 同左（逐字节相同） |
| `snapshot/npu_env.py` | `04a3a122f66a74b7176cfaf7f3b1b701871bfe0e55e252fcf58f8a85b352a77a` | 同左（逐字节相同） |
| `snapshot/pure.py` | `923d7db26a704e778dfafb4bd74a2742364e6a5b8de250518bc08dfd6954ab08` | 同左（逐字节相同） |

`FROZEN.sha256` 是上表的 `sha256sum -c` 兼容清单。

## 3. 怎么复核

```bash
cd docs/evidence/fi-sampling-frozen-20260909
sha256sum -c FROZEN.sha256          # 期望 4 行 OK
python3 make_snapshot.py --check    # 期望 "snapshot OK: 4 files match PROVENANCE.md §4.1"
```

- `make_snapshot.py` 不写任何东西时用 `--check`；不带参数则**重建** `snapshot/` 与
  `FROZEN.sha256`（幂等，可用来验证"入仓字节确实是这套重建规则产出的"）。
- 重建若命中不了记录值，脚本**拒绝写入**并以非零退出 ⇒ 快照不会静默失真。

## 4. 边界（如实登记）

- 快照只含 `PROVENANCE.md` §4.1 **记录过 hash 的 4 个文件**。`reference.py`、`page.py`
  与 `__init__.py` 未搬运、也无记录值（`__init__.py` 是本仓新写），故不在快照内。
- 快照是**重建产物**，不是从 2026-09-09 的工作区磁盘直接拷贝的副本 —— 两者等价这一点
  由 §4.1 的 sha256 命中保证（重建值 = 记录值），不是因为"我们还能访问当时的工作区"。
- 本目录是 `docs/evidence/**`，已被 `pyproject.toml` 的 ruff `extend-exclude` 排除
  （存档件保持原件字节，重排会破坏 hash 清单）。
- 机械守卫：`tests/test_provenance_hashes.py`（vendored hash、逆变换复核、快照重建一致性）。
