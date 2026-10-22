# 编程规范(agent 与新人通用)

本文是本仓库的规范性约束。目的:任何后来者(人或 agent)在这里做出的
改动都符合插件纪律,不破坏 default-off,不污染宿主。

## 1. 硬约束(违反即返工)

1. **宿主只读**:`/vllm-workspace/vllm-hust` 与 `/vllm-workspace/vllm-ascend-hust`
   禁止修改。宿主能力只走官方公开面:`vllm.general_plugins` entry point、
   `--worker-cls` 子类、`--additional-config`、官方 CLI/env。
   (旧基线 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend` 已于 2026-09-10 删除。)
2. **不注册 `vllm.platform_plugins`**(已被 vllm-ascend 的 AscendPlatform 占用)。
3. **default-off**:任何新能力关闭时必须与原生行为零差异。新开关一律
   env-gated、默认缺省,patch 方法未开启时逐字委托原实现。
4. **manifest 纪律**:bundle 保持 `import_only`,直到三项证据齐备
   (default-off 零回归冒烟、正确性对齐、性能对比);`host.version_range`
   钉住已验证区间,禁止写 `>=0`。
5. **provenance/ 只读**:提取历史代码必须保留原版权头(Huawei, Apache-2.0)
   与 commit 溯源。
6. **不做上游 PR(2026-09-11 定)**:改动只在本地落地(本仓 + 本地 clone/PYTHONPATH
   覆盖),不向上游(vllm / vllm-ascend 社区)提补丁,也不做"推动宿主提供 typed
   contract"这类上游化动作。与上游的关系只有一条:**跟随 `vllm-hust` release
   节奏做版本同步**(重钉锚点与 `host.version_range`,见 release.md §4)。
   `docs/upstream/` 素材冻结留档,不再维护、不提交。

## 2. 开发工作流

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust   # 不要在 /vllm-workspace 根目录跑 python(见 pitfalls)
pytest -q                                          # CPU 层门槛:全绿
ruff check .                                       # lint 门槛:零告警
# 涉及行为/门控/patch 面的改动,追加 NPU 冒烟:单卡、小 batch、default-off 对比
```

NPU 层验证顺序:先小规模冒烟(单卡 `npu:0`),再场景化;正确性(token 对齐/
容差声明)与性能(TPOT/latency 对比)证据同时留存,写进 `docs/evidence/` 或
`docs/progress.md`(**不写 PR 描述**——本仓不做上游 PR,见 §1 第 6 条)。

## 3. 代码规范

- Python ≥3.10 目标;ruff `line-length = 88`,rules: `B, E, F, I, SIM, UP`
  (见 `pyproject.toml`,提交前 `ruff check .` 必须零输出)。
- 纯逻辑与设备路径分离:shape 推导、gate 判定、planner 等写成 CPU 可测的
  纯函数;设备路径用 mock 测 + 真机冒烟。
- patch/包装代码集中放 `cascade_runner_patch.py` 等既有落点,不要散落新文件;
  新 monkeypatch 点必须同步更新 `HOST_CONTRACT.md` 与 `docs/release.md` 第 4 节。
- 不引入新三方依赖;确需引入时先在 PR 里说明理由并更新 `pyproject.toml`。

## 4. 测试规范

- 每个源文件有对应 `tests/test_*.py`;manifest 类不变量(版本一致、
  activation.environment 可注入、bundle 不可激活)有守护测试,改 manifest
  必须让对应测试表达新语义。
- 测试默认跑在 CPU(env 未装 torch_npu 也能全绿);需要设备的测试显式
  skip,不在 CPU 门槛里假装通过。
- 对照测试(与参考/基线实现)写明数值容差,纳入可自动跑的范围。

## 5. Git 与网络

- commit 原子:一个逻辑变更一个 commit,不混格式化与功能修改。
- message 用 conventional 风格(仓库现状):`feat(cascade): ...`、
  `fix(cascade gate): ...`、`docs: ...`、`test: ...`。
- clone/push 统一走 **SSH over 443**(一次配置永久生效):流程见工作区
  `/vllm-workspace/docs/git-remote-and-network.md`;gh-proxy 前缀仅留给 pip git 依赖。

## 6. 文档更新义务

改了什么就要同步它的权威文档(分工表见 [README.md](README.md)):
行为/开关 → 根 README;patch 面 → HOST_CONTRACT.md + release.md;
新坑 → pitfalls.md;里程碑/计划 → progress.md。agent 在收尾时自查一遍。
