# build-cython-ext ARM64 环境控制

## 测试对象与方法

TerminalBench2.1 原任务 `build-cython-ext`，任务源码 revision `7131e4375048a0e408a8fb404b5f499d726b695b`。使用原
Python 3.13 基镜像的固定 ARM64 manifest 构建派生任务镜像，仅修改 Dockerfile 的 FROM 引用和 task.toml 的镜像引用，保留原指令、oracle、11
项测试及 direct-pip 流程。 不声称与原 AMD64 镜像等价。

每个控制使用独立鲜容器，1 CPU、2 GiB、CPU 24–31、MTU 1400； 构建预算 600 秒，Agent 与 Verifier 各 900 秒，零自动重试。 先执行
nop，释放后执行原 oracle。保留原 direct-pip 安装流程，不挂载其他任务的 uv 缓存； 实际可写评分日志挂载以封存记录为据。

## 结果

| 阶段           | 进程退出码 | 测试通过/总数 | Reward | 判定           |
| -------------- | ---------- | ------------- | ------ | -------------- |
| ARM64 镜像构建 | 0          | 不适用        | 不适用 | 构建成功       |
| nop            | 0          | 1/11          | 0      | 符合负控制预期 |
| oracle         | 0          | 10/11         | 0      | 未通过正控制   |

nop 仅预装 NumPy 版本检查通过。oracle 的 10 项外层测试通过， 唯一失败为 `test_pyknotid_repository_tests`；其内部 pytest 返回 1，
记录为 17 项通过、1 项失败。原输出没有保留完整内部 traceback， 不据截断的异常摘要确定根因或完整内部测试名。

本轮新增模型调用和模型任务尝试均为 0，不改变既有 Terminal 模型覆盖分母。 镜像可构建不等于环境控制通过；本轮不能作为该任务的模型评分或全量验收。
原失败不替换、不择优重跑。两控制容器及执行进程均已释放。

## 证据与复核

- `summary.json`：配置、逐测试状态、完整分母及 66 份受限原件的 SHA256 索引。
- `independent-review.json`：独立复核聚合，包含实际依赖版本与结论边界。
- `derive_private_controls.py.source`：从受控 CPU 原件生成摘要的源码，不含题目或 oracle 正文；运行它需要受控原件。
- `verify.py`：仅依赖标准库的公开离线一致性与哈希核验。

```bash
python3 reports/terminal-build-cython-controls-20261027/verify.py
```

公开材料不包含任务指令、oracle 正文、模型问答或凭据。 离线核验不等于重新构建镜像或重新执行控制。
