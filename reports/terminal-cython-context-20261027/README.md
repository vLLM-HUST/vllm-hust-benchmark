# TerminalBench2.1：Cython 环境控制与原 Agent 结果

## 测试身份

原 build-cython-ext 任务、8 个任务文件、11 个测试及 oracle 保持不变，在现有 ARM64 派生环境执行。 依赖接口修复仅将 planarity 1.0.0
换为固定源码离线构建的 0.6，其余 24 个版本保留。 原 oracle 10/11、reward=0 不覆盖；新鲜环境 nop 为 1 pass / 10 fail、reward=0，oracle
为 11 pass、reward=1。 环境控制不增加模型覆盖，也不代表原 AMD64 镜像验收通过。

## 原 Agent 结果

指定 B0 为 vLLM 0.18.0 / vLLM-Ascend 0.18.0、Qwen3.5-35B-A3B。 原预算为 100 turn、32K
上下文、max_tokens=4096、seed=17、Agent 与 verifier 各 900 秒。 whole-trial retry=0、SDK retry=0，原嵌套重试上限
3+3；上下文异常不重试，summarization 关闭。

唯一 Agent 运行 478.118260 秒，29 次成功回复后第 30 个 HTTP 请求被上下文上限拒绝； Harbor episode 计数为 30，成功 Agent step 为
29，二者不能混用。 该结果为 ContextLengthExceededError，不是 100 turn 或 900 秒耗尽。 verifier 未启动，reward、CTRF 均为
null，不能写为功能 0/1。 客户端 exit 0 不代表任务通过；资源已排空、释放并封存。

累计 Terminal 已尝试 3/89 项，86 项未执行；旧两题及其恢复记录保留。 原 artifacts 清单为空，未采集完整源文件、native 扩展与安装态现场快照。 已核验 144 个上游
Git blob，按原轨迹重放 11 条编辑，9 个受跟踪文件的差异 SHA 精确匹配原采集值。 这只证明源码可复用，不等于原 native 二进制、安装态或完整现场快照。

## 派生重建评分

保留上述原 null，另以144个已核原始Git blob建立明确标识的本地基线仓库，再覆盖同一最终源码；
本地commit不是原441c807对象。仅执行9条必要的原安装、构建及复制命令，不修答案，不重复Agent或控制。
9条命令均退出0，构建96.468339秒；原verifier运行9.089041秒，11项完整执行、10通过1失败、缺失0，reward=0。
唯一失败test_ccomplexity实际报告NumPy无int属性，固定NumPy2.3.0下原答案ccomplexity.pyx:16仍引用np.int。
原600秒构建、900秒评分及1CPU/2GiB资源不变。三阶段144源码SHA与模式一致，native前后相同；进程退出0，资源已释放，34文件封存通过。
新native与安装产物均是派生重建，不冒充原现场快照；该评分不增加第4次模型尝试，也不覆盖原context失败/null。

## 证据与复算

- [新环境原控制](controls-summary.json)；[原测试名称及状态](controls-test-statuses.json)。
- [原 Agent 结果与独立复核](model-summary.json)，包含 33 个 CPU 与 17 个服务封存文件的索引摘要。
- [确定性源码复用边界](source-boundary.json)，后续构建必须单列为派生重建，不得修答案或重跑 Agent。
- [派生重建评分](recovery-summary.json)；[11项原测试状态](recovery-tests.json)；[原与恢复回执投影](recovery-receipt-projections.json)。
- 两份 whitelist 保留公开摘要 SHA；私有命令、轨迹、答案及构建产物正文不公开。

在仓库根目录运行：

```sh
python reports/terminal-cython-context-20261027/verify.py
```
