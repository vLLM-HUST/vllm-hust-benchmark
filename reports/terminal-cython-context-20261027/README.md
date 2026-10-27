# TerminalBench2.1：Cython 环境控制与原 Agent 结果

## 测试身份

原 build-cython-ext 任务、8 个任务文件、11 个测试及 oracle 保持不变，在现有 ARM64 派生环境执行。
依赖接口修复仅将 planarity 1.0.0 换为固定源码离线构建的 0.6，其余 24 个版本保留。
原 oracle 10/11、reward=0 不覆盖；新鲜环境 nop 为 1 pass / 10 fail、reward=0，oracle 为 11 pass、reward=1。
环境控制不增加模型覆盖，也不代表原 AMD64 镜像验收通过。

## 原 Agent 结果

指定 B0 为 vLLM 0.18.0 / vLLM-Ascend 0.18.0、Qwen3.5-35B-A3B。
原预算为 100 turn、32K 上下文、max_tokens=4096、seed=17、Agent 与 verifier 各 900 秒。
whole-trial retry=0、SDK retry=0，原嵌套重试上限 3+3；上下文异常不重试，summarization 关闭。

唯一 Agent 运行 478.118260 秒，29 次成功回复后第 30 个 HTTP 请求被上下文上限拒绝；
Harbor episode 计数为 30，成功 Agent step 为 29，二者不能混用。
该结果为 ContextLengthExceededError，不是 100 turn 或 900 秒耗尽。
verifier 未启动，reward、CTRF 均为 null，不能写为功能 0/1。
客户端 exit 0 不代表任务通过；资源已排空、释放并封存。

累计 Terminal 已尝试 3/89 项，86 项未执行；旧两题及其恢复记录保留。
原 artifacts 清单为空，未采集完整源文件、native 扩展与安装态现场快照。
已核验 144 个上游 Git blob，按原轨迹重放 11 条编辑，9 个受跟踪文件的差异 SHA 精确匹配原采集值。
这只证明源码可复用，不等于原 native 二进制、安装态或完整现场快照；本目录不声明已经完成评分恢复。

## 证据与复算

- [新环境原控制](controls-summary.json)；[原测试名称及状态](controls-test-statuses.json)。
- [原 Agent 结果与独立复核](model-summary.json)，包含 33 个 CPU 与 17 个服务封存文件的索引摘要。
- [确定性源码复用边界](source-boundary.json)，后续构建必须单列为派生重建，不得修答案或重跑 Agent。
- 两份 whitelist 保留公开摘要 SHA；私有命令、轨迹、答案及构建产物正文不公开。

在仓库根目录运行：

    python reports/terminal-cython-context-20261027/verify.py
