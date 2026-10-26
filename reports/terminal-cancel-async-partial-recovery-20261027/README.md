# TerminalBench2.1部分产物恢复评分

## 结果

TerminalBench2.1已尝试2/89题，87题未执行。OpenSSL完整答案恢复评分6/6通过、reward=1；cancel-async-tasks在提交和评分前因ContextLengthExceededError终止，原reward=null。未提交的原部分代码单独恢复评分，1/6通过、reward=0，不计第二次Agent或完整提交评分；不构成全量或B0/B1配对验收。

## 配置与方法

测试对象为Qwen3.5-35B-A3B，vLLM 0.18.0 / Ascend 0.18.0、Harbor 0.24.0
Terminus2及ARM64派生镜像。原Agent耗时332.567032秒，18次HTTP请求含17次成功和1次上下文拒绝；成功请求输入223132、输出13450
token。被拒请求至少28673输入token加4096输出预算超过32768上下文；原100turn、Agent及Verifier各900秒、seed=17及重试策略不变。1103字节run.py按原命令及终端回显确定性重建，非现场文件快照；原SHA及模式未知，0644为恢复选择，未修补接口。仅Verifier恢复耗时13.483108秒，文件存在测试通过、5项功能测试失败。两项并发测试原trace直接显示缺少run_tasks导入；三项取消测试显示启动任务数为零，其共同导入原因仅为源码推断。原env缺失警告保留。资源1CPU、2GiB、MTU1400，原测试及只读固定依赖传输不变，答案与传输前后SHA一致。新增Agent、模型调用、控制实验均为0，全部资源已释放。

## 证据

summary.json为规范聚合；tests.json及storage-public/ctrf-name-status.json为6项原测试状态。两个profiling-review及independent-review.json为独立复核，storage-public保留白名单投影与原件SHA索引。原题目、答案、run.py正文、请求标识及轨迹不公开；原失败、null与受限原件均保留。

运行 `python3 verify.py` 可离线复核公开聚合、逐项状态及哈希；原始日志复核仍需受控原件，不等同重新执行实验。
