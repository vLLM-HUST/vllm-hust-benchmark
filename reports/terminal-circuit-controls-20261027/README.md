# circuit-fibsqrt ARM64 环境控制

在原任务源码的 ARM64 派生镜像上，各运行一次新鲜 nop 与 oracle：nop 2/3、reward=0；oracle 3/3、reward=1。原三个测试、32000 门限制和 gcc
simulator 保持不变。

原预算为构建 600 秒、Agent 3600 秒、verifier 3600 秒；控制容器实际 1 CPU、2 GiB、cpuset 24-31、MTU 1400，三个原日志 RW 挂载与固定 uv
缓存 RO 挂载均核验。两个容器及控制器已释放，81 份证据封存通过。

这是环境及评分控制证据，不是模型成绩，不增加 Terminal 模型任务覆盖。独立新 Agent 需要另行冻结和准入。summary.json 与 whitelist.json 为 storage
公开投影原字节；不含 oracle 答案或任务源码正文。python verify.py 仅离线核验。
