# FIA 配对采集与分支覆盖

对象：Qwen3.5-35B-A3B，BF16、TP2，学校运行栈，MMLU-Pro 200条输入、out256、无限到达。 本组为M01观察器诊断，不替换正式B1性能结果。

r15保留采集验收失败记录：每rank的732次context-parallel发布没有对应launch几何。 r16补齐两个launch分支，并采集设备侧长度、页表和opaque调度计划。
每rank的2,640组记录完整配对，所有使用页索引均处于所记录的KV cache范围内。 原始请求、日志、进程清理、源码绑定、独立块表/计划探针与r16 HBM数据库一并归档。
r15系统控制记录保留，但未重复包含其完整profiler输出。

`archive.json`记录分卷和每个归档成员的SHA256。按文件名顺序拼接 `evidence.tar.gz.part*`得到gzip tar。归档内路径按原绝对路径去除开头`/`保存，
仅用于追溯，不应直接以root身份恢复到原位置或执行运行控制器。

在仓库运行：

```bash
pytest -q tests/test_midterm_r16_capture.py
```

CPU测试核对分卷、成员哈希、三路配对、分支覆盖、页索引范围和同窗HBM/HCCS积分。 没有宣称模型计划已独立解码、attention数值正确或M01已达标。
新服务和独立运行目录与固定编译缓存种子分别披露，不将编译预热描述为完全冷启动。
