# r14同轮区间核算

本目录核算基于已归档的`../owned-fia-r13-r14/archive.json`，不新增性能成绩。

- `r14-hardware-intervals.json`：分别计算首请求至末响应、末前向事件外界和所有导出同步的HBM及HCCS区间。
- `r14-request-attention.json`：设备长度发布与所在前向调用关联后，分列窗口内及边界/尾部逻辑注意力算量。
- `sources.tar.gz`：原样保存核算源码、测试、模型配置及通信计数器资格记录；各核算JSON包含输入SHA256。

仓库测试`tests/test_midterm_r14_intervals.py`从原始归档复算，保留整数计数精确比较， 浮点HBM比例积分仅允许跨Python版本求和的8 ULP差异。
设备事件映射采用2毫秒资格测试容差，不是硬件时钟误差上界证明。 完整F、匹配参考和M01达成率尚未形成结论。
