# Prefill Microflow FA配对证据

## 测试对象与配置

Qwen3.5-35B-A3B、BF16、TP2、EP关闭、E24/R36、ctx64k、APC、MTP2、FULL图及async scheduling。 每卡20 GiB状态预算、8 GiB
host缓存。输入为既有MMLU-Pro 200请求，无限到达率、out256；不改变提示词或增加复用锚点。
Microflow仅覆盖target全注意力层的分片调度，GDN及draft保持原路径；KV为逐请求重算，不包含分段复用。

实测Microflow源码为`f7ab240381a0e10a4a13f4ee04598cb822db6fdd`，KV管理适配源码为
`947a610b012ce680bec226c4ba8538e75de027f6`。各归档内包含实际导入路径、源码摘要、Manager检查、启动命令及原始测量文件。

## 方法与结果

正式筛选组为OFF r4/r5/r6与ON r9/r10/r11，逐组配对、独立启动、顺序执行。 各轮使用同一份既有B1 Triton/Inductor缓存快照的独立副本；vLLM
AOT缓存保持独立初始化。 六轮启动和请求窗口均未记录外部NPU上下文，每轮200请求成功、0失败，全部退出后确认四卡资源释放。
Manager在控制器主动SIGINT结束服务后返回130；请求客户端均返回0，drain及资源释放记录独立保留。

[summary.json](summary.json)保存六个原始窗口、各指标中位数及范围、配对比值和动态计算的增量。
[repetition-plan.json](repetition-plan.json)保存三组统计口径。未选择最高轮次，未进行显著性推断；任务正确率未重新评测。
每轮的`mmlu-pro.json`保存逐请求输出、SSE计时和MTP统计；`microflow-*.jsonl`保存双rank组件一致性、target捕获和图重放回执。
`kvmat-policy.jsonl`与`kvmat-engine.jsonl`保存200次请求控制及对应盐摘要；设备、退出和文件摘要随各轮归档保留。

## 诊断资料

- `on-r4`：空编译缓存，Manager未激活KV。仅为FA执行诊断，不属于当前B1组合。
- `on-r7`：空编译缓存，KV已激活；旧日志误将MTP启用标志当作draft身份，不能用于target重放资格判定。
- `off-r3`：空编译缓存对照。其低吞吐与空缓存ON接近，不据此宣称Microflow负优化。

上述诊断归档均不混入三组配对统计。更早的启动准入失败未产生性能结果。 编译产物和模型权重不进入仓库；缓存初始状态以清单和SHA256记录。

## 复现核验

在单独目录解开任一正式归档，运行：

```sh
tar -xf verification-tools.tar.gz
python verify_school_run.py <解包根目录> <run_id> --output verification.json
```

验证器核对输入与源码摘要、配置、200请求及token总数、KV策略/引擎对应、双rank控制动作、窗口边界和资源释放。
仓库测试`tests/test_microflow_screen_evidence.py`独立读取六份归档并复算吞吐、配对几何平均与统计中位数。
