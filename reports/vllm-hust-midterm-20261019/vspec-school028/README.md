# MTP local argmax配对证据

## 对象与配置

vLLM-HUST E24/R36 + KV逐请求重算 + Prefill Microflow FA组合，增加vSpec局部argmax候选。
Qwen3.5-35B-A3B、BF16、TP2、EP关闭、ctx64k、APC、原生MTP2、FULL图、async scheduling、每卡20 GiB状态预算及8 GiB
host缓存保持不变。 输入为同一MMLU-Pro 200请求、无限到达率、out256，不修改提示词或增加复用锚点。

实测vSpec源码为`d8cbfb50ee1df8e1664bd080a59e49ed2e47c104`。仅替换MTP草稿token选择，不启用其他vSpec组件。
Microflow源码为`f7ab240381a0e10a4a13f4ee04598cb822db6fdd`，KV管理适配为`947a610b012ce680bec226c4ba8538e75de027f6`。
归档包含独立安装内容、实际导入路径、源码摘要、Extension Manager准入记录及启动命令。

## 方法与结果

正式筛选组按OFF r1 / ON r4、OFF r2 / ON r5、OFF r3 / ON r6顺序执行，各轮独立重启。 Triton/Inductor缓存取同一既有快照的独立副本，vLLM
AOT缓存独立初始化。两卡计算，四卡保留，窗口之间验证资源释放。 各指标取三次运行的中位数并保留完整范围，三组吞吐比使用几何平均；不挑选最高轮次。

[summary.json](summary.json)保存原始窗口索引、动态复算统计、原B1比较及各指标晋级判断。
[repetition-plan.json](repetition-plan.json)保存预定测试顺序和统计规则。
双rank真实BF16模型头一致性、算子捕获、请求窗口内草稿图重放、Microflow控制、KV策略及引擎缓存盐记录均随归档保存。
请求输出、逐token计时、MTP计数、APC记录、设备采样、窗口边界及退出状态可独立检查。 控制器请求drain后Manager可返回130；客户端必须为0，服务drain完成与设备释放单独核验。

结果仅用于服务性能筛选。256-token输出不是完整MMLU-Pro答案评分，未重新评估任务正确率，也不构成整个vSpec的质量或通用性能认证。

## 资格检查与诊断

- ON r1：设备锁被另一实验持有，未启动服务。
- ON r2：实际BetterScale donor与初始宿主源码pin不匹配，准入失败，未测请求。
- ON r3：新增日志字段参数重名，模型头检查记录失败，未测请求。
- ON r7：修复后的200请求资格检查；旧观察器没有识别FIA包装内部的草稿图，因此不纳入正式三组。

原始记录不改写，不与正式统计合并。模型权重和编译缓存产物不进入仓库。

## 核验

在单独目录解开某个正式归档及核验工具，然后运行：

```sh
tar -xf verification-tools.tar.gz
python verify_school028_run.py <解包根目录> <run_id>
```

验证器核对固定配置、输入和源码SHA256、200请求和token总数、两rank执行记录、资源释放。
`tests/test_vspec_screen_evidence.py`独立读取全部归档，复算吞吐、请求时延、三组中位数及配对几何平均。
