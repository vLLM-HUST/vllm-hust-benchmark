# GSM8K 固定子集：可下载原件

这是历史实验原件的首次公开归档。完整 HTTP 请求/响应、两臂逐题评分及错误/截断样本、服务日志、实际运行脚本、依赖清单、源码 diff 和插件源码快照保存在分片包中；原件字节未修改。原始配置里的本机路径仅表示当时来源，核验不访问该机器。

下载本目录的 manifest.json、verify.py 及全部三个 .part 文件，在本目录运行：

```bash
python verify.py
```

该命令只用标准库，逐一校验分片、归档及原文件哈希，并核对请求、答案、两种评分 filter 和计划分母。若还要复跑原始汇总与 bootstrap 计算，使用已安装 numpy 的 Python：

```bash
python verify.py --recompute
```

解包可按 manifest.parts 顺序拼接得到 manifest.archive；解包后直接阅读 bf16/private-results.json、int8/private-results.json、两臂 HTTP JSONL 及日志。原始 SHA256SUMS 覆盖的文件均包含在包中。环境二进制、下载缓存和字节码缓存不归档；模型权重不在此包。

题文/参考答案来自 GSM8K，按随附 GSM8K-LICENSE 保留 MIT 许可。源码快照保留各自原始版权声明，宿主源码按原 diff 记录；这不是新的代码发布或性能实验。

本次只保留功能与容量结论。独立短上下文质量记录不能用于长上下文或总体质量无损声明。
