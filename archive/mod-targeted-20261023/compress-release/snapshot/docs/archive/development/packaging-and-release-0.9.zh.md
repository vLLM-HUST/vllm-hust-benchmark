# 0.9 历史打包与发布指南

[English](../../development/packaging-and-release.md) | 简体中文

> 0.9.0 历史发布配方。当前流程见[打包发布指南](../../development/packaging-and-release.zh.md)。

本仓库遵循 BidKV 打包流程。PyPI distribution 为
`vllm-ascend-kvcompress-hust`，Python 包为 `vllm_ascend_kvcompress`，稳定的
Extension Manager ID 为 `org.vllm-hust.ascend-kvcompress`。

## 发布门槛

发布前必须保持项目版本、`__version__` 和 manifest 版本一致；完成 CPU、包、NPU、
生命周期及声明的服务测试；冻结宿主、模型、校准、数据来源；审核完整 diff。PyPI
文件不可覆盖，任何代码变更都必须提升版本。

0.9.0 仍为实验性版本，并发布供外部压缩方法使用的 method API v1。当前验证仅支持
有边界的工程性能结论，不支持生产可用或
官网正式成绩的声明。发布门槛以当前 [Frontier 规程](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.zh.md)
为准：同步宿主的匹配运行时、262,144 上下文下 APC/MTP2/async/
FULL_AND_PIECEWISE/align 的完整服务检查、官方 SWE 和 AgentX 两项 900 秒
工作负载均有可审计的 B0/B1 配对，并且至少一项 Frontier 负载协议有效且取得
正收益；另一项若退化须如实披露。历史 V4.6 记录不能代替这些结果。
发布负责人确认精确候选制品前，不得上传 PyPI。
合并后 AgentX 900 秒配对的总输出吞吐提升 11.18%、解码 P90 提升 5.27%；
SWE 配对协议有效，但相应指标下降 2.95% 和 1.34%。两项均未实际达到 256K
prompt，也均非官网正式提交。
2026-10-08 的精确 0.9.0 候选在固定发布栈上重新完成两项官方 900 秒配对。
SWE 解码 P90 提升 1.05%，但吞吐和 TTFT 退化；AgentX 解码 P90 提升 1.14%，
TTFT 与 ITL 改善，总吞吐在 0.0001% 内持平。两项均协议有效，无失败请求或
无效原因。这些本地短测满足有边界的发布门槛，但仍只是单次工程观测，不是一小时
正式测试或官网成绩。

精确的 0.8.0 wheel 在锁定的 CANN 9.1/PyTorch 2.13 环境中通过 Ascend 内核与
插件自带 GDN 回退算子数值短测。使用该 wheel 的 Qwen2.5-14B-Instruct
FP16/TP=1 服务在 NPU 0、`--enforce-eager` 下返回 `/health` 200；一次
10,843-token 的事实检索输入正确回答“42”，调度器与 worker 均确认物理 KV
压缩到 8,192 token。相同宿主的默认图编译启动**未通过**：AOT 路径在初始化
缓存时出现 `AssertionError: expected OutputCode, got GraphModuleImpl`。
因此这次追加的 wheel 服务短测只验证 eager 路径，不代表 Qwen2.5 图编译
模式验收，也不改变上述 Frontier 结果的性质。

## 构建与检查

优先采用 BidKV 的 `uv` 流程：

```bash
uv build --no-sources --out-dir dist
```

若发布环境没有 `uv`，已验证的等价回退方式为：

```bash
python -m build --no-isolation --outdir dist
```

随后只检查本次候选文件：

```bash
python -m zipfile -l \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
python -m twine check \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
sha256sum dist/vllm_ascend_kvcompress_hust-0.9.0*
```

wheel 必须包含代码、`LICENSE`、`NOTICE` 和
`manifests/vllm-hust-extension-v0.3.json`；entry-point metadata 必须同时包含
`vllm.general_plugins`、`vllm_hust.extension_bundles` 和
`vllm-ascend-kvcompress-calibrate` 命令入口。两个发行包均不得包含
`artifacts/*.pt`、原始数据、服务日志或历史 `docs/dev/results`。sdist 包含静态
HTML 榜单及其生成的浏览器数据 bundle；`docs/evidence` 只可包含聚合证据摘要。

## 隔离生命周期

在支持的宿主环境中安装 wheel，且不把源码 checkout 加入 `PYTHONPATH`：

```bash
python -m pip install --no-deps \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl
vllm-hust-ext extension validate org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension configure org.vllm-hust.ascend-kvcompress \
  --file /absolute/path/triattention.json
vllm-hust-ext extension enable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension check org.vllm-hust.ascend-kvcompress
vllm-hust-ext run --dry-run -- python -c 'print("manager-run-ok")'
vllm-hust-ext extension disable org.vllm-hust.ascend-kvcompress
vllm-hust-ext extension forget org.vllm-hust.ascend-kvcompress
python -m pip uninstall -y vllm-ascend-kvcompress-hust
```

wheel 的核心 `Requires-Dist` 必须为空，尤其不得把 `vllm`、`vllm-ascend`、
Triton-Ascend、NumPy 或 OpenCV 加入插件核心依赖。加速器宿主由另一套经过验证的
锁定清单部署，扩展清单负责声明兼容宿主范围。安装插件时重新解析宿主，可能把
仅支持 NumPy 1 的 Triton-Ascend 3.2.2 与当前 vLLM 中仅支持 NumPy 2 的 OpenCV
约束混在一起。

确认 `extension list` 不再发现插件。然后重装、配置并启用将要发布的精确 wheel。

## 上传与发布后验证

使用由 PyPI `intellistream` 组织/项目授权的项目级 token，不在命令行明文传递，也不
提交到仓库：

```bash
export UV_PUBLISH_TOKEN='<从密钥存储读取>'
uv publish --check-url https://pypi.org/simple \
  dist/vllm_ascend_kvcompress_hust-0.9.0-py3-none-any.whl \
  dist/vllm_ascend_kvcompress_hust-0.9.0.tar.gz
unset UV_PUBLISH_TOKEN
```

若受保护发布器使用 Twine，则从密钥存储设置 `TWINE_USERNAME=__token__` 和
`TWINE_PASSWORD`，上传同样两个明确文件。最后从正式 PyPI 无缓存安装 0.9.0，
核对哈希、Manager 发现/启用、`/health` 和一个正确性用例。
