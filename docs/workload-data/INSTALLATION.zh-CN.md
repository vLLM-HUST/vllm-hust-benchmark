# 安装数据组件：避免为一个小包检出整个历史归档

本组件是独立软件包，不包含数据集、模型、答案或科研观测模块。已经有仓库时，按[组件说明](../../workload-preparation/README.md)本地安装即可。

只需要数据组件的使用者，优先使用 **CPU 正确性 smoke 验证后的 wheel 工件**。工作流固定检出 PR head SHA，构建 wheel 后用该 wheel
安装并测试，全部通过才上传。工件同时包含 SHA256SUMS 和 BUILD_METADATA.json，后者记录实际源码提交。

## 从指定提交的成功 CI 获取

在 PR 检查中选择成功的 `Workload data CPU correctness smoke` run，核对 head SHA 与期望版本一致。不要把任意“最新成功”当成固定版本。

```sh
gh run download RUN_ID --repo vLLM-HUST/vllm-hust-benchmark --name workload-data-HEAD_SHA --dir /tmp/workload-data-install
cd /tmp/workload-data-install
sha256sum -c SHA256SUMS
python -m pip install 'vllm_hust_workload_data-0.1.0-py3-none-any.whl[data]'
python -c "from vllm_hust_workload_data.provider import provider_identity; print(provider_identity())"
```

Windows 可用 `Get-FileHash -Algorithm SHA256` 核对 wheel，并检查 BUILD_METADATA.json
的源码提交。下载目录使用短路径即可，不需要更改全局 Git 配置、完整检出历史归档或绕过文件验证。

这是 **PR 软件交接工件**，不是已合并发布版；CI 工件保留 30 天，需要后续正式版本时应从获认可提交重新构建/发布，不能把过期临时链接当永久依赖。研究消费者应固定源码 SHA 和 wheel
SHA，而不只固定 0.1.0 版本号。

## 与研究仓的关系

导入名为 `vllm_hust_workload_data`，不覆盖研究仓的
`llm_serving_workloads`。研究环境从数据包读取原始输入引用，自己的观测、标准化、特征及分析继续由研究仓维护。

使用 wheel 只解决安装与依赖定位，不替代真实数据获取、逐来源适配和模型/任务/评分验证。私有与 gated 数据仍由获授权使用者自行取得。
