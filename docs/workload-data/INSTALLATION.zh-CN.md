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
# 不同提交可能仍是0.1.0；上一步用于依赖解析，不能保证替换已安装代码。
python -m pip install --no-deps --force-reinstall 'vllm_hust_workload_data-0.1.0-py3-none-any.whl'
python -c "from vllm_hust_workload_data.provider import provider_identity; print(provider_identity())"
```

Windows 可用 `Get-FileHash -Algorithm SHA256` 核对 wheel，并检查 BUILD_METADATA.json
的源码提交。下载目录使用短路径即可，不需要更改全局 Git 配置、完整检出历史归档或绕过文件验证。

不要以`Requirement already satisfied`或`already installed with the same version`认定已安装目标工件；这可能仍是同版本号的旧代码。`--no-deps --force-reinstall`只替换指定数据包，不强制升级原生benchmark环境的其他固定依赖。切换工件后还应逐文件核对实际安装内容：

```sh
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
import vllm_hust_workload_data as package

wheel = Path("vllm_hust_workload_data-0.1.0-py3-none-any.whl")
base = Path(package.__file__).parent.parent
checked = 0
with ZipFile(wheel) as archive:
    for name in archive.namelist():
        if name.startswith("vllm_hust_workload_data/") and not name.endswith("/"):
            if (base / name).read_bytes() != archive.read(name):
                raise RuntimeError(f"installed payload differs: {name}")
            checked += 1
assert checked > 0
print({"installed_payload_files_byte_matched": checked})
PY
```

该检查只涉及软件包文件，不读取数据集或运行模型；仍须先校验wheel的SHA256及BUILD_METADATA对应源码提交。Windows可将同一段Python代码保存到任务私有脚本运行，或在Python交互环境逐行执行。

这是 **PR 软件交接工件**，不是已合并发布版；CI 工件保留 30 天，需要后续正式版本时应从获认可提交重新构建/发布，不能把过期临时链接当永久依赖。研究消费者应固定源码 SHA 和 wheel
SHA，而不只固定 0.1.0 版本号。

## 与研究仓的关系

导入名为 `vllm_hust_workload_data`，不覆盖研究仓的
`llm_serving_workloads`。研究环境从数据包读取原始输入引用，自己的观测、标准化、特征及分析继续由研究仓维护。

使用 wheel 只解决安装与依赖定位，不替代真实数据获取、逐来源适配和模型/任务/评分验证。私有与 gated 数据仍由获授权使用者自行取得。

## 不依赖临时CI工件的固定源码安装

已有能力迁移对应的数据锁为`1a9776865546d85edd667425661b4a5603696e35`。需要Python 3.10以上和Git；建议在独立虚拟环境中执行：

```sh
python -m pip install 'vllm-hust-workload-data[data] @ git+https://github.com/vLLM-HUST/vllm-hust-benchmark.git@1a9776865546d85edd667425661b4a5603696e35#subdirectory=workload-preparation'
python -c "from vllm_hust_workload_data.provider import provider_identity; print(provider_identity())"
python -m vllm_hust_workload_data.catalog source GSM8K
python -m vllm_hust_workload_data.pipeline --help
```

这是安装公共数据能力，不会取得原题或执行模型。固定Git提交不等于固定构建wheel字节；源码安装由PEP
610记录原仓库、提交及子目录，不能拿本地另一份wheel的SHA冒充本次构建摘要。研究侧已采用这一Git锁，消费核验见[总清单第八节](WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md)。

七个获取/校验工具属于源码/sdist入口，不是上述wheel中的命令。需要这些工具时，取得同一提交的源码再从根目录调用`workload-preparation/source_tools/`；用法见[数据获取](../../workload-preparation/docs/dataset-downloads.md)。准备、索引及回查见[组件用法](../../workload-preparation/README.md)。原始数据根目录、显式manifest和输出目录均由使用者提供；不得自动回退研究仓旧reader或旧QBI派生题。
