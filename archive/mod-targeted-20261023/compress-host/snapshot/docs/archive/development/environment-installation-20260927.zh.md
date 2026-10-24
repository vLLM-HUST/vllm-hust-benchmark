# 2026-09-27 历史宿主安装记录

[English](../../development/environment-installation.md) | 简体中文

> 历史记录：下文绝对路径及安装环境属于旧工作区。本机请使用
> [当前安装指南](../../development/environment-installation.zh.md)。

## 宿主栈依赖边界

插件安装到预先部署好的宿主环境中，不负责解析宿主的 Python 依赖图。因此插件
wheel 的核心 `Requires-Dist` 为空；支持的 `vllm-ascend` 范围仍由 Extension
Manager 清单声明。

原开发环境为 PyTorch 2.10.0、torch-npu 2.10.0.post2 与 CANN 9.0.1；同步后的
Ascend 宿主 `requirements.txt` 则锁定 PyTorch 2.13.0、torch-npu 2.13.0rc1，
其 CMake 配置明确拒绝 PyTorch 2.10.0。此前的原仓库原位 editable 重编译在这一
版本检查处停止，未替换旧原生 `.so`，也未手工改动上游源码。
[TorchNPU 官方 2.13.0rc1 镜像](https://hub.docker.com/r/ascendai/torch-npu)
使用 CANN 9.1.0。不能绕过 PyTorch ABI 保护或凭旧二进制宣称图模式通过。

2026-09-27 已在 `/workspace/zjw/.envs/kvcompress-0.8-host` 安装匹配的隔离栈，
不替换系统 CANN 或原开发环境：CANN Toolkit/910B Ops/NNAL 9.1.0、PyTorch
2.13.0+cpu、torch-npu 2.13.0rc1、torchvision 0.28.0+cpu 和 torchaudio
2.11.0+cpu。官方包和 wheel 均已核对 SHA-256。Ascend 910B2 上执行
`torch.ones(4, device="npu:0").sum()` 返回 4.0。随后在原仓库重编译同步后的
Ascend 宿主，原生模块导入成功。在 2 号卡上，宿主 GDN 输出算子与插件内 Triton
回退算子均与 BF16 参考实现吻合，最大绝对误差 0.000244。图模式、模型服务与
基准资格仍需分别验证。随后匹配的环境在 2/3 号卡上以 TP=2、262,144 上下文、
APC、MTP2、异步调度、`mamba_cache_mode=align` 和
`FULL_AND_PIECEWISE` 图捕获启动 Qwen3.5-35B-A3B BF16。官方 SWE 的
C4／60 秒协议短测 `valid=true`、失败请求为零，共启动 31 次请求，服务端确认
前缀命中；900 秒配对测试仍是独立的验收步骤。PyTorch 2.13 的
AOTAutograd 缓存最初触发 `expected OutputCode, got GraphModuleImpl` 断言；
服务设置 `TORCHINDUCTOR_AUTOGRAD_CACHE=0` 后仅关闭该缓存，不关闭模型编译
或图捕获。使用
`scripts/kvcompress_ascend_env.sh COMMAND [ARGS...]` 启动隔离栈；脚本先清除
旧 CANN 9.0.1 路径，再加载 CANN 9.1 和 NNAL。如果原始宿主仓库已安装其标准
AscendC 算子，启动器还会在 Python 启动前加载其生成的 vendor 环境；插件回退
算子的定义与 JIT 仍全部留在插件内部。PyPI 的通用 PyTorch 2.13
wheel 导入时要求 CUDA 动态库，不适合这里；应使用官方 ARM64 `+cpu` wheel
搭配 torch-npu。
这里的 2.13.0rc1 组合遵循项目组同步后的宿主版本锁定；它是预发布版，
不应描述为[昇腾社区版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
中的通用推荐组合。产生测试成绩前仍须完成服务级验证。
在此隔离栈中以 editable 方式安装当前插件源码后，CPU/集成单测为 **139 通过、
1 跳过**；只读 Frontier 预检也确认模型分片和已冻结的 SWE/AgentX 输入。
这些结果均不能替代宿主原生模块重编译或真实 NPU 服务测试。

不要把旧版 vLLM 0.28 / vLLM-Ascend 0.25 快照或正在适配的 vLLM 0.29
宿主与 Triton-Ascend 3.2.2 混装。这里并不是缺少 NumPy 分发包：
Triton-Ascend 3.2.2 要求
`numpy==1.26.4`，而 vLLM 0.28 要求 `opencv-python-headless>=4.13`，这些可用
OpenCV wheel 又要求 `numpy>=2`，不存在同时满足二者的 NumPy 版本。固定 OpenCV
4.9 也不成立，因为它不满足 vLLM 的 `>=4.13` 约束。应先部署配套的
Triton-Ascend 3.6 / NumPy 2 宿主锁，再用 `--no-deps` 安装本插件。

如果 pip 同时列出不带版本和精确版本的 `vllm` / `vllm-ascend`，不带版本的条目
来自旧插件元数据，并非有意重复安装宿主。请构建并测试核心依赖列表为空的当前
wheel。

本文记录旧环境的故障原因及当前 Triton-Ascend 的构建边界。不要复制上游源码到
临时目录编译；之前用于诊断的源码副本已清理。构建脚本自动应用 Ascend patch
属于上游预期的构建行为，但不应手工修改上游源码。插件专属的 Qwen GDN 算子
定义、编译和安装位于插件内，不应加入宿主或 Triton 仓库。

## 现象与原因

发行元数据显示 `triton-ascend 3.6.0+git8f0a4de8`，但导入 `triton` 报错：

```text
ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'
```

旧 wheel 的 CMake cache 中 `TRITON_PLUGIN_DIRS` 为空；生成的
`libtriton.so` 有通用 binding，但没有 Ascend Python 模块。新版源码的
`setup.py` 已自动调用 `third_party.ascend.build`，需要从当前提交重新构建
并实际验证 Ascend 模块导入，不能只检查发行元数据中的版本号。

## 从原始仓库构建

以下依赖参数已在 Triton 提交 `b90ab001238f` 上验证。应直接在原始仓库构建，
先检查工作树状态，并区分构建脚本自动改写和任何人工源码改动。若需消除自动
改写，先核实目标文件再按仓库流程处理，不盲目清理工作树：

```bash
git -C /workspace/zjw/triton-ascend-hust status --short

env \
  TRITON_HOME=/workspace/zjw/.cache/kvcompress-triton-home \
  TRITON_CACHE_DIR=/workspace/zjw/.cache/kvcompress-triton-runtime \
  UV_CACHE_DIR=/workspace/zjw/.cache/kvcompress-uv \
  CCACHE_DIR=/workspace/zjw/.cache/kvcompress-ccache \
  TMPDIR=/workspace/zjw/.cache/kvcompress-build-tmp \
  LLVM_SYSPATH=/root/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-arm64 \
  JSON_SYSPATH=/root/.triton/json \
  TRITON_BUILD_PROTON=OFF \
  TRITON_BUILD_NPUIR=OFF \
  TRITON_WHEEL_NAME=triton-ascend \
  TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF \
  MAX_JOBS=32 \
  /root/miniconda3/envs/vllm-hust-dev/bin/uv build --wheel \
    --no-build-isolation \
    --python /workspace/zjw/.envs/kvcompress-0.8-host/bin/python \
    --out-dir /workspace/zjw/triton-ascend-hust/dist \
    /workspace/zjw/triton-ascend-hust

/root/miniconda3/envs/vllm-hust-dev/bin/uv pip install \
  --python /workspace/zjw/.envs/kvcompress-0.8-host/bin/python \
  --reinstall --no-deps \
  /workspace/zjw/triton-ascend-hust/dist/triton_ascend-*.whl
```

`LLVM_SYSPATH` 是本机已缓存的匹配 LLVM 工具链；其他机器应使用对应的缓存目录，
或让构建脚本下载。`MAX_JOBS` 按机器资源调整。2026-09-27 生成并导入通过的
wheel 版本为 `3.6.0+gitb90ab001`，SHA-256 为
`9b885f318e8f65893be38b64ca6fbd9549dda2f7c12e28ef114fb8895a15c1b7`。
它标识从原始仓库构建并安装到隔离环境的 wheel，已替换早先临时副本的产物。
构建脚本自动打 Ascend 补丁，使 17 个受其管理的跟踪文件显示为已改；这不是
人工编辑上游源码。

运行插件测试前先验证：

```bash
/workspace/zjw/vllm-ascend-kvcompress-hust/scripts/kvcompress_ascend_env.sh python - <<'PY'
import triton
from triton._C.libtriton.ascend import ir
print(triton.__version__)
print(ir)
PY
```

然后不解析依赖地重装本插件：

```bash
/workspace/zjw/vllm-ascend-kvcompress-hust/scripts/kvcompress_ascend_env.sh \
  python -m pip install --no-deps --no-build-isolation \
    -e /workspace/zjw/vllm-ascend-kvcompress-hust
```

此安装检查仅证明 Triton/Ascend Python 绑定可用，不证明插件已通过新版宿主
功能验收。正式验收仍需记录源码提交、wheel 哈希和 NPU 运行结果。3.2.2 wheel
只能用于降级诊断，不能作为当前宿主栈的证据。
