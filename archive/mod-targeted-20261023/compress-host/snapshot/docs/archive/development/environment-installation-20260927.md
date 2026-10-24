# Historical 2026-09-27 Host Installation

English | [简体中文](../../development/environment-installation.zh.md)

> Historical record: absolute paths and installed environments below refer to
> the previous workspace. Use the [current installation guide](../../development/environment-installation.md) on this machine.

## Host stack dependency boundary

The plugin is installed into a pre-provisioned host and does not own the host's
Python dependency graph. Its wheel therefore has no core `Requires-Dist`; the
supported `vllm-ascend` range remains in the Extension Manager manifest.
When `auto_calibrate` distributes a model across multiple devices, provision
the optional calibration dependency without re-resolving the accelerator host:

```bash
python -m pip install --no-deps 'accelerate>=1.10,<2'
```

The equivalent package extra is `vllm-ascend-kvcompress-hust[calibration]` for
environments whose dependency lock already owns PyTorch and Transformers.

The original development environment has PyTorch 2.10.0, torch-npu
2.10.0.post2, and CANN 9.0.1. The synchronized Ascend host's
`requirements.txt` instead pins PyTorch 2.13.0 and torch-npu 2.13.0rc1, and
its CMake configuration refuses to build with PyTorch 2.10.0. An earlier
editable, in-place rebuild stopped at that version check before replacing the
old native `.so`; no upstream source was manually changed. The official
[TorchNPU 2.13.0rc1 image](https://hub.docker.com/r/ascendai/torch-npu)
uses CANN 9.1.0. Do not bypass the host's PyTorch ABI guard or claim graph-mode
qualification from the old binary.

On 2026-09-27, a matched stack was installed under
`/workspace/zjw/.envs/kvcompress-0.8-host` without replacing the system CANN
or the original development environment: CANN Toolkit/910B Ops/NNAL 9.1.0,
PyTorch 2.13.0+cpu, torch-npu 2.13.0rc1, torchvision 0.28.0+cpu, and
torchaudio 2.11.0+cpu. The official archives and wheels were SHA-256 checked.
An Ascend 910B2 executed a `torch.ones(4, device="npu:0").sum()` smoke check
and returned 4.0. The synchronized Ascend host was then rebuilt in its original
checkout and its native module imported successfully. On device 2, both the
host GDN output operator and the plugin-owned Triton fallback matched a BF16
reference with maximum absolute error 0.000244. Graph mode, model serving,
and benchmark qualification remain separate checks. The matched stack then
started Qwen3.5-35B-A3B BF16 with TP=2 on devices 2/3 at 262,144 context,
APC, MTP2, async scheduling, `mamba_cache_mode=align`, and
`FULL_AND_PIECEWISE` graph capture. Its official SWE C4/60-second protocol
check passed with `valid=true`, zero failed requests, 31 requests started,
and server-side prefix hits. The 900-second paired benchmark remains a
separate acceptance step. PyTorch 2.13 AOTAutograd cache initially raised an
`expected OutputCode, got GraphModuleImpl` assertion; setting
`TORCHINDUCTOR_AUTOGRAD_CACHE=0` for the service disabled that cache without
disabling model compilation or graph capture. Use
`scripts/kvcompress_ascend_env.sh COMMAND [ARGS...]` for this isolated stack;
the launcher removes old CANN 9.0.1 paths before sourcing CANN 9.1 and NNAL.
When the original host checkout has installed its own standard AscendC kernels,
the launcher also sources its generated vendor environment before Python starts;
the plugin's fallback kernel definition and JIT remain inside the plugin.
The generic PyPI PyTorch 2.13 wheel is not suitable here: it requires CUDA
shared libraries at import. Use the official ARM64 `+cpu` wheel with torch-npu.
This 2.13.0rc1 pairing follows the synchronized HUST host's own pin; it is a
prerelease and is not presented as the general Ascend community recommended
combination in the [Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).
Service-level validation is required before using it for results.
After installing the current plugin source as an editable package in this
isolated stack, its CPU/integration suite reports **139 passed, 1 skipped**.
The read-only Frontier preflight also confirms the model shards and frozen
SWE/AgentX inputs. Neither result substitutes for a rebuilt host native module
or a real NPU service test.

Do not combine the old vLLM 0.28 / vLLM-Ascend 0.25 snapshots or the vLLM 0.29
host now under adaptation with Triton-Ascend 3.2.2. The conflict is not a
missing NumPy distribution:
Triton-Ascend 3.2.2 requires `numpy==1.26.4`, while vLLM 0.28 requires
`opencv-python-headless>=4.13` and those available OpenCV wheels require
`numpy>=2`. No NumPy version satisfies both. Pinning OpenCV 4.9 is also invalid
because it fails vLLM's `>=4.13` constraint. Provision the matched
Triton-Ascend 3.6/NumPy 2 host lock, then install this plugin with `--no-deps`.

If pip reports both an unversioned and an exact `vllm`/`vllm-ascend`
requirement, the unversioned entry came from older plugin metadata rather than
an intentional duplicate host install. Build and test the current wheel, whose
core dependency list is empty.

This note records the earlier installation failure and the current
Triton-Ascend build boundary. Do not compile a temporary copy of the upstream
source; the earlier diagnostic copies have been removed. Build-time Ascend
patch application is expected upstream behavior, but upstream source must not
be manually edited. Plugin-specific Qwen GDN operator definition, compilation,
and installation belong inside this plugin, not the host or Triton repository.

## Symptom and cause

Distribution metadata reported `triton-ascend 3.6.0+git8f0a4de8`, but importing
`triton` failed with:

```text
ModuleNotFoundError: No module named 'triton._C.libtriton.ascend'
```

The old wheel's CMake cache showed an empty `TRITON_PLUGIN_DIRS`; its
`libtriton.so` included generic bindings but not the Ascend Python module.
The current `setup.py` automatically invokes `third_party.ascend.build`.
Rebuild from the current commit and verify the actual Ascend import, not just
the distribution version string.

## Build from the original checkout

The dependencies below were verified at Triton commit `b90ab001238f`. Build
directly from the original checkout. Check its state first, and distinguish
automatic build rewrites from manual source changes. If an automatic rewrite
must be removed, inspect the exact files and use the repository's process;
do not blindly clean the worktree:

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

`LLVM_SYSPATH` points to the matching LLVM toolchain already cached on this
machine; use the matching cache on other machines or allow the build to download
it. Adjust `MAX_JOBS` to available resources. The wheel built and imported on
2026-09-27 is `3.6.0+gitb90ab001`, SHA-256
`9b885f318e8f65893be38b64ca6fbd9549dda2f7c12e28ef114fb8895a15c1b7`.
This identifies the wheel built from the original checkout and installed into
the isolated host environment, replacing the earlier temporary-copy artifact.
The build's automatic Ascend patch left 17 tracked files dirty; these are the
upstream build script's own patch targets, not manual edits.

Validate before running plugin tests:

```bash
/workspace/zjw/vllm-ascend-kvcompress-hust/scripts/kvcompress_ascend_env.sh python - <<'PY'
import triton
from triton._C.libtriton.ascend import ir
print(triton.__version__)
print(ir)
PY
```

Then reinstall this plugin without attempting dependency resolution:

```bash
/workspace/zjw/vllm-ascend-kvcompress-hust/scripts/kvcompress_ascend_env.sh \
  python -m pip install --no-deps --no-build-isolation \
    -e /workspace/zjw/vllm-ascend-kvcompress-hust
```

This installation check establishes that Triton/Ascend Python bindings import;
it does not establish plugin functional acceptance on the new host. Formal
acceptance still requires source commit, wheel hash, and NPU runtime evidence.
The 3.2.2 wheel is diagnostic only, not evidence for the current host stack.
