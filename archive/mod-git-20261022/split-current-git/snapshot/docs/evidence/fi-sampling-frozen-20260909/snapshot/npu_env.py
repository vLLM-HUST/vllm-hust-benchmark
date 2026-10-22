"""NPU / triton-ascend device helpers (standalone, no vllm_ascend dependency).

Mirrors the device-property probe that vllm_ascend performs before launching
triton-ascend kernels (vllm_ascend/ops/triton/triton_utils.py:46-54): a bare
``triton`` kernel launch on a fresh process needs the driver's device
properties initialized first.
"""

from __future__ import annotations

import sys

import torch

_initialized = False


def init_device_properties_triton() -> None:
    """Probe NPU device properties through the triton-ascend driver.

    Idempotent. Must be called once per process before the first triton
    kernel launch (otherwise the ascend backend may fail to resolve grid /
    num_warps information on a bare call).
    """
    global _initialized
    if _initialized:
        return
    try:
        import triton  # noqa: WPS433 (runtime probe)

        _ = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
    except Exception as exc:  # noqa: BLE001 — probe is best-effort; real errors surface at launch
        # Non-fatal: some driver builds resolve properties lazily at first launch.
        print(f"[fi_sampling] device-property probe skipped: {exc!r}", file=sys.stderr)
    _initialized = True


def current_npu_device() -> torch.device:
    return torch.device("npu", torch.npu.current_device())


def is_npu_available() -> bool:
    try:
        return torch.npu.is_available()  # type: ignore[union-attr]
    except AttributeError:  # pragma: no cover — torch without torch_npu
        return False
