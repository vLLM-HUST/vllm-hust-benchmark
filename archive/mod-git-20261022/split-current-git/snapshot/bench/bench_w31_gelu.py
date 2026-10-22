# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""W3.1 bench: fused ``gelu_and_mul`` vs three reference schemes on 910B2.

Single-op timing (``torch.npu.Event``, warmup>=10, iters>=100, min-of-3) at the
Qwen2.5-Coder-14B MLP shape (inter=13824, B in {64, 256}), four schemes:

  1. ``native_chain``   -- vLLM ``GeluAndMul.forward_native`` (fp32 cast + gelu +
     mul), the kickoff §3 baseline (~140 us at B=64).
  2. ``npu_gelu+mul``   -- the cheapest ready-made two-step chain
     (``npu_gelu`` fp32 + fp32 mul); also measured on the bf16 path.
  3. ``fi_gelu``        -- the fused triton-ascend kernel (this package), both
     with an internal allocation and with a caller-supplied ``out=``.
  4. ``copy_floor``     -- ``x.clone()`` of the full ``[B, 2*inter]`` input, the
     kickoff's memory-lower-bound proxy (~31 us at B=64).

The bench also measures a bare triton no-op launch: on this triton-ascend build
the *host* launch path alone costs ~100 us (grid- and work-independent), which
is the structural reason scheme 3 cannot reach scheme 4 -- see the JSON
``triton_noop_launch`` row and the W3.1 report.

Run (NPU card 7, from a non-/vllm-workspace CWD):

    cd /tmp && export ASCEND_RT_VISIBLE_DEVICES=7
    python3 /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_gelu.py
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch_npu

from vllm_ascend_split_batch.fi_gelu import api
from vllm_ascend_split_batch.fi_gelu.kernels import _erf_probe_kernel

INTER = 13824
BATCHES = (64, 256)
WARMUP = 10
ITERS = 100
ROUNDS = 3

OUT = Path(__file__).resolve().parent / "results" / "bench_w31_gelu.json"


def bench_event(
    fn, iters: int = ITERS, warmup: int = WARMUP, rounds: int = ROUNDS
) -> float:
    """min-of-``rounds`` per-call us, events around a loop of ``iters`` calls."""
    best = float("inf")
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.npu.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / iters)
    return best


def _native_callable():
    """vLLM ``GeluAndMul.forward_native`` wrapped like the kickoff probe.

    The op is constructed inside ``set_current_vllm_config`` (as the kickoff
    probe does) and then called outside it, so per-call config construction
    stays out of the timed loop.
    """
    try:
        from vllm.config import set_current_vllm_config
        from vllm.config.vllm import VllmConfig
        from vllm.model_executor.layers.activation import GeluAndMul
    except Exception as exc:  # noqa: BLE001 - keep the other schemes runnable
        print(f"[bench] native chain unavailable: {exc!r}", file=sys.stderr)
        return None
    with set_current_vllm_config(VllmConfig()):
        act = GeluAndMul()
    return lambda x: act.forward_native(x)


def bench_noop_launch() -> float:
    dummy = torch.zeros(8, dtype=torch.float32).npu()
    _erf_probe_kernel[(1,)](dummy, dummy, 1, BLOCK=1)
    torch.npu.synchronize()
    return bench_event(lambda: _erf_probe_kernel[(1,)](dummy, dummy, 1, BLOCK=1))


def main() -> int:
    torch.npu.set_device(0)
    try:
        dev_name = torch.npu.get_device_name(0)
    except Exception:  # noqa: BLE001 - cosmetic only
        dev_name = "npu:0"
    print(
        f"device: {dev_name}  inter={INTER}  "
        f"(iters={ITERS}, warmup={WARMUP}, rounds={ROUNDS})"
    )

    noop_us = bench_noop_launch()
    print(f"triton no-op launch (host overhead, grid=(1,)): {noop_us:.1f} us\n")

    native = _native_callable()
    rows: list[dict] = []

    header = f"{'B':>5} {'op':<38} {'us':>8} {'vs native':>10} {'vs floor':>9}"
    print(header)
    print("-" * len(header))

    for batch in BATCHES:
        g = torch.Generator().manual_seed(0)
        x = (
            torch.randn(batch, 2 * INTER, generator=g, dtype=torch.float32)
            .bfloat16()
            .npu()
        )
        gate = x[:, :INTER]
        up = x[:, INTER:]
        out = torch.empty(batch, INTER, dtype=torch.bfloat16).npu()

        schemes: list[tuple[str, Callable[[], object]]] = []
        if native is not None:
            schemes.append(("native_chain(fp32_cast+gelu+mul)", lambda x=x: native(x)))
        schemes.append(
            (
                "npu_gelu+mul(fp32 two-step)",
                lambda g=gate, u=up: torch_npu.npu_gelu(g.float()) * u.float(),
            )
        )
        schemes.append(
            (
                "npu_gelu+mul(bf16 two-step)",
                lambda g=gate, u=up: torch_npu.npu_gelu(g) * u,
            )
        )
        schemes.append(
            ("fi_gelu(fused triton kernel)", lambda x=x: api.gelu_and_mul(x))
        )
        schemes.append(
            (
                "fi_gelu(out= preallocated)",
                lambda x=x, o=out: api.gelu_and_mul(x, out=o),
            )
        )
        schemes.append(("copy_floor(x.clone())", lambda x=x: x.clone()))

        timings = {name: bench_event(fn) for name, fn in schemes}
        floor_us = timings["copy_floor(x.clone())"]
        native_us = timings.get("native_chain(fp32_cast+gelu+mul)")

        for name, us in timings.items():
            rows.append(
                {
                    "batch": batch,
                    "inter": INTER,
                    "op": name,
                    "us": round(us, 1),
                    "vs_native": round(us / native_us, 3) if native_us else None,
                    "vs_copy_floor": round(us / floor_us, 3),
                }
            )
            print(
                f"{batch:>5} {name:<38} {us:>8.1f} "
                f"{(us / native_us if native_us else float('nan')):>10.2f} "
                f"{us / floor_us:>9.2f}"
            )
        print()

    rows.append(
        {
            "batch": None,
            "inter": None,
            "op": "triton_noop_launch",
            "us": round(noop_us, 1),
            "vs_native": None,
            "vs_copy_floor": None,
        }
    )

    for r in rows:
        if r["op"].startswith("fi_gelu"):
            print(f"fi_gelu B={r['batch']}: {r['op']} = {r['us']}us")
    print("kickoff anchors: native ~140.4us, floor ~30.8us (B=64)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
