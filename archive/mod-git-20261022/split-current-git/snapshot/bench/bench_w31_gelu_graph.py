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

"""W3.1 graph bench: ``gelu_and_mul`` three schemes under ACL graph replay.

Companion to ``bench_w31_gelu.py`` (eager, ``results/bench_w31_gelu.json``).  The
eager run showed the fused triton-ascend kernel is dominated by its *host*
launch path (~110 us on this build, grid/work independent), so it loses to the
plain ``npu_gelu+mul`` bf16 two-step chain despite doing a single fused pass.
This bench asks whether that ranking flips once the launch is amortised by ACL
graph capture/replay -- and, first of all, whether the triton kernel can be
captured at all.

Shapes: Qwen2.5-Coder-14B MLP (``inter=13824``), B in {64, 256}.  Three required
schemes plus two references:

  1. ``native_chain``     -- ``x.float() -> F.gelu -> mul -> .to(bf16)``
     (kickoff §3 native chain: fp32 cast + gelu + fp32 mul).
  2. ``two_step_bf16``    -- ``torch_npu.npu_gelu(x[:, :d]) * x[:, d:]`` (bf16).
  3. ``fi_gelu_fused``    -- the fused triton-ascend kernel with a preallocated
     ``out=`` (``fi_gelu.api.gelu_and_mul``).
  4. ``vllm_native_bf16`` -- reference: vLLM ``GeluAndMul.forward_native`` on bf16
     (``F.gelu(x[:, :d]) * x[:, d:]``, no fp32 cast).  This is what the eager
     bench actually timed as its ~130 us "native" anchor; the literal §3 fp32
     chain (scheme 1) costs more, so it is kept as a separate reference.
  5. ``copy_floor``       -- ``x.clone()`` of the full ``[B, 2*inter]`` input.

Both modes are timed with ``torch.npu.Event``:

* eager: warmup>=10, iters>=100, min-of-3 (identical to the eager bench);
* graph: capture once (fresh ``torch.npu.NPUGraph`` handle per scheme, never
  reused), warmup, then >=200 replays per round x 3 rounds; reports both best
  and median per-replay us.  The capture call itself is timed separately as a
  one-off wiring cost and is NOT part of the replay number.

Correctness (mandatory for every scheme): a capture records kernels but -- as
verified on this stack -- does *not* execute them, so the captured output buffer
holds garbage until the first replay.  After replay we snapshot the graph output
and compare it element-wise against a freshly-eager output of the same scheme,
using the ``tests/test_npu_w31_gelu.py`` bf16 mixed-tolerance口径
(rtol=atol=2**-6, pass if ``|a-g| <= atol + rtol*|g|``); ``matched_ratio`` should
be 1.0.  For the triton scheme we additionally verify that (a) repeated replay
is state-stable and (b) an eager triton call after capture still matches, plus a
bonus check against the fp32-erf golden.

Run (NPU card 7, from a non-/vllm-workspace CWD):

    cd /tmp && export ASCEND_RT_VISIBLE_DEVICES=7
    python3 /vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_gelu_graph.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend_split_batch.fi_gelu import api, reference

INTER = 13824
BATCHES = (64, 256)

WARMUP = 10
ITERS = 100
ROUNDS = 3

GRAPH_WARMUP = 10
GRAPH_ITERS = 200
GRAPH_ROUNDS = 3

# ops-precision-standard, float_compute.md §3, BFLOAT16 row (same as the tests).
BF16_RTOL = 2.0**-6
BF16_ATOL = 2.0**-6

OUT = Path(__file__).resolve().parent / "results" / "bench_w31_gelu_graph.json"


def bench_event(
    fn: Callable[[], object],
    iters: int = ITERS,
    warmup: int = WARMUP,
    rounds: int = ROUNDS,
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


def matched(actual: torch.Tensor, golden: torch.Tensor) -> dict:
    """bf16 mixed-tolerance comparison, verbatim口径 of the W3.1 NPU tests."""
    a = actual.float().cpu().numpy().astype(np.float64)
    g = golden.float().cpu().numpy().astype(np.float64)
    if a.shape != g.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {g.shape}")
    abs_err = np.abs(a - g)
    passed = abs_err <= BF16_ATOL + BF16_RTOL * np.abs(g)
    total = a.size
    return {
        "matched_ratio": 1.0 if total == 0 else float(passed.sum()) / total,
        "max_abs_error": 0.0 if total == 0 else float(abs_err.max()),
    }


def native_chain(x: torch.Tensor) -> torch.Tensor:
    """kickoff §3 native chain: fp32 cast + F.gelu + fp32 mul -> bf16."""
    xf = x.float()
    return (F.gelu(xf[:, :INTER]) * xf[:, INTER:]).to(torch.bfloat16)


def two_step_bf16(x: torch.Tensor) -> torch.Tensor:
    """Cheapest ready-made chain: bf16 ``npu_gelu`` + bf16 ``mul``."""
    return torch_npu.npu_gelu(x[:, :INTER]) * x[:, INTER:]


def vllm_native_bf16(x: torch.Tensor) -> torch.Tensor:
    """Reference: vLLM ``GeluAndMul.forward_native`` on bf16 (no fp32 cast)."""
    return F.gelu(x[:, :INTER]) * x[:, INTER:]


def graph_bench(
    fn: Callable[[], torch.Tensor],
    *,
    extra_triton: bool = False,
    x: torch.Tensor | None = None,
) -> dict:
    """Capture ``fn`` once into a fresh NPUGraph and time replay.

    Returns the replay timing (best/median per-call us, over ``GRAPH_ROUNDS``
    rounds of ``GRAPH_ITERS`` replays), the one-off capture cost, and the
    replay-vs-eager correctness numbers.
    """
    # No pending host control flow must be outstanding when capture starts.
    for _ in range(GRAPH_WARMUP):
        fn()
    torch.npu.synchronize()

    handle = torch.npu.NPUGraph()
    t0 = time.perf_counter()
    with torch.npu.graph(handle):
        graph_out = fn()
    torch.npu.synchronize()
    capture_us = (time.perf_counter() - t0) * 1e6

    round_us: list[float] = []
    for _ in range(GRAPH_ROUNDS):
        for _ in range(GRAPH_WARMUP):
            handle.replay()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(GRAPH_ITERS):
            handle.replay()
        end.record()
        torch.npu.synchronize()
        round_us.append(start.elapsed_time(end) * 1e3 / GRAPH_ITERS)

    # Capture records but does not execute, so the output is only meaningful
    # after the first replay.  Snapshot the replay result BEFORE running the
    # eager reference (the fused scheme shares its preallocated ``out`` buffer).
    for _ in range(5):
        handle.replay()
    torch.npu.synchronize()
    graph_snapshot = graph_out.clone()
    eager_ref = fn().clone()
    torch.npu.synchronize()

    res: dict = {
        "us_best": min(round_us),
        "us_median": statistics.median(round_us),
        "round_us": [round(u, 1) for u in round_us],
        "capture_us": capture_us,
        **matched(graph_snapshot, eager_ref),
    }

    if extra_triton:
        # (a) repeated replay must not accumulate/串 state.
        handle.replay()
        torch.npu.synchronize()
        snap_a = graph_out.clone()
        for _ in range(20):
            handle.replay()
        torch.npu.synchronize()
        snap_b = graph_out.clone()
        res["repeat_replay_stable"] = bool(torch.equal(snap_a, snap_b))
        # (b) an eager triton call after capture still produces the right result.
        after = fn().clone()
        torch.npu.synchronize()
        res["eager_after_capture"] = matched(after, eager_ref)
        # bonus: fused output vs the fp32-erf golden (bf16-rounded).
        if x is not None:
            golden = reference.gelu_and_mul_reference(x.cpu(), dtype=torch.bfloat16)
            res["vs_bf16_golden"] = matched(graph_snapshot, golden)
    return res


def main() -> int:
    torch.npu.set_device(0)
    try:
        dev = torch.npu.get_device_name(0)
    except Exception:  # noqa: BLE001 - cosmetic only
        dev = "npu:0"
    print(
        f"device: {dev}  inter={INTER}  "
        f"eager(iters={ITERS},warmup={WARMUP},rounds={ROUNDS})  "
        f"graph(replay>={GRAPH_ITERS},rounds={GRAPH_ROUNDS})"
    )

    rows: list[dict] = []
    capture_rows: list[dict] = []
    triton_record: dict = {
        "op": "triton_ingraph",
        "mode": "meta",
        "captured": None,
        "error_type": None,
        "error": None,
        "traceback": None,
        "batches": [],
    }
    triton_failed = False

    header = (
        f"{'B':>5} {'op':<38} {'mode':<6} {'us':>8} {'med':>8} "
        f"{'ratio':>7} {'maxabs':>9}"
    )
    print(header)
    print("-" * len(header))

    for batch in BATCHES:
        g = torch.Generator().manual_seed(0)
        x = (
            torch.randn(batch, 2 * INTER, generator=g, dtype=torch.float32)
            .bfloat16()
            .npu()
        )
        out_fi = torch.empty(batch, INTER, dtype=torch.bfloat16).npu()

        schemes: list[tuple[str, Callable[[], torch.Tensor], bool]] = [
            ("native_chain(fp32_cast+gelu+mul)", lambda x=x: native_chain(x), False),
            ("two_step_bf16(npu_gelu+mul)", lambda x=x: two_step_bf16(x), False),
            (
                "fi_gelu_fused(triton,out=prealloc)",
                lambda x=x, o=out_fi: api.gelu_and_mul(x, out=o),
                True,
            ),
            (
                "vllm_native_bf16(F.gelu*mul,ref)",
                lambda x=x: vllm_native_bf16(x),
                False,
            ),
            ("copy_floor(x.clone())", lambda x=x: x.clone(), False),
        ]

        # -- eager baseline ---------------------------------------------------
        for name, fn, _is_fi in schemes:
            eager_us = bench_event(fn)
            rows.append(
                {
                    "batch": batch,
                    "inter": INTER,
                    "op": name,
                    "mode": "eager",
                    "us": round(eager_us, 1),
                    "us_median": None,
                    "vs_native": None,
                    "vs_copy_floor": None,
                    "matched_ratio": None,
                    "max_abs_error": None,
                }
            )

        # -- graph capture / replay ------------------------------------------
        for name, fn, is_fi in schemes:
            if is_fi and triton_failed:
                continue
            try:
                res = graph_bench(fn, extra_triton=is_fi, x=x if is_fi else None)
            except Exception as exc:  # noqa: BLE001 - the traceback IS the result
                tb = traceback.format_exc()
                print(
                    f"\n[graph] {name} B={batch} CAPTURE FAILED:\n{tb}",
                    file=sys.stderr,
                )
                if is_fi:
                    triton_record["captured"] = False
                    triton_record["error_type"] = type(exc).__name__
                    triton_record["error"] = str(exc)
                    triton_record["traceback"] = tb
                    triton_record["batch"] = batch
                    triton_failed = True
                continue

            rows.append(
                {
                    "batch": batch,
                    "inter": INTER,
                    "op": name,
                    "mode": "graph",
                    "us": round(res["us_best"], 1),
                    "us_median": round(res["us_median"], 1),
                    "vs_native": None,
                    "vs_copy_floor": None,
                    "matched_ratio": res["matched_ratio"],
                    "max_abs_error": res["max_abs_error"],
                }
            )
            capture_rows.append(
                {
                    "batch": batch,
                    "inter": INTER,
                    "op": name,
                    "mode": "capture",
                    "capture_us": round(res["capture_us"], 1),
                }
            )
            if is_fi:
                triton_record["captured"] = True
                triton_record["batches"].append(
                    {
                        "batch": batch,
                        "capture_us": round(res["capture_us"], 1),
                        "us_best": round(res["us_best"], 1),
                        "us_median": round(res["us_median"], 1),
                        "matched_ratio": res["matched_ratio"],
                        "max_abs_error": res["max_abs_error"],
                        "repeat_replay_stable": res["repeat_replay_stable"],
                        "eager_after_capture": res["eager_after_capture"],
                        "vs_bf16_golden": res.get("vs_bf16_golden"),
                    }
                )

        print()

    # -- ratios (per batch, per mode) -----------------------------------------
    for batch in (*BATCHES, None):
        for mode in ("eager", "graph"):
            grp = [r for r in rows if r["batch"] == batch and r["mode"] == mode]
            if not grp:
                continue
            native = next(
                (r["us"] for r in grp if r["op"].startswith("native_chain")), None
            )
            floor = next(
                (r["us"] for r in grp if r["op"].startswith("copy_floor")), None
            )
            for r in grp:
                if native:
                    r["vs_native"] = round(r["us"] / native, 3)
                if floor:
                    r["vs_copy_floor"] = round(r["us"] / floor, 3)

    # -- report ---------------------------------------------------------------
    for r in rows:
        med = "" if r["us_median"] is None else f"{r['us_median']:.1f}"
        ratio = "" if r["matched_ratio"] is None else f"{r['matched_ratio']:.3f}"
        maxabs = "" if r["max_abs_error"] is None else f"{r['max_abs_error']:.2e}"
        print(
            f"{r['batch']:>5} {r['op']:<38} {r['mode']:<6} {r['us']:>8.1f} "
            f"{med:>8} {ratio:>7} {maxabs:>9}"
        )

    print("\ncapture one-off cost (excluded from replay numbers):")
    for c in capture_rows:
        print(f"  B={c['batch']:<4} {c['op']:<38} {c['capture_us']:>8.1f} us")

    print("\neager -> graph (best replay us):")
    for batch in BATCHES:
        print(f"  B={batch}:")
        for op_prefix in (
            "native_chain",
            "two_step_bf16",
            "fi_gelu_fused",
            "vllm_native_bf16",
            "copy_floor",
        ):
            e = next(
                (
                    r
                    for r in rows
                    if r["batch"] == batch
                    and r["mode"] == "eager"
                    and r["op"].startswith(op_prefix)
                ),
                None,
            )
            gg = next(
                (
                    r
                    for r in rows
                    if r["batch"] == batch
                    and r["mode"] == "graph"
                    and r["op"].startswith(op_prefix)
                ),
                None,
            )
            if e and gg:
                print(
                    f"    {op_prefix:<16} eager {e['us']:>6.1f}  "
                    f"graph {gg['us']:>6.1f}  ratio {gg['matched_ratio']}"
                )

    if triton_record["captured"]:
        print(
            f"\ntriton in-graph: CAPTURED ok; "
            f"{len(triton_record['batches'])} batch(es), "
            f"matched_ratio all 1.0="
            f"{all(b['matched_ratio'] == 1.0 for b in triton_record['batches'])}, "
            f"repeat_replay_stable="
            f"{[b['repeat_replay_stable'] for b in triton_record['batches']]}"
        )
    elif triton_record["captured"] is False:
        print(
            f"\ntriton in-graph: CAPTURE FAILED "
            f"({triton_record['error_type']}): {triton_record['error']}",
            file=sys.stderr,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows + capture_rows + [triton_record], indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
