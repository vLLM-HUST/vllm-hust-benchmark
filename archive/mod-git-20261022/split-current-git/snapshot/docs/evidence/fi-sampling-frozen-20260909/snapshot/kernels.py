"""Triton-ascend sampling kernels (flashinfer semantics, main@3a4e7052).

Algorithm reference (upstream CUDA, include/flashinfer/sampling.cuh):
  - SamplingFromProbKernel         :785-844   (single-pass CDF inverse)
  - TopKSamplingFromProbKernel     :849-977   (pivot bisection, accept count<k)
  - TopPSamplingFromProbKernel     :982-1103  (pivot bisection, accept tail-sum<p)
  - TopKTopPSamplingFromProbKernel :1202-1330 (joint: count<k AND tail-sum<p)
  - DeviceSamplingFromProb         :579-654   (in-row CDF prefix-sum sampling core)

One program per row. Data-dependent pivot-bisection rejection loop; each round
consumes exactly one uniform (``u = rand * q``), scans the row with the strict
predicate ``x > low`` (ties kept/dropped atomically), selects the first
CDF-crossing index, verifies the candidate against ``(count < k) AND
(tail-sum < p)`` and bisects ``[low, high]`` on rejection — i.e. the sampled
output distribution equals the kept-renormalized distribution
``P(i) = p_i * 1[cond(i)] / Z`` with ``cond(i) = (#(x>p_i) < k) & (sum(x>p_i) < p)``.

Numerical guards (q6 d.4 "必抄清单"; provenance per item):
  - uniform open at 1: ``u = rand(...) * (1 - FLT_EPS)``. NOTE on provenance
    (2026-09-09 review errata): FI's ``kSCALE = 1 - FLT_EPSILON`` lives in
    ``GenerateGumbelNoise`` (sampling.cuh:685), i.e. the logits/Gumbel path,
    NOT the probs path. The probs kernels draw a bare ``curand_uniform`` in
    (0, 1] and handle the u==1 corner with the last-valid fallback
    (sampling.cuh:810/:826-839). Our ``* KSCALE`` is therefore a self-added
    defensive guard, not a copied constant; the domain difference
    (tl.rand ∈ [0,1) vs curand ∈ (0,1]) is a measure-zero difference documented
    in README "与 flashinfer 的刻意偏差" #5.
  - pivot bisection in plain IEEE fp32 mul/add (sampling.cuh:1045-1046); the
    CUDA-side fast-math FTZ hazard (#769/#774) does not exist on the
    bishengir fp32 path (verified by the subnormal probe in the validation suite)
  - degenerate rows (no token under the predicate) fall back to the last valid
    index; fully empty rows output 0 with valid=false (sampling.cuh:826-839)
  - rejection loop capped at MAX_ROUNDS (FI relies on ``low < high``
    saturation; the cap is a defensive equivalent and is not expected to bind)

Constraints: fp32 probs, row-major contiguous, B * V < 2**31.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Self-added defensive guard: keep u strictly below 1.0. NOT copied from FI —
# FI's kSCALE = 1 - FLT_EPSILON belongs to the Gumbel/logits path
# (sampling.cuh:685); the probs path uses a bare curand_uniform in (0, 1] and
# relies on the last-valid fallback for the u==1 corner. Domain difference
# (tl.rand ∈ [0,1) vs curand ∈ (0,1]) is measure-zero; see README deviation #5.
KSCALE: float = 1.0 - float(torch.finfo(torch.float32).eps)

MAX_ROUNDS: int = 64


@triton.jit(do_not_specialize=["seed", "offset"])
def _sampling_from_probs_kernel(
    probs_ptr,  # [B, d] fp32
    out_ptr,  # [B] int32
    valid_ptr,  # [B] int8 (0/1)
    indices_ptr,  # [B] int32/int64 row mapping, or dummy
    top_k_ptr,  # [B] int32 per-row k, or dummy
    top_p_ptr,  # [B] fp32 per-row p, or dummy
    us_ptr,  # [B, MAX_ROUNDS] fp32 pre-drawn uniforms, or dummy
    seed,  # int (philox seed for the tl.rand path)
    offset,  # int (philox offset base for the tl.rand path)
    k_val,  # int32 scalar k (when K_IS_SCALAR)
    p_val,  # fp32 scalar p (when P_IS_SCALAR)
    d,  # vocab size (runtime)
    HAS_TOP_K: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
    K_IS_SCALAR: tl.constexpr,
    P_IS_SCALAR: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    USE_TL_RAND: tl.constexpr,
    MAX_ROUNDS_C: tl.constexpr,
    KSCALE_C: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    if HAS_INDICES:
        row = tl.load(indices_ptr + pid).to(tl.int32)
    else:
        row = pid

    # ---- per-row truncation parameters (FI: top_k_arr/top_p_arr or scalars) ----
    if HAS_TOP_K:
        if K_IS_SCALAR:
            k_eff = k_val
        else:
            k_eff = tl.load(top_k_ptr + row).to(tl.int32)
    if HAS_TOP_P:
        if P_IS_SCALAR:
            p_eff = p_val
        else:
            p_eff = tl.load(top_p_ptr + row)

    num_chunks = tl.cdiv(d, BLOCK_V)

    # ---- pivot-bisection rejection loop (sampling.cuh:1009-1097) ----
    low = 0.0
    high = 1.0
    q = 1.0
    sampled = d  # sentinel: not yet resolved
    valid_row = 1
    done = 0
    rnd = 0
    while (done == 0) & (rnd < MAX_ROUNDS_C):
        # one uniform per round (FI: u = curand_uniform * q, :1013)
        if USE_TL_RAND:
            u = tl.rand(seed, offset + row * MAX_ROUNDS_C + rnd) * KSCALE_C
        else:
            u = tl.load(us_ptr + row * MAX_ROUNDS_C + rnd) * KSCALE_C
        u = u * q

        # ---- in-row CDF sampling under predicate x > low (:579-654) ----
        aggregate = 0.0
        cand = d
        crossed = 0
        last_valid = -1
        ch = 0
        while (ch < num_chunks) & (crossed == 0):
            offs = ch * BLOCK_V + tl.arange(0, BLOCK_V)
            mask = offs < d
            x = tl.load(probs_ptr + row * d + offs, mask=mask, other=0.0)
            keep = (x > low) & mask  # strict > : tie classes atomic (q6 c.2)
            xc = tl.where(keep, x, 0.0)
            local = tl.sum(xc, axis=0)
            # FI last-valid: pred(x) AND in-bounds (sampling.cuh:590, :636-651)
            lvi = tl.max(tl.where(keep, offs, -1), axis=0)
            last_valid = tl.maximum(last_valid, lvi)
            if (aggregate + local) > u:
                cdf = tl.cumsum(xc, axis=0)
                flag = keep & ((cdf + aggregate) > u)
                cand = tl.min(tl.where(flag, offs, d), axis=0)
                crossed = 1
            aggregate += local
            ch += 1

        # degenerate guard: never crossed -> last valid index (:826-839)
        if crossed == 0:
            cand = last_valid

        if cand < 0:
            # fully empty row: output 0 with valid=false (FI :831-836)
            sampled = 0
            valid_row = 0
            done = 1
        else:
            sampled = cand
            if (HAS_TOP_K) or (HAS_TOP_P):
                # pivot bisection (:1045-1046): IEEE fp32 mul/add
                pivot_0 = tl.load(probs_ptr + row * d + sampled)
                pivot_1 = (pivot_0 + high) * 0.5

                s0 = 0.0
                c0 = 0
                s1 = 0.0
                c1 = 0
                ch2 = 0
                while ch2 < num_chunks:
                    offs = ch2 * BLOCK_V + tl.arange(0, BLOCK_V)
                    mask = offs < d
                    x = tl.load(probs_ptr + row * d + offs, mask=mask, other=0.0)
                    gt0 = (x > pivot_0) & mask
                    gt1 = (x > pivot_1) & mask
                    c0 += tl.sum(gt0.to(tl.int32), axis=0)
                    s0 += tl.sum(tl.where(gt0, x, 0.0), axis=0)
                    c1 += tl.sum(gt1.to(tl.int32), axis=0)
                    s1 += tl.sum(tl.where(gt1, x, 0.0), axis=0)
                    ch2 += 1

                # joint accept: (count < k) AND (tail-sum < p) (:1310-1323)
                if HAS_TOP_K and HAS_TOP_P:
                    acc0 = (c0 < k_eff) & (s0 < p_eff)
                    acc1 = (c1 < k_eff) & (s1 < p_eff)
                elif HAS_TOP_K:
                    acc0 = c0 < k_eff
                    acc1 = c1 < k_eff
                else:
                    acc0 = s0 < p_eff
                    acc1 = s1 < p_eff

                if acc0:
                    done = 1  # case 1: pivot_0 accepted
                elif acc1:
                    # case 2: pivot_0 rejected, pivot_1 accepted
                    low = pivot_0
                    high = pivot_1
                    q = s0
                else:
                    # case 3: both rejected
                    low = pivot_1
                    q = s1
            else:
                done = 1  # no truncation: first CDF sample is accepted
        rnd += 1

    tl.store(out_ptr + pid, sampled.to(tl.int32))
    tl.store(valid_ptr + pid, valid_row.to(tl.int8))
