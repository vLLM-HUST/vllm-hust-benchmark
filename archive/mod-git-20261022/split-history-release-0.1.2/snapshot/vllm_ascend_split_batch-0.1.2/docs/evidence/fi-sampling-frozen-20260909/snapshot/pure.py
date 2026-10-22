"""Pure host-side logic for the flashinfer-semantics sampling port.

CPU-testable (torch, any backend with fp64 sort/searchsorted — tests run on
CPU). This module owns the *semantics* of truncation/renormalization:

  kept(i) = (count(x > p_i) < k)  AND  (sum(x > p_i) < p)      [strict >]
  P_out(i) = p_i * kept(i) / max(sum_j p_j * kept(j), 1e-8)

The strict-predicate tie rule is the load-bearing semantic (q6 c.2 / §c):
flashinfer's pivot condition is strictly ``>``, so equal probabilities are
kept or dropped *atomically* — e.g. top_k=1 on [0.4, 0.4, 0.2] keeps BOTH
0.4-tokens (50/50 sampling), unlike sort-based implementations that pick a
single element. ``renorm`` reproduces the FI guard ``rcp(max(sum, 1e-8))``
(topk.cuh:1963/1970, air_top_p.cuh:448).

Complexities: O(V log V) per row via ascending sort + searchsorted; no O(V^2)
materialization, so V=152064 (Qwen-14B vocab) is handled on CPU.
"""

from __future__ import annotations

import torch

__all__ = [
    "count_gt_and_sum_gt",
    "joint_kept_mask",
    "renorm",
    "target_distribution",
]

_ScalarOrTensor = int | float | torch.Tensor | None


def _as_per_row(value: _ScalarOrTensor, batch_size: int, device, dtype: torch.dtype):
    """Normalize scalar / [B] tensor / None into ([B] tensor or None)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        v = value.to(device=device, dtype=dtype).reshape(-1)
        if v.numel() == 1:
            v = v.expand(batch_size)
        if v.numel() != batch_size:
            raise ValueError(f"per-row parameter must have batch_size={batch_size} elements, got {v.numel()}")
        return v
    return torch.full((batch_size,), float(value), device=device, dtype=dtype)


def count_gt_and_sum_gt(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each element: (# strictly greater) and (sum of strictly greater).

    fp64 internals for exact ties + cancellation-free suffix sums.
    Returns (count_gt int64 [B, V], sum_gt fp64 [B, V]).
    """
    x = probs.to(torch.float64)
    sorted_asc = x.sort(dim=-1).values
    # #{j: x_j <= x_i} via searchsorted(right) on the ascending row — ties share it
    cnt_le = torch.searchsorted(sorted_asc, x, right=True)
    count_gt = x.shape[-1] - cnt_le
    prefix = sorted_asc.cumsum(dim=-1)  # inclusive prefix sums, fp64
    tot = prefix[:, -1:]
    prev_idx = (cnt_le - 1).clamp(min=0)
    prev = prefix.gather(1, prev_idx)
    sum_gt = tot - prev  # suffix beyond position cnt_le-1 == sum of elements > x_i
    return count_gt, sum_gt


def joint_kept_mask(
    probs: torch.Tensor,
    top_k: _ScalarOrTensor = None,
    top_p: _ScalarOrTensor = None,
) -> torch.Tensor:
    """FI joint-truncation kept mask (strict >, fp64 decisions).

    probs: fp32 [B, V]; top_k/top_p: scalar or [B] or None.
    """
    if probs.dim() != 2:
        raise ValueError("probs must be 2D [B, V]")
    count_gt, sum_gt = count_gt_and_sum_gt(probs)
    keep = torch.ones_like(count_gt, dtype=torch.bool)
    if top_k is not None:
        k = _as_per_row(top_k, probs.shape[0], probs.device, torch.float64)
        keep &= count_gt < k.unsqueeze(1).to(torch.int64)
    if top_p is not None:
        p = _as_per_row(top_p, probs.shape[0], probs.device, torch.float64)
        keep &= sum_gt < p.unsqueeze(1)
    return keep


def renorm(weights: torch.Tensor) -> torch.Tensor:
    """FI renorm guard: out = w * rcp(max(sum(w), 1e-8)) (fp32 parity)."""
    total = weights.sum(dim=-1, keepdim=True, dtype=torch.float32)
    return weights / torch.clamp(total, min=1e-8)


def target_distribution(
    probs: torch.Tensor,
    top_k: _ScalarOrTensor = None,
    top_p: _ScalarOrTensor = None,
) -> torch.Tensor:
    """Analytic output distribution of flashinfer's rejection sampling.

    P(i) = p_i * kept(i) / max(Z, 1e-8),  Z = sum_i p_i * kept(i).

    This is the distribution the pivot-bisection sampler draws from (proven
    invariant of the rejection loop; verified empirically by the CPU
    simulator-vs-analytic tests and the NPU TV suite).
    """
    keep = joint_kept_mask(probs, top_k, top_p)
    w = probs.to(torch.float64) * keep.to(torch.float64)
    z = w.sum(dim=-1, keepdim=True)
    return w / torch.clamp(z, min=1e-8)


def kept_support(target: torch.Tensor) -> torch.Tensor:
    """Indices with nonzero target probability (the support of the output)."""
    return target > 0
