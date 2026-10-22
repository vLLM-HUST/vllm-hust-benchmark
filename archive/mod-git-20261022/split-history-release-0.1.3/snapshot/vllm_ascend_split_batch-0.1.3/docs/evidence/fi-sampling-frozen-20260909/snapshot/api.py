"""Host wrappers: flashinfer-signature-compatible sampling APIs on triton-ascend.

Ported from flashinfer main@3a4e7052:
  - sampling.py:874  ``sampling_from_probs``
  - sampling.py:976  ``top_p_sampling_from_probs``      (convenience, same kernel)
  - sampling.py:1096 ``top_k_sampling_from_probs``      (convenience, same kernel)
  - sampling.py:1579 ``top_k_top_p_sampling_from_probs`` (filter_apply_order="joint")

Signature semantics kept: probs fp32 [B, V], ``indices`` row mapping (output
dtype follows indices when given, else int32), ``deterministic`` accepted for
parity (our scan is a fixed-shape block cumsum — deterministic by
construction for a fixed kernel config), ``generator``/``seed`` RNG control,
``check_nan``, ``return_valid``.

Deliberate deviations (documented in README):
  - ``filter_apply_order="top_k_first"`` raises NotImplementedError (W2 scope
    is the joint kernel; top_k_first needs top_k_renorm_probs, i.e. W3 renorm).
    FI defaults to ``"top_k_first"`` (sampling.py:1584); this package defaults
    to ``"joint"``, so the FI default is not usable here.
  - ``top_k=None`` is accepted by the joint entrypoint (= no count constraint,
    FI ``top_p_sampling_from_probs`` semantics); the test matrix uses k=INF.
  - RNG stream semantics: ``generator`` is used **only as a seed source**
    (``generator.initial_seed()``) and is never advanced — repeated calls with
    the same generator and input return bitwise identical tokens. FI instead
    advances the generator on every call (``get_seed_and_offset``,
    sampling.py:47-63: ``offset += (increment + 3) // 4 * 4`` with
    ``increment = batch_size * 32``), so each FI call consumes a fresh stream.
    Callers needing cross-call randomness must supply a new ``seed``/``offset``
    per call (or their own ``rng_mode="host"`` uniforms). The per-call stream
    width also differs: FI reserves ``batch * 32`` offsets per call while this
    kernel consumes ``B * MAX_ROUNDS = B * 64`` (philox offset =
    ``offset + row * MAX_ROUNDS + round``), so reusing FI's increment verbatim
    would make consecutive calls overlap.
  - ``seed``/``offset`` are ints (per-row philox offset = offset + row *
    MAX_ROUNDS + round); per-row seed/offset *tensors* (CUDA-graph parity) are
    out of W2 scope.
  - ``deterministic=False`` does not switch algorithms (only one scan
    implementation exists); FI's two scan variants differ only in fp
    associativity.
  - per-row ``top_k``/``top_p`` are both indexed by the probs row
    (``row_idx``). Upstream is inconsistent: the joint kernel indexes top_k by
    row too (``top_k_arr[row_idx]``, sampling.cuh:1216 — same as this
    package), but the top-k-only kernel indexes it by output slot
    (``top_k_arr[bx]``, sampling.cuh:862); only that path differs from us.
    Independently of indexing, the *tensor length* semantics differ: FI
    documents the tensors as ``(batch_size,)`` = number of **outputs**, while
    this package requires ``numel() == unique_rows`` (probs rows; see
    ``_prepare_top_k``/``_prepare_top_p``) — with ``indices`` the two coincide
    only when ``unique_rows == out_batch``.
"""

from __future__ import annotations

import torch

from fi_sampling.kernels import KSCALE, MAX_ROUNDS, _sampling_from_probs_kernel
from fi_sampling.npu_env import init_device_properties_triton

__all__ = [
    "sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_sampling_from_probs",
]

_ScalarOrTensor = int | torch.Tensor | None


def _default_seed() -> int:
    """Fresh seed from the CPU default RNG (no device sync)."""
    return int(torch.randint(0, 2**31 - 1, (1,)).item())


def _resolve_seed(
    generator: torch.Generator | None, seed: int | None
) -> int:
    """Resolve the kernel seed; ``generator`` is a seed source only.

    The generator is *not* advanced: every call resolves to the same
    ``generator.initial_seed()``, so repeated calls with the same generator and
    the same input produce identical tokens (offset stays 0 unless the caller
    passes ``offset=``). This deviates from flashinfer, where each call advances
    the generator (``get_seed_and_offset``, sampling.py:47-63, ``offset +=
    (batch * 32 + 3) // 4 * 4``). Callers that need fresh randomness across
    calls must pass a new ``seed``/``offset`` themselves (or use
    ``rng_mode="host"`` with their own uniforms). See README "与 flashinfer 的
    刻意偏差" #6.
    """
    if seed is not None:
        return int(seed) & 0x7FFFFFFF
    if generator is not None:
        return int(generator.initial_seed()) & 0x7FFFFFFF
    return _default_seed() & 0x7FFFFFFF


def _prepare_probs(probs: torch.Tensor, check_nan: bool) -> torch.Tensor:
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2D [B, V], got shape {tuple(probs.shape)}")
    if check_nan and bool(torch.isnan(probs).any().item()):
        raise ValueError("Input probs contains NaN.")
    if not probs.is_contiguous():
        probs = probs.contiguous()
    if probs.dtype != torch.float32:
        probs = probs.float()
    return probs


def _prepare_top_k(top_k: _ScalarOrTensor, batch_size: int, device) -> tuple[torch.Tensor | None, int | None, bool]:
    """Returns (per_row_tensor_or_None, scalar_or_None, is_scalar)."""
    if top_k is None:
        return None, None, False
    if isinstance(top_k, torch.Tensor):
        t = top_k.to(device=device, dtype=torch.int32).reshape(-1).contiguous()
        if t.numel() == 1:
            return None, int(t.item()), True
        if t.numel() != batch_size:
            raise ValueError(f"top_k tensor must have batch_size={batch_size} elements")
        if bool((t < 1).any().item()):
            raise ValueError("top_k must be >= 1")
        return t, None, False
    k = int(top_k)
    if k < 1:
        raise ValueError("top_k must be >= 1")
    return None, k, True


def _prepare_top_p(top_p: _ScalarOrTensor, batch_size: int, device) -> tuple[torch.Tensor | None, float | None, bool]:
    if top_p is None:
        return None, None, False
    if isinstance(top_p, torch.Tensor):
        t = top_p.to(device=device, dtype=torch.float32).reshape(-1).contiguous()
        if t.numel() == 1:
            return None, float(t.item()), True
        if t.numel() != batch_size:
            raise ValueError(f"top_p tensor must have batch_size={batch_size} elements")
        if bool(((t <= 0) | (t > 1.0)).any().item()):
            raise ValueError("top_p values must lie in (0, 1]")
        return t, None, False
    p = float(top_p)
    if not (0.0 < p <= 1.0):
        raise ValueError("top_p must lie in (0, 1]")
    return None, p, True


def _launch(
    probs: torch.Tensor,
    has_top_k: bool,
    has_top_p: bool,
    k_tensor: torch.Tensor | None,
    k_scalar: int | None,
    p_tensor: torch.Tensor | None,
    p_scalar: float | None,
    indices: torch.Tensor | None,
    use_tl_rand: bool,
    seed: int,
    offset: int,
    return_valid: bool,
    block_v: int,
    num_warps: int,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    B, V = probs.shape
    out_batch = int(indices.numel()) if indices is not None else B
    if out_batch == 0:
        out = torch.empty(0, dtype=torch.int32, device=probs.device)
        return (out, torch.empty(0, dtype=torch.bool, device=probs.device)) if return_valid else out
    if B * V >= 2**31:
        raise ValueError("B * V must be < 2**31 (int32 addressing)")

    dev = probs.device
    out = torch.empty(out_batch, dtype=torch.int32, device=dev)
    valid = torch.empty(out_batch, dtype=torch.int8, device=dev)

    k_ptr = k_tensor if k_tensor is not None else probs  # dummy ptr when scalar/absent
    p_ptr = p_tensor if p_tensor is not None else probs
    idx_ptr = indices if indices is not None else probs

    us: torch.Tensor | None = None
    if not use_tl_rand:
        # NOTE: torch CPU generators truncate the seed to 32 bits, so a naive
        # `seed * 2**31 + offset` mix collapses to 2 distinct states. Mix into
        # the low bits instead. us is indexed by probs-row (B = unique_rows),
        # not by output slot — same probs row -> same uniform stream, matching
        # the per-row philox offsets of the default path.
        mixed = (seed * 1000003 + offset * 97 + 1) % (2**31 - 1)
        g = torch.Generator(device="cpu")
        g.manual_seed(mixed)
        us = torch.rand((B, MAX_ROUNDS), generator=g, dtype=torch.float32).to(dev)
    us_ptr = us if us is not None else probs

    init_device_properties_triton()
    _sampling_from_probs_kernel[(out_batch,)](
        probs,
        out,
        valid,
        idx_ptr,
        k_ptr,
        p_ptr,
        us_ptr,
        seed,
        offset,
        k_scalar if k_scalar is not None else 0,
        p_scalar if p_scalar is not None else 0.0,
        V,
        HAS_TOP_K=has_top_k,
        HAS_TOP_P=has_top_p,
        K_IS_SCALAR=(k_tensor is None),
        P_IS_SCALAR=(p_tensor is None),
        HAS_INDICES=(indices is not None),
        USE_TL_RAND=use_tl_rand,
        MAX_ROUNDS_C=MAX_ROUNDS,
        KSCALE_C=KSCALE,
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
    out = out.to(indices.dtype) if indices is not None else out
    if return_valid:
        return out, valid.bool()
    return out


def _check_indices(indices: torch.Tensor | None, unique_rows: int) -> torch.Tensor | None:
    """FI semantics: indices[i]=j -> output i samples from probs[j].

    indices length = output batch; probs rows = unique_batch_size.
    """
    if indices is None:
        return None
    if indices.dim() != 1:
        raise ValueError("indices must be 1D")
    if indices.numel() == 0:
        raise ValueError("indices must be non-empty")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32 or int64")
    if bool((indices < 0).any().item()) or bool((indices >= unique_rows).any().item()):
        raise ValueError(f"indices values must lie in [0, {unique_rows})")
    if not indices.is_contiguous():
        indices = indices.contiguous()
    return indices


def sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    return_valid: bool = False,
    *,
    rng_mode: str = "philox",
    block_v: int = 1024,
    num_warps: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """flashinfer ``sampling_from_probs`` (sampling.py:874) on triton-ascend.

    Samples token ids from row-normalized (or unnormalized, non-negative)
    probabilities via a single-pass in-row CDF prefix sum; 1 uniform per row.
    """
    probs = _prepare_probs(probs, check_nan)
    indices = _check_indices(indices, probs.shape[0])
    seed_r = _resolve_seed(generator, seed)
    off_r = int(offset) if offset is not None else 0
    return _launch(
        probs,
        has_top_k=False,
        has_top_p=False,
        k_tensor=None,
        k_scalar=None,
        p_tensor=None,
        p_scalar=None,
        indices=indices,
        use_tl_rand=(rng_mode == "philox"),
        seed=seed_r,
        offset=off_r,
        return_valid=return_valid,
        block_v=block_v,
        num_warps=num_warps,
    )


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: _ScalarOrTensor,
    top_p: _ScalarOrTensor,
    indices: torch.Tensor | None = None,
    filter_apply_order: str = "joint",
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    return_valid: bool = False,
    *,
    rng_mode: str = "philox",
    block_v: int = 1024,
    num_warps: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """flashinfer ``top_k_top_p_sampling_from_probs`` (sampling.py:1579), joint order.

    Pivot-bisection rejection sampling with in-row joint truncation:
    kept(i) = (count(x>p_i) < k) & (sum(x>p_i) < p); ties atomic (strict >).
    ``top_k=None`` (= k=INF) selects FI top-p-only semantics; ``top_p=None``
    selects FI top-k-only semantics.
    """
    if filter_apply_order != "joint":
        raise NotImplementedError(
            "filter_apply_order='top_k_first' requires top_k_renorm_probs (W3 renorm scope); "
            "W2 ships the 'joint' single-kernel path only."
        )
    probs = _prepare_probs(probs, check_nan)
    indices = _check_indices(indices, probs.shape[0])
    k_tensor, k_scalar, _ = _prepare_top_k(top_k, probs.shape[0], probs.device)
    p_tensor, p_scalar, _ = _prepare_top_p(top_p, probs.shape[0], probs.device)
    if k_tensor is None and k_scalar is None and p_tensor is None and p_scalar is None:
        raise ValueError("at least one of top_k / top_p must be provided")
    seed_r = _resolve_seed(generator, seed)
    off_r = int(offset) if offset is not None else 0
    return _launch(
        probs,
        has_top_k=(k_tensor is not None or k_scalar is not None),
        has_top_p=(p_tensor is not None or p_scalar is not None),
        k_tensor=k_tensor,
        k_scalar=k_scalar,
        p_tensor=p_tensor,
        p_scalar=p_scalar,
        indices=indices,
        use_tl_rand=(rng_mode == "philox"),
        seed=seed_r,
        offset=off_r,
        return_valid=return_valid,
        block_v=block_v,
        num_warps=num_warps,
    )


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: _ScalarOrTensor,
    indices: torch.Tensor | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    return_valid: bool = False,
    *,
    rng_mode: str = "philox",
    block_v: int = 1024,
    num_warps: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """flashinfer ``top_p_sampling_from_probs`` (sampling.py:976) — thin alias."""
    return top_k_top_p_sampling_from_probs(
        probs,
        top_k=None,
        top_p=top_p,
        indices=indices,
        filter_apply_order="joint",
        deterministic=deterministic,
        generator=generator,
        check_nan=check_nan,
        seed=seed,
        offset=offset,
        return_valid=return_valid,
        rng_mode=rng_mode,
        block_v=block_v,
        num_warps=num_warps,
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: _ScalarOrTensor,
    indices: torch.Tensor | None = None,
    deterministic: bool = True,
    generator: torch.Generator | None = None,
    check_nan: bool = False,
    seed: int | None = None,
    offset: int | None = None,
    return_valid: bool = False,
    *,
    rng_mode: str = "philox",
    block_v: int = 1024,
    num_warps: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """flashinfer ``top_k_sampling_from_probs`` (sampling.py:1096) — thin alias."""
    return top_k_top_p_sampling_from_probs(
        probs,
        top_k=top_k,
        top_p=None,
        indices=indices,
        filter_apply_order="joint",
        deterministic=deterministic,
        generator=generator,
        check_nan=check_nan,
        seed=seed,
        offset=offset,
        return_valid=return_valid,
        rng_mode=rng_mode,
        block_v=block_v,
        num_warps=num_warps,
    )
