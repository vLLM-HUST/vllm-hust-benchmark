# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic generator for the W1R GMM ``tuning_config`` / counts-dump patch.

Tier-2 host proposal, see ``/root/PURPOSE.md`` sec. 8: the lever only touches
two private call sites in the host file
``vllm_ascend/ops/fused_moe/routed_experts.py``, so the patch original lives in
this repository (``host-proposals/W1R-gmm-tuning-config/``) and the runtime tree
only receives a deployed copy.

Contract of this generator
--------------------------
* fail-closed on host drift: ``host_orig`` **and** (when ``--host`` is given)
  the live host file must match :data:`EXPECTED_ORIG_SHA256`, else exit 2;
* deterministic: two runs produce byte-identical ``patch.diff``;
* ``--check`` compares the **on-disk** ``patch.diff`` with a freshly computed
  one and exits 4 when the file is stale;
* every anchor must occur exactly once, else exit 3 (no silent no-op).
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
HOST_REL = "vllm_ascend/ops/fused_moe/routed_experts.py"
HOST_ORIG = HERE / "host_orig" / "routed_experts.py.orig"
PATCH_OUT = HERE / "patch.diff"

#: Exact host file this patch is written against (0.25.1rc2.dev125+hust.20260903.4).
#: Reproducible baseline:
#:   git -C <host repo> cat-file -p 74f0c0a27:vllm_ascend/ops/fused_moe/\
#:       routed_experts.py | sha256sum
EXPECTED_ORIG_SHA256 = (
    "b17e5a105d59fc280f244bdc4036b4cb41efd1d5a2ff1accca4b56620cc91ab7"
)

EXIT_DRIFT = 2
EXIT_ANCHOR = 3
EXIT_STALE = 4

INSERT_ANCHOR = (
    "from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz\n"
    "\n"
    "\n"
    "def get_moe_offload_runtime():"
)

GMM1_ANCHOR = """        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1 if isinstance(w1, list) else [w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )[0]
"""

GMM2_ANCHOR = """        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2 if isinstance(w2, list) else [w2],
            bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
        )[0]
"""

GMM1_PATCHED = """        gate_up_out = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1 if isinstance(w1, list) else [w1],
            bias=[w1_bias.to(dtype=torch.float32)] if w1_bias is not None else None,
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            **_hust_w1r_tuning_kwargs(),
        )[0]
        _hust_w1r_maybe_dump(
            hidden_states,
            w1,
            mlp_compute_input.group_list,
            mlp_compute_input.group_list_type,
            "apply_gmm1",
            mlp_compute_input.need_trans,
        )
"""

GMM2_PATCHED = """        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2 if isinstance(w2, list) else [w2],
            bias=[w2_bias.to(dtype=torch.float32)] if w2_bias is not None else None,
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            **_hust_w1r_tuning_kwargs(),
        )[0]
        _hust_w1r_maybe_dump(
            silu_out,
            w2,
            mlp_compute_input.group_list,
            mlp_compute_input.group_list_type,
            "apply_gmm2",
            mlp_compute_input.need_trans,
        )
"""

#: Default-off runtime block inserted right after the host imports.
HELPERS = '''# --- vllm-hust W1R tier-2 host proposal (default-off) BEGIN ---
# Carrier: vllm-ascend-split-batch-hust/host-proposals/W1R-gmm-tuning-config/
# VLLM_HUST_GMM_TUNING_CONFIG: ONE positive value from {1,2,8,32}; the GMM
#     tiling hint "expected tokens per expert" -> tuning_config=[value] on the
#     two GMM calls below. A comma list is rejected to keep the arm single.
# VLLM_HUST_GMM_COUNTS_DUMP=<path>: record group_list counts + shape and
#     provenance samples. ``{rank}`` in the path is substituted; without it a
#     TP=2 run appends ``.rank<RANK>`` so two ranks never share a file.
# Both unset -> no extra import, no IO, no extra kwargs, and no shape reads.
_HUST_W1R_TUNING_ENV = "VLLM_HUST_GMM_TUNING_CONFIG"
_HUST_W1R_DUMP_ENV = "VLLM_HUST_GMM_COUNTS_DUMP"
_HUST_W1R_ALLOWED = (1, 2, 8, 32)
_HUST_W1R_DUMP_SCHEMA = "vllm-hust-gmm-counts-dump.v2"
_HUST_W1R_SAMPLING = (
    "sample_source=eager means the host ran this Python call outside graph "
    "capture; inside capture the hook skips before any device read, so capture "
    "dummy rows are never recorded as counts; a FULL graph replay does not "
    "execute Python at all"
)
_HUST_W1R_STATE = {
    "tuning": None,
    "tuning_ready": False,
    "tuning_error": None,
    "dump_path": None,
    "dump_path_ready": False,
    "dump_meta_written": False,
    "dump_seq": 0,
    "dump_per_key": {},
    "error_reported": False,
}


def _hust_w1r_parse_tuning_config(raw):
    """Strict single-value parse: empty/None -> None, else ``[value]``."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if "," in text:
        raise ValueError(
            f"{_HUST_W1R_TUNING_ENV}={raw!r}: exactly one value is supported "
            f"(allowed {_HUST_W1R_ALLOWED}); a list would break the single arm"
        )
    if not text.isdigit():
        raise ValueError(
            f"{_HUST_W1R_TUNING_ENV}={raw!r}: expect one positive integer "
            f"(allowed {_HUST_W1R_ALLOWED})"
        )
    value = int(text)
    if value not in _HUST_W1R_ALLOWED:
        raise ValueError(
            f"{_HUST_W1R_TUNING_ENV}={raw!r}: allowed {_HUST_W1R_ALLOWED}"
        )
    return [value]


def _hust_w1r_tuning_kwargs():
    """Return {} when disabled (zero diff), else {"tuning_config": [value]}."""
    state = _HUST_W1R_STATE
    if not state["tuning_ready"]:
        import os

        try:
            state["tuning"] = _hust_w1r_parse_tuning_config(
                os.environ.get(_HUST_W1R_TUNING_ENV)
            )
        except ValueError as exc:
            state["tuning_error"] = str(exc)
            state["tuning_ready"] = True
        else:
            state["tuning_ready"] = True
    if state["tuning_error"]:
        raise ValueError(state["tuning_error"])
    value = state["tuning"]
    if value is None:
        return {}
    return {"tuning_config": list(value)}


def _hust_w1r_dump_limit():
    import os

    raw = os.environ.get("VLLM_HUST_GMM_COUNTS_DUMP_MAX_PER_KEY", "")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


def _hust_w1r_report_error(message):
    """Warn once; the dump never raises and never fakes a successful sample."""
    state = _HUST_W1R_STATE
    if state["error_reported"]:
        return
    state["error_reported"] = True
    try:
        logger.warning(
            "VLLM_HUST_GMM_COUNTS_DUMP: sample skipped after error (%s); "
            "further errors are silent",
            message,
        )
    except Exception:
        pass


def _hust_w1r_dump_path():
    """Resolve the per-rank dump path once; empty/unusable -> None (disabled)."""
    state = _HUST_W1R_STATE
    if state["dump_path_ready"]:
        return state["dump_path"]
    state["dump_path_ready"] = True
    try:
        import os

        raw = (os.environ.get(_HUST_W1R_DUMP_ENV) or "").strip()
        if not raw:
            return None
        rank = os.environ.get("RANK")
        if "{rank}" in raw:
            path = raw.format(rank=rank if rank is not None else "local")
        elif rank is not None:
            path = f"{raw}.rank{rank}"
        else:
            path = raw
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        state["dump_target"] = path
        state["dump_path"] = path
        return path
    except Exception as exc:  # unusable path must not break the compute path
        _hust_w1r_report_error(f"path {exc!r}")
        state["dump_path"] = None
        return None


def _hust_w1r_capture_state():
    """True inside graph capture, False outside, None when undetectable."""
    try:
        import torch

        probe = getattr(getattr(torch, "npu"), "is_current_stream_capturing")
        return bool(probe())
    except Exception:
        return None


def _hust_w1r_tensor_shape(tensor):
    """Read ``.shape`` (host side, no D2H); None when unavailable."""
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _hust_w1r_phase(x):
    """Coarse phase label from M (256 is the decode ceiling of this cohort)."""
    shape = _hust_w1r_tensor_shape(x)
    if not shape:
        return "unknown"
    return "decode" if shape[0] <= 256 else "prefill"


def _hust_w1r_split_weight(weight, need_trans):
    """Return (first weight, is_weight_list, k, n) honouring ``need_trans``."""
    is_list = isinstance(weight, (list, tuple))
    first = weight[0] if is_list else weight
    shape = _hust_w1r_tensor_shape(first) or []
    if len(shape) < 2:
        return first, is_list, None, None
    if need_trans:
        return first, is_list, shape[-1], shape[-2]
    return first, is_list, shape[-2], shape[-1]


def _hust_w1r_normalize_counts(raw, group_list_type):
    """Return per-expert counts. type 1 = counts, type 0 = cumsum bounds.

    Any other group_list_type is not normalised by this helper; the caller
    records ``counts=None`` plus ``counts_normalization=unsupported...``.
    """
    if raw is None:
        return None
    counts = [int(item) for item in raw]
    if int(group_list_type) == 1:
        return counts
    normalized = []
    previous = 0
    for bound in counts:
        normalized.append(int(bound) - previous)
        previous = int(bound)
    return normalized


def _hust_w1r_read_group_list(group_list):
    tolist = getattr(group_list, "tolist", None)
    if tolist is None:
        return None
    return [int(item) for item in tolist()]


def _hust_w1r_maybe_dump(
    x, weight, group_list, group_list_type, call_site, need_trans
):
    """Record one counts sample when enabled; always returns None."""
    path = _hust_w1r_dump_path()
    if not path:
        return None
    try:
        import json
        import os
        import time

        state = _HUST_W1R_STATE
        capture = _hust_w1r_capture_state()
        x_shape = _hust_w1r_tensor_shape(x) or []
        first, is_list, k_shape, n_shape = _hust_w1r_split_weight(
            weight, need_trans
        )
        w_shape = _hust_w1r_tensor_shape(first) or []
        experts = len(weight) if is_list else (w_shape[0] if w_shape else None)
        m = x_shape[0] if x_shape else None
        normalization = "skipped"
        if capture is True:
            sample_source = "skipped_inside_capture"
            counts = None
            raw_counts = None
        elif capture is None:
            sample_source = "skipped_unknown_capture_state"
            counts = None
            raw_counts = None
        else:
            sample_source = "eager"
            raw_counts = _hust_w1r_read_group_list(group_list)
            if int(group_list_type) in (0, 1):
                counts = _hust_w1r_normalize_counts(
                    raw_counts, group_list_type
                )
                normalization = (
                    "cumsum_to_counts" if int(group_list_type) == 0 else "counts"
                )
            else:
                counts = None
                normalization = "unsupported_group_list_type"
        key = f"{call_site}|{group_list_type}|{x_shape}|{w_shape}|{sample_source}"
        seen = state["dump_per_key"].get(key, 0)
        if seen >= _hust_w1r_dump_limit():
            return None
        state["dump_seq"] += 1
        rank = os.environ.get("RANK")
        record = {
            "schema": _HUST_W1R_DUMP_SCHEMA,
            "kind": "sample",
            "seq": state["dump_seq"],
            "ts_unix": time.time(),
            "phase": _hust_w1r_phase(x),
            "call_site": call_site,
            "sample_source": sample_source,
            "capture_state": capture,
            "group_list_type": int(group_list_type),
            "m": m,
            "k": x_shape[-1] if x_shape else k_shape,
            "n": n_shape,
            "experts": experts,
            "counts_normalization": normalization,
            "x_shape": x_shape,
            "weight_shape": w_shape,
            "split_weight_list": is_list,
            "need_trans": bool(need_trans),
            "counts_len": None if counts is None else len(counts),
            "counts_sum": None if counts is None else sum(counts),
            "nonzero_experts": (
                None if counts is None else sum(1 for c in counts if c > 0)
            ),
            "counts": counts,
            "group_list_raw": raw_counts,
            "pid": os.getpid(),
            "rank": int(rank) if rank is not None and rank.isdigit() else None,
            "local_rank": os.environ.get("LOCAL_RANK"),
            "cohort": os.environ.get("VLLM_HUST_COHORT_ID"),
            "caliber": os.environ.get("VLLM_HUST_LOAD_CALIBER_ID"),
            "sampling_semantics": _HUST_W1R_SAMPLING,
        }
        with open(path, "a", encoding="utf-8") as handle:
            if not state["dump_meta_written"]:
                meta = {
                    "schema": _HUST_W1R_DUMP_SCHEMA,
                    "kind": "meta",
                    "ts_unix": time.time(),
                    "env": {
                        _HUST_W1R_TUNING_ENV: os.environ.get(
                            _HUST_W1R_TUNING_ENV
                        ),
                        _HUST_W1R_DUMP_ENV: os.environ.get(_HUST_W1R_DUMP_ENV),
                    },
                    "dump_target": path,
                    "dump_max_per_key": _hust_w1r_dump_limit(),
                    "cohort": os.environ.get("VLLM_HUST_COHORT_ID"),
                    "caliber": os.environ.get("VLLM_HUST_LOAD_CALIBER_ID"),
                    "sampling_semantics": _HUST_W1R_SAMPLING,
                    "warning": "dump artefact, never a timing artefact",
                }
                handle.write(json.dumps(meta, ensure_ascii=False) + "\\n")
                state["dump_meta_written"] = True
            handle.write(json.dumps(record, ensure_ascii=False) + "\\n")
        state["dump_per_key"][key] = seen + 1
    except Exception as exc:  # never break the compute path, never fake a row
        _hust_w1r_report_error(repr(exc))
        return None
    return None
# --- vllm-hust W1R tier-2 host proposal (default-off) END ---
'''


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_patched(original: str) -> str:
    """Return the instrumented source; raise ValueError on missing anchors."""
    for name, anchor in (
        ("INSERT_ANCHOR", INSERT_ANCHOR),
        ("GMM1_ANCHOR", GMM1_ANCHOR),
        ("GMM2_ANCHOR", GMM2_ANCHOR),
    ):
        if original.count(anchor) != 1:
            raise ValueError(f"{name} not unique")

    patched = original.replace(
        INSERT_ANCHOR,
        INSERT_ANCHOR.replace(
            "def get_moe_offload_runtime():",
            HELPERS + "\ndef get_moe_offload_runtime():",
        ),
        1,
    )
    patched = patched.replace(GMM1_ANCHOR, GMM1_PATCHED, 1)
    patched = patched.replace(GMM2_ANCHOR, GMM2_PATCHED, 1)
    if patched.count("**_hust_w1r_tuning_kwargs(),") != 2:
        raise ValueError("tuning kwargs not injected exactly twice")
    call_marker = "        _hust_w1r_maybe_dump(\n            hidden_states,"
    if patched.count(call_marker) != 1:
        raise ValueError("dump hook not injected at apply_gmm1")
    call_marker2 = "        _hust_w1r_maybe_dump(\n            silu_out,"
    if patched.count(call_marker2) != 1:
        raise ValueError("dump hook not injected at apply_gmm2")
    return patched


def build_diff(original: str, patched: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"a/{HOST_REL}",
        tofile=f"b/{HOST_REL}",
        n=3,
    )
    return "".join(diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="",
        help="live host file to validate against (fail-closed on drift)",
    )
    parser.add_argument("--out", default=str(PATCH_OUT), help="patch output path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the on-disk patch with a freshly computed one (stale -> 4)",
    )
    args = parser.parse_args(argv)

    original = HOST_ORIG.read_text(encoding="utf-8")
    digest = sha256_text(original)
    if digest != EXPECTED_ORIG_SHA256:
        print(
            f"[drift] host_orig sha256 {digest} != {EXPECTED_ORIG_SHA256}",
            file=sys.stderr,
        )
        return EXIT_DRIFT
    if args.host:
        live = pathlib.Path(args.host).read_text(encoding="utf-8")
        live_digest = sha256_text(live)
        if live_digest != EXPECTED_ORIG_SHA256 or live != original:
            print(
                f"[drift] live host {args.host} sha256 {live_digest} != "
                f"{EXPECTED_ORIG_SHA256}",
                file=sys.stderr,
            )
            return EXIT_DRIFT

    try:
        patched = build_patched(original)
    except ValueError as exc:
        print(f"[anchor] {exc}", file=sys.stderr)
        return EXIT_ANCHOR

    diff = build_diff(original, patched)
    if not diff:
        print("[anchor] empty diff", file=sys.stderr)
        return EXIT_ANCHOR

    if args.check:
        on_disk = pathlib.Path(args.out)
        current = on_disk.read_text(encoding="utf-8") if on_disk.exists() else ""
        if current != diff:
            print(
                f"[stale] {on_disk} differs from a fresh generation "
                f"(disk={sha256_text(current)[:12]} "
                f"fresh={sha256_text(diff)[:12]})",
                file=sys.stderr,
            )
            return EXIT_STALE
        print(f"[ok] {on_disk} is up to date")
    else:
        pathlib.Path(args.out).write_text(diff, encoding="utf-8")
        print(f"[ok] wrote {args.out} ({len(diff)} bytes)")

    print(f"[sha] host_orig={digest}")
    print(f"[sha] patch={sha256_text(diff)}")
    print(f"[sha] patched={sha256_text(patched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
