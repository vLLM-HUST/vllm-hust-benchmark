# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import argparse
import difflib
import hashlib
from pathlib import Path

HOST_SHA256 = "1081d19738da1130cf85beb7c1bcb4d061a0ca83f0edd4f3a7f2b30bd185e35d"
TARGET = "vllm_ascend/ops/gdn.py"
FLAG = """_FUSE_GDN_QKV_NORM = (
    os.environ.get("VLLM_HUST_GDN_QKV_NORM", "0") == "1"
    and os.environ.get("USE_DEFAULT_FLA_NORM", "0") == "0"
)
"""
UNPACK_OLD = (
    "        query_non_spec, key_non_spec, value_non_spec = "
    "self.rearrange_mixed_qkv(mixed_qkv_non_spec)\n"
)
UNPACK_NEW = """        qk_non_spec_normalized = False
        if (
            _FUSE_GDN_QKV_NORM
            and spec_sequence_masks is None
            and self.num_spec == 0
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and attn_metadata.num_decode_tokens == 16
            and mixed_qkv_non_spec is not None
            and mixed_qkv_non_spec.shape == (16, 4096)
            and mixed_qkv_non_spec.stride() == (4096, 1)
            and mixed_qkv_non_spec.dtype == torch.bfloat16
            and mixed_qkv_non_spec.device.type == "npu"
            and not mixed_qkv_non_spec.requires_grad
            and self.key_dim // self.tp_size == 1024
            and self.value_dim // self.tp_size == 2048
            and self.head_k_dim == 128
            and self.head_v_dim == 128
            and get_pcp_group().world_size == 1
        ):
            from vllm_ascend_split_batch.gdn_qkv_norm import unpack_norm

            query_non_spec, key_non_spec, value_non_spec = unpack_norm(
                mixed_qkv_non_spec
            )
            qk_non_spec_normalized = True
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
"""
NORM_OLD = """            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
"""
NORM_NEW = """            if not qk_non_spec_normalized:
                query_non_spec = l2norm_fwd(query_non_spec)
                key_non_spec = l2norm_fwd(key_non_spec)
"""


def build_patched(source):
    if hashlib.sha256(source.encode()).hexdigest() != HOST_SHA256:
        raise ValueError("host source drift: do not apply this pinned proposal")
    anchor = "\n\nclass AscendGatedDeltaNetAttention("
    assert source.count(anchor) == 1
    result = source.replace(anchor, "\n\n" + FLAG + anchor)
    for old, new in [(UNPACK_OLD, UNPACK_NEW), (NORM_OLD, NORM_NEW)]:
        if result.count(old) != 1:
            raise ValueError("host callsite drift")
        result = result.replace(old, new)
    return result


def make_diff(source):
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            build_patched(source).splitlines(keepends=True),
            fromfile="a/" + TARGET,
            tofile="b/" + TARGET,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(make_diff(args.source.read_text()), end="")
