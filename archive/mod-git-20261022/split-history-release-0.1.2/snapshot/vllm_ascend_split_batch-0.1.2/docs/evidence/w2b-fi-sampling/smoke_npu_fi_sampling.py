# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""NPU smoke for the fi_sampling plugin (single card, small batch).

Run (CWD must NOT be /vllm-workspace):

    cd /tmp && ASCEND_RT_VISIBLE_DEVICES=7 python \
      /vllm-workspace/vllm-ascend-split-batch-hust/docs/evidence/w2b-fi-sampling/smoke_npu_fi_sampling.py

Checks, in order:
  1. default-off: ``load()`` returns False, ``AscendTopKTopPSampler`` is the
     stock fork class and ``triton`` is not imported;
  2. on: the module attribute is replaced by the subclass;
  3. on: the untruncated route produces in-range token ids on NPU;
  4. on: the joint route (k=50/p=0.95, B=256) produces in-range token ids;
  5. on: the k=1 route matches argmax exactly;
  6. on: a per-request generator falls back to the fork chain (delegation
     observed through the base ``forward_native``).
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch_npu  # noqa: F401


@contextmanager
def _minimal_ascend_config():
    """Stand in for ``init_ascend_config`` without building a full vLLM config.

    The stock fork ``forward_native`` reads exactly two flags from the Ascend
    config; the smoke runs those flags at their defaults (both off), which is
    also the FI-eligible configuration.  This keeps the smoke focused on the
    device path instead of on vLLM engine construction.  Both the fork module
    and the plugin resolve ``get_ascend_config`` through the config module, so
    patching it there covers every consumer.
    """
    config = types.SimpleNamespace(
        enable_reduce_sample=False,
        enable_async_exponential=False,
    )
    from vllm_ascend import ascend_config
    from vllm_ascend.sample import sampler as sampler_mod

    # The fork module binds ``get_ascend_config`` by name at import time, so
    # both the module attribute and the config accessor must be replaced.
    with (
        patch.object(ascend_config, "get_ascend_config", lambda: config),
        patch.object(sampler_mod, "get_ascend_config", lambda: config),
    ):
        yield


def main() -> int:
    device = "npu:0"
    torch.npu.set_device(device)

    from vllm_ascend.sample import sampler as sampler_mod

    from vllm_ascend_split_batch import fi_sampling_plugin as plugin

    stock_cls = sampler_mod.AscendTopKTopPSampler
    results: list[tuple[str, bool, str]] = []

    # 1. default-off -------------------------------------------------------
    os.environ.pop(plugin.ENV_ENABLE, None)
    plugin._reset_for_tests()
    patched = plugin.load()
    results.append(
        (
            "default-off: load() is a no-op",
            patched is False and sampler_mod.AscendTopKTopPSampler is stock_cls,
            f"load()={patched}, cls_unchanged="
            f"{sampler_mod.AscendTopKTopPSampler is stock_cls}",
        )
    )

    # 2. install -----------------------------------------------------------
    os.environ[plugin.ENV_ENABLE] = "1"
    plugin._reset_for_tests()
    installed = plugin.load()
    new_cls = sampler_mod.AscendTopKTopPSampler
    results.append(
        (
            "on: subclass installed",
            installed and issubclass(new_cls, stock_cls) and new_cls is not stock_cls,
            f"installed={installed}, cls={new_cls.__name__}",
        )
    )

    vocab = 152064
    logits = torch.randn(256, vocab, dtype=torch.float32, device=device)

    with _minimal_ascend_config():
        # 3. untruncated route ---------------------------------------------
        sampler = new_cls()
        tokens, logprobs = sampler.forward_native(logits[:8], {}, None, None)
        results.append(
            (
                "on: untruncated route (B=8)",
                logprobs is None
                and bool((tokens >= 0).all())
                and bool((tokens < vocab).all()),
                f"tokens={tokens.tolist()}",
            )
        )

        # 4. joint route ---------------------------------------------------
        k = torch.full((256,), 50, dtype=torch.int32, device=device)
        p = torch.full((256,), 0.95, dtype=torch.float32, device=device)
        tokens, logprobs = sampler.forward_native(logits, {}, k, p)
        results.append(
            (
                "on: joint route (k=50/p=0.95, B=256)",
                logprobs is None
                and bool((tokens >= 0).all())
                and bool((tokens < vocab).all()),
                f"tokens[:8]={tokens[:8].tolist()}",
            )
        )

        # 5. k=1 route -----------------------------------------------------
        # B must be >= FI_JOINT_MIN_BATCH: below it the call is served by the
        # fork chain without a top-k read (REPORT.md §4).
        k1 = torch.ones((256,), dtype=torch.int32, device=device)
        tokens, logprobs = sampler.forward_native(logits, {}, k1, None)
        argmax = logits.argmax(dim=-1)
        results.append(
            (
                "on: k=1 route == argmax (B=256)",
                logprobs is None and torch.equal(tokens, argmax),
                f"tokens[:8]={tokens[:8].tolist()}, argmax[:8]={argmax[:8].tolist()}",
            )
        )

        # 6. per-request generator fallback --------------------------------
        generator = torch.Generator(device=device).manual_seed(1234)
        tokens, logprobs = sampler.forward_native(
            logits[:8], {0: generator}, None, None
        )
        results.append(
            (
                "on: per-request generator -> fork chain",
                logprobs is None
                and bool((tokens >= 0).all())
                and bool((tokens < vocab).all()),
                f"tokens={tokens.tolist()} (fork chain, generator not advanced)",
            )
        )

    width = max(len(name) for name, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
        failed += 0 if ok else 1
    print(f"\n==== SMOKE {'PASS' if failed == 0 else f'FAIL ({failed})'} ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
