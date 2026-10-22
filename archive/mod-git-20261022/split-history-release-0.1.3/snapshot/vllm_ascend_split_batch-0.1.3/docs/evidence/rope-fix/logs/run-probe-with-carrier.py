#!/usr/bin/env python3
"""Run the RoPE-variants probe with the *plugin carrier* loaded.

Why a runner at all: the carrier is wired through the ``vllm.general_plugins``
entry point (``rope-fix = vllm_ascend_split_batch.rope_fix_plugin:load``), but the
probe is a bare script that never calls ``load_general_plugins()``.  This runner
therefore

1. calls ``vllm.plugins.load_general_plugins()`` (the exact call vLLM's own
   engine makes) and reports whether the ``rope-fix`` entry point is visible in
   the installed metadata, then
2. explicitly imports and calls the carrier's ``load()`` (the carrier module is
   the installed editable package -- no PYTHONPATH source override anywhere),
3. and finally executes the probe in-process, one ``--part`` per process
   (``--part all`` is just ``main`` then ``atb`` in two subprocesses).

Usage:
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-on  --part main
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-on  --part atb
  python /tmp/ropefix-probe-run.py --probe /tmp/ropefix-probe-off --part main
"""

import argparse
import json
import os
import runpy
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="directory holding the probe")
    parser.add_argument("--part", choices=("main", "atb"), required=True)
    args = parser.parse_args()

    probe = os.path.join(os.path.abspath(args.probe), "rope_variants_e2e_probe.py")
    assert os.path.isfile(probe), probe

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    import importlib.metadata as md

    discovered = {
        ep.name: ep.value
        for ep in md.distribution("vllm-ascend-split-batch").entry_points
        if ep.group == "vllm.general_plugins"
    }
    print(
        "[runner] installed vllm.general_plugins entry points: "
        + json.dumps(discovered, ensure_ascii=False)
    )
    print(
        "[runner] rope-fix entry point discovered: "
        f"{'rope-fix' in discovered} (metadata is an install-time snapshot)"
    )

    enabled = os.getenv("VLLM_HUST_ROPE_FIX") == "1"
    import vllm_ascend_split_batch.rope_fix_plugin as carrier

    print(
        f"[runner] carrier invoked directly (VLLM_HUST_ROPE_FIX="
        f"{os.getenv('VLLM_HUST_ROPE_FIX')!r}): load() -> {carrier.load()}"
    )
    if enabled:
        assert carrier.stats()["installed"], carrier.stats()

    sys.argv = [probe, "--part", args.part]
    runpy.run_path(probe, run_name="__main__")
    print(f"[runner] probe part={args.part} done; carrier stats={carrier.stats()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
