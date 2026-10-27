"""Freeze and launch one small TP2 qualification cell through npu-share."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tp2_pilot import digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--backend", choices=["FFTS+", "HOST", "AIV"], required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripts = output.with_suffix(".scripts")
    scripts.mkdir(exist_ok=False)
    for name in ("tp2_pilot.py", "launch_tp2_pilot.py"):
        shutil.copy2(Path(__file__).with_name(name), scripts / name)
    cache = output.parent / "cache"
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("TRACELOOM_"):
            del env[k]
    runtime = a.runtime.resolve()
    env.update(
        PYTHONPATH=os.pathsep.join(
            [
                str(runtime / "vllm-stock"),
                str(runtime / "vllm-ascend"),
                env.get("PYTHONPATH", ""),
            ]
        ),
        HCCL_OP_EXPANSION_MODE=a.backend,
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ENABLE_V1_MULTIPROCESSING="1",
        OMP_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        VLLM_PLUGINS="ascend,ascend_kv_connector,ascend_model,ascend_model_loader,ascend_service_profiling",
        VLLM_CACHE_ROOT=str(cache / "vllm"),
        XDG_CACHE_HOME=str(cache),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        TRITON_CACHE_DIR=str(cache / "triton"),
    )
    command = [
        "npu-share",
        "run",
        "profiling",
        "--devices",
        "2",
        "--quantum",
        "900",
        "--wait",
        "60",
        "--",
        sys.executable,
        str(scripts / "tp2_pilot.py"),
        "--runtime",
        str(runtime),
        "--output",
        str(output),
        "--model",
        str(a.model.resolve()),
        "--backend",
        a.backend,
    ]
    with output.with_suffix(".launch.json").open("x") as f:
        json.dump(
            dict(
                command=command,
                environment={
                    k: v
                    for k, v in env.items()
                    if k.startswith(
                        (
                            "VLLM_",
                            "HCCL_",
                            "PYTHON",
                            "XDG_",
                            "TORCHINDUCTOR_",
                            "TRITON_",
                        )
                    )
                },
                frozen_scripts={p.name: digest(p) for p in scripts.iterdir()},
            ),
            f,
            indent=2,
        )
    with output.with_suffix(".log").open("x") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    with output.with_suffix(".exit.json").open("x") as f:
        json.dump(dict(returncode=result.returncode), f)
    print(output, "exit", result.returncode, flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
