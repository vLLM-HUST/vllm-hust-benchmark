"""Bounded private probe: physical locks, fresh mapping, isolated process group."""

import ctypes
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cards = [int(x) for x in sys.argv[1].split(",")]
label = sys.argv[2]
timeout = int(os.environ.get("LUMI_TIMEOUT_SECONDS", "900"))
assert 60 <= timeout <= 14400
assert not (root / "evidence" / f"{label}-command.json").exists(), (
    "Do not overwrite evidence"
)
assert cards and len(cards) == len(set(cards)) and set(cards) <= set(range(8))
locks = []
for card in sorted(cards):
    for lock in [
        f"/root/npu-device-{card}.lock",
        f"/tmp/npu-physical-{card}.lock",
        f"/var/lock/betterscale-npu-{card}.lock",
    ]:
        # The open handle intentionally keeps the advisory lock for this process.
        f = open(lock, "a")  # noqa: SIM115
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
info = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
assert all(f"No running processes found in NPU {c}" in info for c in cards), (
    "NPU occupied"
)
lib = ctypes.CDLL("libascendcl.so")
assert lib.aclInit(None) == 0
rt = ctypes.CDLL("libruntime.so")
n = ctypes.c_uint32()
assert lib.aclrtGetDeviceCount(ctypes.byref(n)) == 0
mapping = {}
for i in range(n.value):
    p = ctypes.c_int32(-1)
    assert rt.rtGetDevicePhyIdByIndex(i, ctypes.byref(p)) == 0
    mapping[p.value] = i
lib.aclFinalize()
(root / "evidence" / f"{label}-admission.json").write_text(
    json.dumps({"cards": cards, "map": mapping, "smi": info}, indent=2)
)
env = os.environ.copy()
env.update(
    ASCEND_RT_VISIBLE_DEVICES=",".join(str(mapping[c]) for c in cards),
    LUMI_PHYSICAL_CARDS=",".join(map(str, cards)),
    LUMI_RUN_LABEL=label,
    PYTHONPATH=os.pathsep.join([str(root / "gamma4-guard"), env.get("PYTHONPATH", "")]),
    OMP_NUM_THREADS="4",
    VLLM_PLUGINS=env.get("VLLM_PLUGINS", "ascend"),
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
(root / "evidence" / f"{label}-command.json").write_text(
    json.dumps(
        {
            "argv": sys.argv[3:],
            "environment": {
                k: v
                for k, v in env.items()
                if k.startswith(
                    (
                        "HUST_VSPEC_",
                        "VSPEC_",
                        "LUMI_",
                        "VLLM_PLUGINS",
                        "VLLM_WORKER_MULTIPROC_METHOD",
                        "ASCEND_RT_VISIBLE_DEVICES",
                    )
                )
            },
        },
        indent=2,
    )
)
with (root / "evidence" / f"{label}.log").open("w") as log:
    process = subprocess.Popen(
        [str(root / "venv/bin/python"), *sys.argv[3:]],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        rc = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
(root / "evidence" / f"{label}-exit.json").write_text(json.dumps({"returncode": rc}))
raise SystemExit(rc)
