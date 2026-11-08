"""Original scientific grader inside a restricted container, using real NPU scores.

Run only in a per-task owned Docker container: network none, read-only root,
no-new-privileges, and only SETUID/SETGID/DAC_OVERRIDE capabilities. Generated
programs run unprivileged; references and original grader stay root-only.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import socket
from pathlib import Path
from types import SimpleNamespace


SOURCE_SHA256 = "5cb7e93cc0403e0aac99e14c9feb2cf45973351f7bd47861c5c82d374bbf2eb0"  # pragma: allowlist secret - original publisher grader


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verified_npu_scalar(receipt):
    if (
        receipt.get("status")
        != "ORIGINAL_AUXILIARY_METRIC_COMPLETED_NOT_SCIENTIFIC_TASK_DELIVERY"
        or receipt.get("model") != "neulab/codebert-python"
        or receipt.get("model_layers") != 11
        or receipt.get("distribution_version") != "0.4.1"
        or not str(receipt.get("device", "")).startswith("npu:")
        or receipt.get("completed_npu_forwards", 0) <= 0
    ):
        raise ValueError("actual original NPU metric receipt required")
    scores = {}
    for item in receipt["results"]:
        key = tuple(
            item[name]
            for name in ("task_sha256", "prediction_sha256", "reference_sha256")
        )
        value = item["codebert_score"]
        if key in scores or type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("unique finite actual NPU scalar records required")
        scores[key] = value

    def transport(task_inst, pred_fname, gold_fname):
        key = (
            hashlib.sha256(task_inst.encode()).hexdigest(),
            digest(pred_fname),
            digest(gold_fname),
        )
        if key not in scores:
            raise ValueError(
                "original task/program/reference do not match saved NPU metric"
            )
        return scores[key]

    return transport


def restricted_subprocess(run, expected_command, uid, gid):
    if uid <= 0 or gid < 0:
        raise ValueError("non-root scientific program user required")

    def execute(command, **kwargs):
        if command != expected_command or kwargs != {
            "capture_output": True,
            "timeout": 900,
        }:
            raise ValueError(
                "original scientific execution command and900-second budget required"
            )
        environment = dict(
            os.environ,
            HOME="/home/nonroot",
            XDG_CACHE_HOME="/tmp/science-program-cache",
            NUMBA_CACHE_DIR="/tmp/science-program-cache/numba",
            MPLCONFIGDIR="/tmp/science-program-cache/matplotlib",
        )
        return run(
            command, **kwargs, user=uid, group=gid, extra_groups=[], env=environment
        )

    return execute


def check_container():
    if not Path("/.dockerenv").is_file() or os.geteuid() != 0:
        raise RuntimeError(
            "restricted per-task Docker container required, never execute on host"
        )
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    allowed = (1 << 1) | (1 << 6) | (1 << 7)
    if (
        int(status["NoNewPrivs"].strip()) != 1
        or int(status["CapBnd"].strip(), 16) != allowed
        or int(status["CapEff"].strip(), 16) != allowed
    ):
        raise RuntimeError(
            "no-new-privileges and restricted capability bounding set required"
        )
    # In nested Docker, sysfs may describe the outer mount namespace. Query
    # interfaces in the running process's actual network namespace instead.
    if {name for _, name in socket.if_nameindex()} != {"lo"}:
        raise RuntimeError("network-none execution required")
    root = next(
        line.split()
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if line.split()[4] == "/"
    )
    if "ro" not in root[5].split(","):
        raise RuntimeError("read-only container root required")


def main(args):
    check_container()
    import pwd

    if Path.cwd() != Path("/testbed"):
        raise RuntimeError("original /testbed working directory required")
    source = args.original_source.resolve(strict=True)
    if digest(source) != SOURCE_SHA256:
        raise ValueError("unchanged original compute_scores.py required")
    scalar = verified_npu_scalar(
        json.loads(args.npu_receipt.read_text(encoding="utf-8"))
    )
    example = json.loads(
        Path("/testbed/instance_path/input/input.json").read_text(encoding="utf-8")
    )
    name = example["gold_program_name"]
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".py")
        or "\\" in name
    ):
        raise ValueError("simple original Python program filename required")
    user = pwd.getpwnam("nonroot")
    gold_dir = Path("/testbed/benchmark/gold_programs")
    if gold_dir.stat().st_uid != 0 or gold_dir.stat().st_mode & 0o077:
        raise RuntimeError(
            "reference directory must be root-only before executing untrusted code"
        )
    spec = importlib.util.spec_from_file_location(
        "sab_original_scientific_grader", source
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    original.codebert_score = scalar
    original.subprocess = SimpleNamespace(
        run=restricted_subprocess(
            original.subprocess.run,
            ["python", "-m", "program_to_eval.pred_" + Path(name).stem],
            user.pw_uid,
            user.pw_gid,
        )
    )
    # Original900-second execution, validity rules, eval script, success/score
    # override and result tuple are unchanged. Only scalar transport and UID bind.
    original.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", required=True, type=Path)
    parser.add_argument("--npu-receipt", required=True, type=Path)
    main(parser.parse_args())
