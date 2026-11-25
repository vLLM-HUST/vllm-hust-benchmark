"""Score completed private FS/HLE runs with separate target-only judges.

Run only after all solver services have stopped. No GitHub/PyPI operations.
The judge is restarted for each dataset because their text/image profiles differ.
Use --first-round-only to score the existing ten completed first-round runs.
Use --round2-only to score the ten independently repeated second-round runs.
Use --round3-only to score the ten independently repeated third-round runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/root/recovery/issue18-delivery-20261113")
BIN = Path("/root/recovery/vspec-install-r8-ZoYz6V/venv/bin")
CLIENT = Path("/root/recovery/vspec-client-clean-20261107/bin/python")
RUNNER = Path("/root/recovery/vspec-download-reinstall-20261108/examples/qwen35_three_datasets/paired.py")
ARMS = {
    "frontierscience": {
        "MTP2": (Path("/root/f1311n2"), "native", "r4"),
        "MTP4": (Path("/root/f1311n4"), "native", "r3"),
        "MTP6": (Path("/root/f1311n6"), "native", "r1"),
        "MTP8": (Path("/root/f1311n8"), "native", "r1"),
        "vSpec": (Path("/root/f1311bridge"), "dynamic", "r1"),
    },
    "hle-verified": {
        "MTP2": (Path("/root/h1311n2"), "native", "r1"),
        "MTP4": (Path("/root/h1311n4"), "native", "r1"),
        "MTP6": (Path("/root/h1311n6"), "native", "r1"),
        "MTP8": (Path("/root/h1311n8"), "native", "r2"),
        "vSpec": (Path("/root/h1311hist1"), "dynamic", "r1"),
    },
}
ROUND2_ARMS = {
    "frontierscience": {
        **{name: (root, arm, "round2") for name, (root, arm, _) in ARMS["frontierscience"].items()},
        "vSpec": (Path("/root/f1311bridge-retest-20261116"), "dynamic", "round2"),
    },
    "hle-verified": {
        **{name: (root, arm, "round2") for name, (root, arm, _) in ARMS["hle-verified"].items()},
        "vSpec": (Path("/root/h1311hist1-retest-20261116"), "dynamic", "round2"),
    },
}
ROUND3_ARMS = {
    dataset: {
        name: (root, arm, "round3")
        for name, (root, arm, _) in arms.items()
    }
    for dataset, arms in ROUND2_ARMS.items()
}
PLANS = {
    "frontierscience": WORK / "fs-plan160-alt",
    "hle-verified": WORK / "hle-plan1811-alt",
}
JUDGE_CONFIGS = {
    "frontierscience": WORK / "fs-target-only-judge.yaml",
    "hle-verified": WORK / "hle-target-only-judge.yaml",
}


def wait_for_health(process: subprocess.Popen, timeout_seconds: int = 1800) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Judge service exited with code {process.returncode}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:18196/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(5)
    raise TimeoutError("Target-only judge did not become healthy in 1800 seconds")


def score_dataset(
    dataset: str,
    first_round_only: bool = False,
    round2_only: bool = False,
    round3_only: bool = False,
) -> None:
    arms = ROUND3_ARMS[dataset] if round3_only else ROUND2_ARMS[dataset] if round2_only else ARMS[dataset]
    for root, arm, first_tag in arms.values():
        tags = (first_tag,) if first_round_only or round2_only or round3_only else (first_tag, "round2", "round3")
        for tag in tags:
            receipt = root / "runs" / dataset / arm / tag / "solver-receipt.json"
            if not receipt.is_file():
                raise FileNotFoundError(f"Incomplete solver matrix: {receipt}")

    setup = json.loads((arms["MTP4"][0] / "setup.json").read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update(setup["common_env"])
    env["HI_PYTHON"] = str(BIN / "python")
    env["VSPEC_IPC_BASE"] = "/root/.vspec-ipc-1311-judge"
    env["VSPEC_JUDGE_UPSTREAM_MAMBA_GROUPS"] = "1"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = WORK / f"judge-{dataset}-{stamp}.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(BIN / "python"), "-m", "vllm.entrypoints.cli.main", "serve", "--config", str(JUDGE_CONFIGS[dataset])],
            cwd=str(WORK), env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_health(process)
            for name, (root, arm, first_tag) in arms.items():
                tags = (first_tag,) if first_round_only or round2_only or round3_only else (first_tag, "round2", "round3")
                for tag in tags:
                    run_dir = root / "runs" / dataset / arm / tag
                    if (run_dir / "score-receipt.json").is_file():
                        print(f"Already scored: {dataset} {name} {tag}", flush=True)
                        continue
                    print(f"Scoring: {dataset} {name} {tag}", flush=True)
                    subprocess.run(
                        [str(CLIENT), str(RUNNER), "score", "--setup", str(root / "setup.json"),
                         "--run-dir", str(run_dir), "--plan", str(PLANS[dataset])],
                        cwd=str(WORK), check=True,
                    )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    round_mode = parser.add_mutually_exclusive_group()
    round_mode.add_argument("--first-round-only", action="store_true")
    round_mode.add_argument("--round2-only", action="store_true")
    round_mode.add_argument("--round3-only", action="store_true")
    parser.add_argument("--dataset", choices=tuple(ARMS))
    args = parser.parse_args()
    for dataset in ((args.dataset,) if args.dataset else ARMS):
        score_dataset(
            dataset,
            first_round_only=args.first_round_only,
            round2_only=args.round2_only,
            round3_only=args.round3_only,
        )


if __name__ == "__main__":
    main()
