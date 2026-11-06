"""Plan or run the fixed original 27-case/39-session duplex capture.

Calls the pinned upstream record CLIs unchanged. Does not deploy a model,
transcribe, score, or qualify a source as delivered. Default is offline plan.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

OMNI_REVISION = "272845a9caadb82f220d7f3d0b64fdd1098fbbca"  # pragma: allowlist secret - publisher revision
MODULE_HASHES = {
    "v1.0": "d7b68e0def9dd8c47d58f97c46912372be9ec24075ec759b789282d60cd53923",  # pragma: allowlist secret - public CLI SHA256
    "v1.5": "360cc2655c3b0e8c8afb02532243bc65689571d0f62b36782a480d7bf17f98dc",  # pragma: allowlist secret - public CLI SHA256
}
FAMILIES = {
    "v1.0": {
        "synthetic_pause_handling": (1, 2, 3),
        "candor_pause_handling": (1, 2, 3),
        "candor_turn_taking": (1, 2, 3),
        "synthetic_user_interruption": (1, 2, 3),
        "icc_backchannel": (0, 1, 2),
    },
    "v1.5": {
        "user_interruption": (1, 2, 3),
        "user_backchannel": (1, 2, 3),
        "talking_to_other": (1, 2, 3),
        "background_speech": (1, 2, 3),
    },
}


def module(version):
    return "benchmarks.eval.benchmark_duplex_" + version.replace(".", "")


def plan(args):
    endpoint = urlsplit(args.url)
    if (
        endpoint.scheme not in {"ws", "wss"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path != "/v1/realtime"
    ):
        raise ValueError("credential-free native /v1/realtime WebSocket URL required")
    commands = []
    for version, families in FAMILIES.items():
        for family, ids in families.items():
            for case in ids:
                sample = f"{family}/{case}"
                output = args.output / f"case-{len(commands):02d}"
                command = [
                    args.python,
                    "-m",
                    module(version),
                    "record",
                    "--dataset-root",
                    str(args.dataset_root / version),
                    "--url",
                    args.url,
                    "--output",
                    str(output),
                    "--profile",
                    "minicpmo-native-pr2377",
                    "--dataset-revision",
                    args.dataset_revision_v10
                    if version == "v1.0"
                    else args.dataset_revision_v15,
                    "--server-revision",
                    OMNI_REVISION,
                    "--model",
                    args.model,
                    "--model-revision",
                    args.model_revision,
                    "--runtime",
                    args.runtime,
                    "--sample-id",
                    sample,
                ]
                commands.append(
                    {
                        "version": version,
                        "sample_id": sample,
                        "command": command,
                        "output": str(output),
                    }
                )
    return commands


def verify_inputs(args):
    root = args.omni_root.resolve(strict=True)
    for version, expected in MODULE_HASHES.items():
        path = root / (module(version).replace(".", "/") + ".py")
        if (
            path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("pinned original capture CLI differs: " + version)
        for family, ids in FAMILIES[version].items():
            for case in ids:
                audio = args.dataset_root / version / family / str(case) / "input.wav"
                if not audio.is_file():
                    raise ValueError("fixed original input missing: " + str(audio))
    # Full archive hash/CRC verification is separately required before extraction.
    return root


def execute(args, commands):
    root = verify_inputs(args)
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("capture output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixed-plan-public.json").write_text(
        json.dumps(commands, indent=2), encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    for i, row in enumerate(commands):
        with (output / f"case-{i:02d}.log").open("wb") as log:
            code = subprocess.run(
                row["command"],
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        results.append(
            {
                "version": row["version"],
                "sample_id": row["sample_id"],
                "returncode": code,
            }
        )
        (output / "capture-progress-public.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        if code:
            return code  # Preserve the failed original case; no replacement or retries.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--url", required=True)
    parser.add_argument("--dataset-revision-v10", required=True)
    parser.add_argument("--dataset-revision-v15", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.omni_root = args.omni_root.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.resolve()
    commands = plan(args)
    if args.execute:
        return execute(args, commands)
    if args.verify_only:
        verify_inputs(args)
    print(
        json.dumps(
            {
                "status": "INPUTS_VERIFIED_NOT_EXECUTED"
                if args.verify_only
                else "PLAN_ONLY_NO_MODEL_CALLS",
                "cases": commands,
                "selected_cases": 27,
                "selected_sessions": 39,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
