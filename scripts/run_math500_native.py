"""Run the pinned HuggingFaceH4 MATH-500 Inspect evaluation unchanged.

This wrapper launches Inspect's upstream ``eval.yaml`` for either its fixed
first-three-task smoke or the full 500-task split. It does not implement a
replacement prompt, scorer, epoch policy, or dataset loader. HF cache and Inspect
logs are required to remain outside this repository because they can contain
source tasks, references, and model output.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

SOURCE_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
INSPECT_VERSION = "0.3.283"
INSPECT_TASK = f"hf/HuggingFaceH4/MATH-500@{SOURCE_REVISION}"
SAMPLE_LIMIT = 3
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def require_external_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must be outside the repository")


def validate_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path.rstrip("/") != "/v1"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "base URL must be an http(s) /v1 endpoint without credentials or query"
        )
    return base_url.rstrip("/")


def require_inspect_version(inspect_bin: str) -> None:
    result = subprocess.run(
        [inspect_bin, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or result.stdout.strip() != INSPECT_VERSION:
        raise RuntimeError(f"this entrypoint requires inspect-ai=={INSPECT_VERSION}")


def build_command(
    *,
    inspect_bin: str,
    model_id: str,
    base_url: str,
    log_dir: Path,
    full: bool = False,
) -> list[str]:
    if not model_id.strip() or model_id.startswith("vllm/"):
        raise ValueError(
            "provide the served model ID without the Inspect provider prefix"
        )
    command = [
        inspect_bin,
        "eval",
        INSPECT_TASK,
        "--model",
        f"vllm/{model_id}",
        "--model-base-url",
        validate_base_url(base_url),
        "--log-dir",
        str(log_dir),
        "--max-retries",
        "0",
    ]
    if not full:
        command.extend(["--limit", str(SAMPLE_LIMIT)])
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", required=True, help="Model ID advertised by the existing server"
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Existing OpenAI-compatible server base URL, ending in /v1",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        type=Path,
        help="Private Inspect log directory outside this repository",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all 500 tasks; default is the preselected first three",
    )
    args = parser.parse_args(argv)

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        parser.error("set HF_HOME to a cache directory outside this repository")
    inspect_bin = shutil.which("inspect")
    if inspect_bin is None:
        parser.error(
            f"install inspect-ai=={INSPECT_VERSION} in the active evaluation environment"
        )
    try:
        require_external_path(hf_home, label="HF_HOME")
        log_dir = require_external_path(args.log_dir, label="Inspect log directory")
        require_inspect_version(inspect_bin)
        command = build_command(
            inspect_bin=inspect_bin,
            model_id=args.model_id,
            base_url=args.base_url,
            log_dir=log_dir,
            full=args.full,
        )
    except ValueError as error:
        parser.error(str(error))
    except RuntimeError as error:
        parser.error(str(error))
    print(
        f"Running pinned upstream MATH-500 Inspect task; "
        f"mode={'full-500' if args.full else 'first-3'}; logs remain outside the repository."
    )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
