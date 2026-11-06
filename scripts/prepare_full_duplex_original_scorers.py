"""Prepare a pinned subset of official offline scorers/examples, not a benchmark run.

Default: offline plan. Acquisition writes only a fresh external directory.
Optional --include-client also prepares pinned inference/ASR source, not models.
No whole repository, node_modules, audio, model weights or judge is downloaded.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REPOSITORY = "DanielLin94144/Full-Duplex-Bench"
REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"  # pragma: allowlist secret - public publisher revision
PREFIX = "v1_v1.5/evaluation/"
FILES = {
    "LICENSE": "c657cab45c850057a63b3605897f5195f3c4ac02",  # pragma: allowlist secret
    "README.md": "5964b7c6c8814bcb0aacff531c1865b1e50f7263",  # pragma: allowlist secret
    "v1_v1.5/README.md": "3575d436ef3cf482de8fbdfcd81b9ea8026d3bac",  # pragma: allowlist secret
    "v1_v1.5/dataset/README.md": "fdd062e339acb4408c08f12c8a7d131098bc0807",  # pragma: allowlist secret
    PREFIX
    + "README.md": "aa1ed10a7b99ccec8e0b1259660d91429816380a",  # pragma: allowlist secret
    "v1_v1.5/requirements.txt": "cb87e00ec03f09ee6b75f5ffd7993e54b4e9cfef",  # pragma: allowlist secret
    PREFIX
    + "eval_pause_handling.py": "9438d2361f637eb42249059de117a5d299fcd99c",  # pragma: allowlist secret
    PREFIX
    + "eval_smooth_turn_taking.py": "743781e225b9ec02d1cd972d0e5489e937165de0",  # pragma: allowlist secret
    PREFIX
    + "example_data/pause_handling_example/1/output.json": "ded34ac3154794ae970f74cc56b1b42549576d09",  # pragma: allowlist secret
    PREFIX
    + "example_data/smooth_turn_taking_example/1/output.json": "9c84430894877f9a3af49db1fe399c12a1755024",  # pragma: allowlist secret
    PREFIX
    + "example_data/smooth_turn_taking_example/1/turn_taking.json": "089998bdc6ad97ced6a7056b51a4f24773db440d",  # pragma: allowlist secret
}
MAX_FILE_BYTES = 100_000
CLIENT_PREFIX = "v1_v1.5/"
CLIENT_FILES = {
    CLIENT_PREFIX
    + "model_inference/moshi/README.md": "b59c2433f6e330c2a40fc2a551544aabbdab750f",  # pragma: allowlist secret - public original Git blob
    CLIENT_PREFIX
    + "model_inference/moshi/inference.py": "5bd9521a6e990c519fcc83aa4c8ccaa291d7dcfe",  # pragma: allowlist secret - public original Git blob
    CLIENT_PREFIX
    + "get_transcript/README.md": "0c9b0aa4dc5dbe7abae15ec9a8e00a01f091a1f4",  # pragma: allowlist secret - public original Git blob
    CLIENT_PREFIX
    + "get_transcript/asr.py": "822bc204e372606545845396416d05d9ab99ba8e",  # pragma: allowlist secret - public original Git blob
}
REFERENCE_FILES = {
    PREFIX
    + "get_timing.py": "64b28dae8c52d6a33429f455eaa7318c91fead7e",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "eval_behavior.py": "f00ae67d8a5ae946a56872e40cb8e88d5e24f850",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "instruction/behavior.txt": "59bdddd128263376d2eafa6aabcd7b9c8c2b1026",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "evaluate.py": "c3947fcd65b6d74f14b05dae0f032096a9f1ec84",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "eval_user_interruption.py": "2a53bda332f80c8e884561096a544603b0315897",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "eval_backchannel.py": "7398ba2409e5beb8365eb1324348bc413dbb1fcf",  # pragma: allowlist secret - publisher Git blob
    PREFIX
    + "icc_gt_distribution.json": "acf606f2d1069a230138cb3fb916cdd8cb9a034c",  # pragma: allowlist secret - publisher Git blob
}
FILE_BYTE_LIMITS = {PREFIX + "icc_gt_distribution.json": 207576}


def selected_files(*, include_client=False, include_reference=False):
    selected = dict(FILES)
    if include_client or include_reference:
        selected.update(CLIENT_FILES)
    if include_reference:
        selected.update(REFERENCE_FILES)
    return selected


def check_bytes(name, raw):
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if (
        len(raw) > FILE_BYTE_LIMITS.get(name, MAX_FILE_BYTES)
        or identity != (FILES | CLIENT_FILES | REFERENCE_FILES)[name]
    ):
        raise ValueError("pinned original Git blob mismatch: " + name)


def verify(root, *, include_client=False, include_reference=False):
    root = Path(root).resolve(strict=True)
    selected = selected_files(
        include_client=include_client, include_reference=include_reference
    )
    for name in selected:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("original file escapes selected root")
        with path.open("rb") as stream:
            check_bytes(
                name, stream.read(FILE_BYTE_LIMITS.get(name, MAX_FILE_BYTES) + 1)
            )
    return {
        "source_id": "FULL_DUPLEX_BENCH",
        "revision": REVISION,
        "status": "PINNED_SCORER_AND_CLIENT_SUBSET_VERIFIED_NOT_NATIVE_EXECUTION"
        if include_client or include_reference
        else "PINNED_SCORER_SUBSET_VERIFIED_NOT_NATIVE_EXECUTION",
        "selected_original_files_verified": len(selected),
        "client_sources_included": include_client or include_reference,
        "complete_reference_sources_included": include_reference,
        "whole_repository_verified": False,
        "audio_acquired": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def acquire(root, *, opener=urlopen, include_client=False, include_reference=False):
    root = Path(root).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("use external server storage, not the benchmark repository")
    root.mkdir(parents=True, exist_ok=False)
    selected = selected_files(
        include_client=include_client, include_reference=include_reference
    )
    for name in selected:
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}"
        with opener(url, timeout=30) as response:
            raw = response.read(FILE_BYTE_LIMITS.get(name, MAX_FILE_BYTES) + 1)
        check_bytes(name, raw)
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return verify(
        root, include_client=include_client, include_reference=include_reference
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-client",
        action="store_true",
        help="include four original Moshi/ASR software files; no model execution",
    )
    parser.add_argument(
        "--include-reference",
        action="store_true",
        help="include all v1.0/v1.5 reference timing/behavior/category scorers and client source; no models or scoring execution",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.download or args.verify_only:
        if args.output is None:
            parser.error("--output is required for acquisition/verification")
        result = (
            acquire(
                args.output,
                include_client=args.include_client,
                include_reference=args.include_reference,
            )
            if args.download
            else verify(
                args.output,
                include_client=args.include_client,
                include_reference=args.include_reference,
            )
        )
    else:
        result = {
            "status": "PLAN_ONLY_NO_DOWNLOAD",
            "repository": REPOSITORY,
            "revision": REVISION,
            "selected_files": list(
                selected_files(
                    include_client=args.include_client,
                    include_reference=args.include_reference,
                )
            ),
            "client_sources_included": args.include_client or args.include_reference,
            "complete_reference_sources_included": args.include_reference,
            "model_calls": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
