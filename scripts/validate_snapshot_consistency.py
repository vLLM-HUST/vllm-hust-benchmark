#!/usr/bin/env python3
"""Validate cross-repo consistency of leaderboard snapshot checksums.

Compares sha256 checksums of leaderboard snapshot files across three
sources: the local benchmark repo, a Hugging Face dataset repo, and the
website repo. External sources that are unavailable are warned about and
skipped; any checksum mismatch is a hard failure (exit code 1).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_FILES = (
    "leaderboard_single.json",
    "leaderboard_multi.json",
    "leaderboard_compare.json",
    "last_updated.json",
)
SNAPSHOT_SUBDIR = Path("leaderboard-data") / "snapshots"
WEBSITE_DATA_CANDIDATES = (
    Path("public") / "data",
    Path("data"),
    Path("static") / "data",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate cross-repo snapshot checksum consistency."
    )
    parser.add_argument(
        "--benchmark-repo",
        default=str(REPO_ROOT),
        help="Benchmark repository root directory.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face dataset repo ID. Use 'none' or omit to skip HF validation.",
    )
    parser.add_argument(
        "--website-repo",
        default=None,
        help="Website repository local path.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF API token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--skip-website",
        action="store_true",
        help="Skip website repository validation.",
    )
    parser.add_argument(
        "--snapshot-files",
        nargs="+",
        default=list(DEFAULT_SNAPSHOT_FILES),
        help="Snapshot file names to validate.",
    )
    return parser.parse_args()


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def resolve_hf_token(token: str | None) -> str | None:
    return (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def validate_local(
    snapshot_dir: Path, snapshot_files: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Compute sha256 for local snapshot files. Returns (hashes, errors)."""
    hashes: dict[str, str] = {}
    errors: list[str] = []
    for file_name in snapshot_files:
        path = snapshot_dir / file_name
        if not path.is_file():
            errors.append(f"missing local snapshot: {path}")
            continue
        hashes[file_name] = compute_sha256(path)
    return hashes, errors


def validate_hf(
    hf_repo_id: str,
    snapshot_files: list[str],
    local_hashes: dict[str, str],
    token: str | None,
    errors: list[str],
    warnings: list[str],
) -> bool:
    """Download HF snapshots and compare sha256. Returns True if any file checked."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        warnings.append(
            "huggingface_hub not installed; skipping HF validation. "
            "Install with: pip install 'vllm-hust-benchmark[publish]'"
        )
        return False

    resolved_token = resolve_hf_token(token)
    checked = False
    for file_name in snapshot_files:
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=file_name,
                repo_type="dataset",
                token=resolved_token,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"failed to download {file_name} from HF repo {hf_repo_id}: {exc}"
            )
            continue
        hf_sha = compute_sha256(Path(local_path))
        local_sha = local_hashes.get(file_name, "")
        checked = True
        if hf_sha != local_sha:
            errors.append(
                f"{file_name}: HF checksum mismatch (local={local_sha}, hf={hf_sha})"
            )
    return checked


def validate_website(
    website_repo: Path,
    snapshot_files: list[str],
    local_hashes: dict[str, str],
    errors: list[str],
    warnings: list[str],
) -> bool:
    """Find website data dir and compare sha256. Returns True if any file checked."""
    data_dir: Path | None = None
    for candidate in WEBSITE_DATA_CANDIDATES:
        candidate_path = website_repo / candidate
        if candidate_path.is_dir():
            data_dir = candidate_path
            break
    if data_dir is None:
        warnings.append(
            f"website data directory not found under {website_repo} "
            f"(tried: {', '.join(str(c) for c in WEBSITE_DATA_CANDIDATES)}); "
            "skipping website validation"
        )
        return False

    checked = False
    for file_name in snapshot_files:
        path = data_dir / file_name
        if not path.is_file():
            errors.append(f"missing website snapshot: {path}")
            continue
        website_sha = compute_sha256(path)
        local_sha = local_hashes.get(file_name, "")
        checked = True
        if website_sha != local_sha:
            errors.append(
                f"{file_name}: website checksum mismatch "
                f"(local={local_sha}, website={website_sha})"
            )
    return checked


def main() -> int:
    args = parse_args()
    benchmark_repo = Path(args.benchmark_repo)
    snapshot_dir = benchmark_repo / SNAPSHOT_SUBDIR

    errors: list[str] = []
    warnings: list[str] = []

    local_hashes, local_errors = validate_local(snapshot_dir, args.snapshot_files)
    errors.extend(local_errors)

    if local_errors:
        print("snapshot consistency validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    # HF validation
    hf_repo_id = (args.hf_repo_id or "").strip()
    if hf_repo_id and hf_repo_id.lower() != "none":
        validate_hf(
            hf_repo_id,
            args.snapshot_files,
            local_hashes,
            args.token,
            errors,
            warnings,
        )
    else:
        warnings.append("HF repo id not provided ('none'); skipping HF validation")

    # Website validation
    if not args.skip_website:
        website_repo = (args.website_repo or "").strip()
        if not website_repo:
            warnings.append(
                "website repo path not provided; skipping website validation"
            )
        else:
            validate_website(
                Path(website_repo),
                args.snapshot_files,
                local_hashes,
                errors,
                warnings,
            )

    if errors:
        print("snapshot consistency validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    print("all repos in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
