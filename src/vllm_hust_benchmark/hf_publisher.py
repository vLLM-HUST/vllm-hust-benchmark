"""Hugging Face Dataset Hub publisher for leaderboard artifacts.

Token resolution order (highest to lowest priority):
  1. ``token`` argument passed explicitly
  2. ``HF_TOKEN`` environment variable
  3. Token cached by ``huggingface-cli login``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_LEADERBOARD_FILES = [
    "leaderboard_single.json",
    "leaderboard_multi.json",
    "leaderboard_compare.json",
    "last_updated.json",
]
_OPTIONAL_FILES = [
    "hard_constraints.json",
    "leaderboard_historical.json",
    "historical_recovery_report.json",
]


def _require_huggingface_hub() -> object:
    try:
        import huggingface_hub

        return huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for HF publishing.\n"
            "Install it with:  pip install 'huggingface_hub>=0.20'\n"
            "Or:               pip install 'vllm-hust-benchmark[publish]'"
        ) from exc


def _resolve_token(token: str | None) -> str | None:
    if token:
        return token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token
    return None  # huggingface_hub falls back to its own cached token


def _create_commit_on_branch(
    api: object,
    *,
    repo_id: str,
    repo_type: str,
    branch: str,
    operations: list[object],
    commit_message: str,
) -> object:
    """Support both new (`revision`) and old (`branch`) HF commit APIs."""
    try:
        return api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=branch,
            operations=operations,
            commit_message=commit_message,
        )
    except TypeError as exc:
        if "revision" not in str(exc):
            raise
        return api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            branch=branch,
            operations=operations,
            commit_message=commit_message,
        )


def upload_leaderboard_to_hf(
    *,
    data_dir: Path,
    repo_id: str,
    token: str | None = None,
    branch: str = "main",
    path_in_repo_prefix: str = "",
    commit_message: str = "chore: update leaderboard data",
    dry_run: bool = False,
) -> list[str]:
    """Upload aggregated leaderboard JSON files to a HF dataset repo."""
    hf = _require_huggingface_hub()
    resolved_token = _resolve_token(token)
    api = hf.HfApi(token=resolved_token)

    uploads: list[tuple[Path, str]] = []
    for filename in _LEADERBOARD_FILES + _OPTIONAL_FILES:
        local_path = data_dir / filename
        if not local_path.exists():
            if filename in _OPTIONAL_FILES:
                continue
            raise FileNotFoundError(
                f"Required leaderboard file not found: {local_path}\n"
                "Run 'publish-website' first to generate aggregated outputs."
            )
        repo_path = (
            f"{path_in_repo_prefix}{filename}" if path_in_repo_prefix else filename
        )
        uploads.append((local_path, repo_path))

    if not uploads:
        raise ValueError("No files found for upload in: %s" % data_dir)

    uploaded: list[str] = []
    if dry_run:
        print(f"[dry-run] Would upload {len(uploads)} file(s) to {repo_id}@{branch}:")
        for local_path, repo_path in uploads:
            size_kb = local_path.stat().st_size / 1024
            print(
                f"  {local_path.name} → hf://datasets/{repo_id}/{repo_path}  ({size_kb:.1f} KB)"
            )
            uploaded.append(repo_path)
        return uploaded

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        logger.info("Repo %s not found; creating as private…", repo_id)
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )

    CommitOperationAdd = hf.CommitOperationAdd
    operations = [
        CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
        for local_path, repo_path in uploads
    ]

    commit_info = _create_commit_on_branch(
        api,
        repo_id=repo_id,
        repo_type="dataset",
        branch=branch,
        operations=operations,
        commit_message=commit_message,
    )
    commit_url = getattr(commit_info, "commit_url", str(commit_info))
    print(f"✅ Uploaded {len(operations)} file(s) to {repo_id}@{branch}")
    print(f"   commit: {commit_url}")

    return [repo_path for _, repo_path in uploads]
