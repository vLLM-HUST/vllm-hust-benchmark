from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REBASE_REQUIRED_EXIT_CODE = 3


def _git_stdout(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed for {repo}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def evaluate_branch_freshness(
    *,
    repo: Path,
    base_ref: str,
    head_ref: str = "HEAD",
    recommended_days: int = 3,
    recommended_commits: int = 20,
    required_days: int = 7,
    required_commits: int = 50,
    now: datetime | None = None,
) -> dict[str, Any]:
    merge_base_sha = _git_stdout(repo, "merge-base", base_ref, head_ref)
    commit_timestamp = int(_git_stdout(repo, "show", "-s", "--format=%ct", merge_base_sha))
    effective_now = now or datetime.now(tz=UTC)
    merge_base_dt = datetime.fromtimestamp(commit_timestamp, tz=UTC)
    age_seconds = max((effective_now - merge_base_dt).total_seconds(), 0.0)
    merge_base_age_days = math.floor(age_seconds / 86400.0)
    base_branch_ahead_commits = int(
        _git_stdout(repo, "rev-list", "--count", f"{merge_base_sha}..{base_ref}")
    )

    status = "fresh"
    if (
        merge_base_age_days > required_days
        or base_branch_ahead_commits > required_commits
    ):
        status = "rebase_required"
    elif (
        merge_base_age_days > recommended_days
        or base_branch_ahead_commits > recommended_commits
    ):
        status = "rebase_recommended"

    return {
        "repo": str(repo),
        "base_ref": base_ref,
        "head_ref": head_ref,
        "merge_base_sha": merge_base_sha,
        "merge_base_age_days": merge_base_age_days,
        "base_branch_ahead_commits": base_branch_ahead_commits,
        "recommended_threshold": {
            "days": recommended_days,
            "commits": recommended_commits,
        },
        "required_threshold": {
            "days": required_days,
            "commits": required_commits,
        },
        "status": status,
    }


def render_branch_freshness_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "## Branch Freshness",
        f"- Status: `{payload['status']}`",
        f"- Merge base: `{payload['merge_base_sha']}`",
        f"- Merge base age: `{payload['merge_base_age_days']}` days",
        f"- Base branch ahead commits: `{payload['base_branch_ahead_commits']}`",
        f"- Base ref: `{payload['base_ref']}`",
        f"- Head ref: `{payload['head_ref']}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check-branch-freshness",
        description="Compute branch freshness status for an L1 smoke benchmark PR run.",
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--recommended-days", type=int, default=3)
    parser.add_argument("--recommended-commits", type=int, default=20)
    parser.add_argument("--required-days", type=int, default=7)
    parser.add_argument("--required-commits", type=int, default=50)
    parser.add_argument("--output")
    parser.add_argument("--summary-file")
    args = parser.parse_args(argv)

    payload = evaluate_branch_freshness(
        repo=Path(args.repo).resolve(),
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        recommended_days=args.recommended_days,
        recommended_commits=args.recommended_commits,
        required_days=args.required_days,
        required_commits=args.required_commits,
    )

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    if args.summary_file:
        summary_path = Path(args.summary_file).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(render_branch_freshness_markdown(payload), encoding="utf-8")

    if payload["status"] == "rebase_required":
        return REBASE_REQUIRED_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
