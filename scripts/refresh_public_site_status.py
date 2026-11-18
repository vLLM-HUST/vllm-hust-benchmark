#!/usr/bin/env python3
"""Refresh public GitHub/PyPI facts without changing curated acceptance evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    "vllm-hust": "core",
    "vllm-ascend-hust": "core",
    "triton-ascend-hust": "core",
    "vllm-hust-dev-hub": "infrastructure",
    "ascend-runtime-manager": "infrastructure",
    "vllm-hust-workstation": "infrastructure",
    "vllm-hust-benchmark": "infrastructure",
}


def read_json(url: str) -> dict:
    headers = {"User-Agent": "vllm-hust-website-status", "Accept": "application/json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Expected an object from the public metadata endpoint")
    return result


def refresh_issues(document: dict, read, now: str) -> dict:
    result = copy.deepcopy(document)
    repo = result["source_repo"]
    if repo != "vLLM-HUST/vllm-hust-benchmark":
        raise ValueError("Unexpected issue source repository")
    for item in result["issues"]:
        issue = read(f"https://api.github.com/repos/{repo}/issues/{item['number']}")
        pr = read(f"https://api.github.com/repos/{repo}/pulls/{item['pr']['number']}")
        if issue["number"] != item["number"] or pr["number"] != item["pr"]["number"]:
            raise ValueError("GitHub returned a different tracked record")
        if issue["state"] not in {"open", "closed"} or pr["state"] not in {
            "open",
            "closed",
        }:
            raise ValueError("Unknown GitHub state")
        item.update(
            state=issue["state"],
            state_reason=issue.get("state_reason"),
            closed_at=issue.get("closed_at"),
            github_updated_at=issue["updated_at"],
        )
        item["pr"].update(
            state=pr["state"],
            draft=pr["draft"],
            merged_at=pr.get("merged_at"),
            updated_at=pr["updated_at"],
            url=pr["html_url"],
            title=pr["title"],
            head_branch=pr["head"]["ref"],
        )
        if issue["state"] == "closed":
            completed = issue.get("state_reason") == "completed"
            item["status"] = "completed" if completed else "closed"
            item["status_label"] = (
                {"en": "Completed", "zh": "已完成"}
                if completed
                else {"en": "Closed", "zh": "已关闭"}
            )
        elif item.get("status") == "blocked":
            # GitHub open/closed cannot establish whether a curated blocker cleared.
            pass
        else:
            item["status"] = (
                "draft" if pr["state"] == "open" and pr["draft"] else "in-progress"
            )
            item["status_label"] = (
                {"en": "Draft", "zh": "草稿"}
                if item["status"] == "draft"
                else {"en": "Open", "zh": "开放"}
            )
    result["last_updated"] = now
    result.setdefault("curated_at", document["last_updated"])
    return result


def refresh_versions(document: dict, read, now: str) -> dict:
    result = copy.deepcopy(document)
    packages = []
    for name, group in REPOSITORIES.items():
        base = f"https://api.github.com/repos/vLLM-HUST/{name}"
        repo = read(base)
        if (
            repo["full_name"].lower() != f"vllm-hust/{name}"
            or repo["private"]
            or repo["archived"]
        ):
            raise ValueError(f"Not an active canonical public repository: {name}")
        if repo["default_branch"] != "main":
            raise ValueError(f"Repository branch policy needs review: {name}")
        # These repositories explicitly publish their integration line on main.
        commit = read(f"{base}/commits/main")
        sha = commit["sha"]
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(f"Invalid commit identity for {name}")
        packages.append(
            {
                "name": name,
                "repo": repo["html_url"],
                "group": group,
                "version": f"main@{sha[:8]}",
                "source_branch": "main",
                "source_commit_url": f"{repo['html_url']}/commit/{sha}",
                "source_updated_at": commit["commit"]["committer"]["date"],
                "version_display_label": "Repository snapshot; not a production compatibility approval",
                "version_note_zh": "仓库 main 快照；不代表该组合已通过生产兼容性验证。",
            }
        )
    registry = read("https://pypi.org/pypi/vllm-hust/json")
    result["registry"] = {
        "name": "vllm-hust",
        "version": registry["info"]["version"],
        "url": "https://pypi.org/project/vllm-hust/",
        "uploaded_at": max(
            (file["upload_time_iso_8601"] for file in registry["urls"]), default=None
        ),
        # A new registry version is NOT automatically approved by this fetch.
        "approved_for_current_stack": False,
    }
    result["packages"] = packages
    result["updated_at"] = now
    return result


def same_facts(left: dict, right: dict, timestamp: str) -> bool:
    a, b = copy.deepcopy(left), copy.deepcopy(right)
    a.pop(timestamp, None)
    b.pop(timestamp, None)
    return a == b


def is_stale(timestamp: str, now: str) -> bool:
    try:
        age = datetime.fromisoformat(
            now.replace("Z", "+00:00")
        ) - datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return age.total_seconds() > 7 * 86400 or age.total_seconds() < -300
    except (ValueError, TypeError, AttributeError):
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="Fail on remote drift; never write"
    )
    mode.add_argument(
        "--refresh", action="store_true", help="Refresh the two public snapshots"
    )
    args = parser.parse_args()
    now = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    updates = []
    for filename, refresh, timestamp in (
        ("issues.json", refresh_issues, "last_updated"),
        ("version_meta.json", refresh_versions, "updated_at"),
    ):
        path = ROOT / "website-publication" / "data" / filename
        old = json.loads(path.read_text(encoding="utf-8"))
        new = refresh(old, read_json, now)
        updates.append((path, old, new, timestamp))
    # No writes until every API lookup succeeded; do not replace facts with defaults.
    drift = False
    for path, old, new, timestamp in updates:
        changed = not same_facts(old, new, timestamp)
        print(f"{path.name}: {'remote facts changed' if changed else 'facts verified'}")
        stale = is_stale(old.get(timestamp), now)
        if stale:
            print(
                f"{path.name}: verification older than seven days or invalid; refresh required"
            )
        drift |= changed or stale
        if args.refresh:
            path.write_text(
                json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 1 if args.check and drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
