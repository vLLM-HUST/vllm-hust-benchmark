#!/usr/bin/env python3
"""Run historical PR same-spec benchmarks and publish each result immediately."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_REPO = "intellistream/vllm-hust-benchmark-results"
DEFAULT_OFFICIAL_SPEC_DIR = REPO_ROOT / "docs" / "official-baselines"
DEFAULT_RESULT_ROOT = REPO_ROOT / ".benchmarks" / "historical-pr-backfill"
DEFAULT_CURRENT_PYTHON = "/root/miniconda3/envs/vllm-hust-dev/bin/python"
IMPORTANT_REF_GREP = (
    "perf|performance|optimi|throughput|latency|decode|scheduler|cache|prefix|kv"
)


@dataclass(frozen=True)
class TargetRef:
    label: str
    core_ref: str
    plugin_ref: str
    pr_number: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class OfficialSpec:
    path: Path
    scenario: str
    workload: str
    model: str
    precision: str
    chip_model: str
    chip_count: int
    node_count: int


def run_command(
    command: list[str],
    *,
    cwd: Path,
    execute: bool,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command)
    if not execute:
        print(f"[dry-run] {cwd}$ {printable}")
        return subprocess.CompletedProcess(command, 0, "", "")

    print(f"[run] {cwd}$ {printable}")
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        check=check,
    )


def capture_command(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9._+-]+", "-", lowered)
    lowered = lowered.strip("-")
    return lowered or "item"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scenario_to_workload(scenario: str) -> str:
    return scenario


def collect_official_specs(
    spec_dir: Path,
    *,
    workloads: set[str] | None = None,
    include_multi_chip: bool = False,
) -> list[OfficialSpec]:
    specs: list[OfficialSpec] = []
    for path in sorted(spec_dir.glob("official-ascend-jan-2026-v0180-*.json")):
        payload = load_json(path)
        chip_model = str(payload.get("hardware_chip_model") or "")
        chip_count = int(payload.get("chip_count") or 1)
        node_count = int(payload.get("node_count") or 1)
        scenario = str(payload.get("scenario") or "").strip()
        workload = scenario_to_workload(scenario)
        if chip_model != "910B2":
            continue
        if not include_multi_chip and (chip_count != 1 or node_count != 1):
            continue
        if workloads and workload not in workloads and scenario not in workloads:
            continue
        specs.append(
            OfficialSpec(
                path=path,
                scenario=scenario,
                workload=workload,
                model=str(payload.get("model") or ""),
                precision=str(payload.get("model_precision") or ""),
                chip_model=chip_model,
                chip_count=chip_count,
                node_count=node_count,
            )
        )
    return specs


def load_plan(plan_file: Path) -> list[TargetRef]:
    payload = load_json(plan_file)
    if not isinstance(payload, dict):
        raise ValueError(f"{plan_file}: plan must be a JSON object")
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise ValueError(f"{plan_file}: targets must be a list")

    parsed: list[TargetRef] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise ValueError(f"{plan_file}: targets[{index}] must be an object")
        core_ref = str(target.get("core_ref") or "").strip()
        if not core_ref:
            raise ValueError(f"{plan_file}: targets[{index}].core_ref is required")
        plugin_ref = str(target.get("plugin_ref") or "main").strip()
        label = str(target.get("label") or core_ref[:12]).strip()
        pr_number = target.get("pr_number")
        parsed.append(
            TargetRef(
                label=label,
                core_ref=core_ref,
                plugin_ref=plugin_ref,
                pr_number=int(pr_number) if pr_number is not None else None,
                notes=str(target.get("notes") or ""),
            )
        )
    return parsed


def discover_targets_from_git(
    *,
    repo: Path,
    plugin_ref: str,
    max_refs: int,
    grep: str,
) -> list[TargetRef]:
    output = capture_command(
        [
            "git",
            "log",
            "--extended-regexp",
            "--regexp-ignore-case",
            f"--grep={grep}",
            f"--max-count={max_refs}",
            "--format=%H%x09%s",
        ],
        cwd=repo,
    )
    targets: list[TargetRef] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        commit, subject = line.split("\t", 1)
        pr_match = re.search(r"#(\d+)", subject)
        label_bits = []
        if pr_match:
            label_bits.append(f"pr-{pr_match.group(1)}")
        label_bits.append(commit[:10])
        targets.append(
            TargetRef(
                label="-".join(label_bits),
                core_ref=commit,
                plugin_ref=plugin_ref,
                pr_number=int(pr_match.group(1)) if pr_match else None,
                notes=subject,
            )
        )
    return targets


def read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"runs": {}}
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), dict):
        raise ValueError(f"{path}: invalid state file")
    return payload


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def state_key(target: TargetRef, spec: OfficialSpec) -> str:
    return "|".join([target.core_ref, target.plugin_ref, spec.path.name])


def ensure_worktree(
    *,
    source_repo: Path,
    worktree_root: Path,
    ref: str,
    prefix: str,
    execute: bool,
) -> Path:
    resolved = capture_command(["git", "rev-parse", ref], cwd=source_repo)
    path = worktree_root / f"{prefix}-{resolved[:12]}"
    if path.exists():
        return path
    run_command(
        ["git", "worktree", "add", "--detach", str(path), resolved],
        cwd=source_repo,
        execute=execute,
    )
    return path


def copy_submission_to_repo(submission_dir: Path, submissions_root: Path) -> Path:
    target = submissions_root / submission_dir.parent.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(submission_dir, target)
    return target


def git_has_changes(repo: Path) -> bool:
    return bool(capture_command(["git", "status", "--short"], cwd=repo))


def commit_and_push(repo: Path, *, message: str, execute: bool) -> None:
    if execute and not git_has_changes(repo):
        print(f"[publish] no git changes in {repo}")
        return
    run_command(["git", "add", "-A"], cwd=repo, execute=execute)
    if execute and not git_has_changes(repo):
        print(f"[publish] no staged changes in {repo}")
        return
    run_command(["git", "commit", "-m", message], cwd=repo, execute=execute)
    run_command(["git", "push", "origin", "main"], cwd=repo, execute=execute)


def publish_result(
    *,
    args: argparse.Namespace,
    submission_dir: Path,
    run_id: str,
    execute: bool,
) -> None:
    publish_submission = submission_dir
    if not execute:
        print(f"[dry-run] would publish submission: {submission_dir}")
    elif args.mirror_to_benchmark_submissions:
        publish_submission = copy_submission_to_repo(
            submission_dir, REPO_ROOT / "submissions"
        )

    if args.publish_each:
        run_command(
            [
                sys.executable,
                "-m",
                "vllm_hust_benchmark.cli",
                "sync-submission-to-hf",
                "--submission-dir",
                str(publish_submission),
                "--aggregate-output-dir",
                str(REPO_ROOT / "leaderboard-data" / "snapshots"),
                "--repo-id",
                args.hf_repo,
                "--branch",
                args.hf_branch,
                "--submissions-prefix",
                args.hf_submissions_prefix,
                "--commit-message",
                f"data: publish historical PR benchmark {run_id}",
                "--execute",
            ],
            cwd=REPO_ROOT,
            execute=execute,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )
    else:
        run_command(
            [
                sys.executable,
                "-m",
                "vllm_hust_benchmark.cli",
                "publish-website",
                "--source-dir",
                str(REPO_ROOT / "submissions"),
                "--output-dir",
                str(REPO_ROOT / "leaderboard-data" / "snapshots"),
                "--execute",
            ],
            cwd=REPO_ROOT,
            execute=execute,
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        )

    run_command(
        [
            sys.executable,
            "scripts/validate_public_leaderboard_snapshots.py",
            "--snapshot-dir",
            "leaderboard-data/snapshots",
        ],
        cwd=REPO_ROOT,
        execute=execute,
    )

    if args.sync_website_each:
        website_repo = Path(args.website_repo).resolve()
        run_command(
            [
                sys.executable,
                "scripts/sync_leaderboard_snapshots.py",
                "--source-dir",
                str(REPO_ROOT / "leaderboard-data" / "snapshots"),
                "--target-dir",
                str(website_repo / "data"),
            ],
            cwd=website_repo,
            execute=execute,
        )

    if args.commit_push_each:
        commit_and_push(
            REPO_ROOT,
            message=f"data: add historical PR benchmark {run_id}",
            execute=execute,
        )
        if args.sync_website_each:
            commit_and_push(
                Path(args.website_repo).resolve(),
                message=f"data: sync historical PR benchmark {run_id}",
                execute=execute,
            )


def run_target_spec(
    *,
    args: argparse.Namespace,
    target: TargetRef,
    spec: OfficialSpec,
    core_worktree: Path,
    plugin_worktree: Path,
    execute: bool,
) -> Path:
    core_commit = (
        capture_command(["git", "rev-parse", "HEAD"], cwd=core_worktree)
        if core_worktree.is_dir()
        else capture_command(["git", "rev-parse", target.core_ref], cwd=Path(args.core_repo))
    )
    plugin_commit = (
        capture_command(["git", "rev-parse", "HEAD"], cwd=plugin_worktree)
        if plugin_worktree.is_dir()
        else capture_command(
            ["git", "rev-parse", target.plugin_ref], cwd=Path(args.plugin_repo)
        )
    )
    run_id = "-".join(
        [
            "historical-pr",
            slugify(target.label),
            slugify(spec.workload),
            core_commit[:10],
            plugin_commit[:10],
        ]
    )
    result_dir = Path(args.result_root).resolve() / "runs" / run_id
    env = {
        **os.environ,
        "CURRENT_VLLM_HUST_REPO": str(core_worktree),
        "CURRENT_VLLM_ASCEND_HUST_REPO": str(plugin_worktree),
        "CURRENT_RUNTIME_PYTHON": args.runtime_python,
        "CURRENT_SUBMITTER": args.submitter,
        "CURRENT_DATA_SOURCE": "real-online-historical-pr-backfill",
        "CURRENT_GIT_COMMIT": core_commit,
        "CURRENT_GITHUB_REF": target.label,
        "CURRENT_GITHUB_REPOSITORY": args.core_github_repository,
        "CURRENT_PLUGIN_GIT_COMMIT": plugin_commit,
        "CURRENT_PLUGIN_GITHUB_REF": target.plugin_ref,
        "CURRENT_PLUGIN_GITHUB_REPOSITORY": args.plugin_github_repository,
        "RESULT_DIR": str(result_dir),
        "RUN_ID": run_id,
    }
    if args.current_env_prefix:
        env["CURRENT_ENV_PREFIX"] = args.current_env_prefix
    if args.server_port:
        env["CURRENT_SERVER_PORT"] = str(args.server_port)
        env["CURRENT_CLIENT_PORT"] = str(args.server_port)

    run_command(
        ["bash", str(REPO_ROOT / "scripts" / "run-current-ascend-same-spec.sh"), str(spec.path)],
        cwd=REPO_ROOT,
        execute=execute,
        env=env,
    )
    return result_dir / "submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill real-online vllm-hust benchmarks for historical PR refs. "
            "Dry-run by default; pass --execute to launch services."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--discover-from-log", action="store_true")
    parser.add_argument("--max-discovered-refs", type=int, default=12)
    parser.add_argument("--discover-grep", default=IMPORTANT_REF_GREP)
    parser.add_argument("--core-repo", default=str(REPO_ROOT.parent / "vllm-hust"))
    parser.add_argument("--plugin-repo", default=str(REPO_ROOT.parent / "vllm-ascend-hust"))
    parser.add_argument("--default-plugin-ref", default="main")
    parser.add_argument("--spec-dir", type=Path, default=DEFAULT_OFFICIAL_SPEC_DIR)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--include-multi-chip", action="store_true")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--state-file")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--publish-each", action="store_true")
    parser.add_argument("--sync-website-each", action="store_true")
    parser.add_argument("--commit-push-each", action="store_true")
    parser.add_argument(
        "--mirror-to-benchmark-submissions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--hf-branch", default="main")
    parser.add_argument("--hf-submissions-prefix", default="submissions-auto")
    parser.add_argument("--website-repo", default=str(REPO_ROOT.parent / "vllm-hust-website"))
    parser.add_argument("--runtime-python", default=DEFAULT_CURRENT_PYTHON)
    parser.add_argument("--current-env-prefix", default="")
    parser.add_argument("--server-port", default="")
    parser.add_argument("--submitter", default="historical-pr-backfill")
    parser.add_argument("--core-github-repository", default="vLLM-HUST/vllm-hust")
    parser.add_argument("--plugin-github-repository", default="vLLM-HUST/vllm-ascend-hust")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execute = bool(args.execute)
    result_root = Path(args.result_root).resolve()
    state_file = Path(args.state_file).resolve() if args.state_file else result_root / "state.json"
    worktree_root = result_root / "worktrees"
    state = read_state(state_file)

    core_repo = Path(args.core_repo).resolve()
    plugin_repo = Path(args.plugin_repo).resolve()
    if args.plan_file:
        targets = load_plan(args.plan_file.resolve())
    elif args.discover_from_log:
        targets = discover_targets_from_git(
            repo=core_repo,
            plugin_ref=args.default_plugin_ref,
            max_refs=args.max_discovered_refs,
            grep=args.discover_grep,
        )
    else:
        core_head = capture_command(["git", "rev-parse", "HEAD"], cwd=core_repo)
        targets = [
            TargetRef(
                label=f"current-{core_head[:10]}",
                core_ref=core_head,
                plugin_ref=args.default_plugin_ref,
                notes="current core HEAD",
            )
        ]

    specs = collect_official_specs(
        args.spec_dir.resolve(),
        workloads=set(args.workload) if args.workload else None,
        include_multi_chip=args.include_multi_chip,
    )
    if not specs:
        raise SystemExit("no official specs matched the requested filters")
    if not targets:
        raise SystemExit("no historical target refs resolved")

    print(f"[backfill] targets: {len(targets)}")
    print(f"[backfill] specs: {len(specs)}")
    print(f"[backfill] state: {state_file}")
    print(f"[backfill] mode: {'execute' if execute else 'dry-run'}")

    for target in targets:
        core_worktree = ensure_worktree(
            source_repo=core_repo,
            worktree_root=worktree_root,
            ref=target.core_ref,
            prefix="vllm-hust",
            execute=execute,
        )
        plugin_worktree = ensure_worktree(
            source_repo=plugin_repo,
            worktree_root=worktree_root,
            ref=target.plugin_ref,
            prefix="vllm-ascend-hust",
            execute=execute,
        )

        for spec in specs:
            key = state_key(target, spec)
            previous = state["runs"].get(key, {})
            if previous.get("status") == "completed" and not args.rerun_completed:
                print(f"[backfill] skip completed: {target.label} / {spec.workload}")
                continue

            print(f"[backfill] running: {target.label} / {spec.workload}")
            state["runs"][key] = {
                "status": "running",
                "target": target.__dict__,
                "spec": str(spec.path),
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            if execute:
                write_state(state_file, state)
            try:
                submission_dir = run_target_spec(
                    args=args,
                    target=target,
                    spec=spec,
                    core_worktree=core_worktree,
                    plugin_worktree=plugin_worktree,
                    execute=execute,
                )
                if execute and not submission_dir.is_dir():
                    raise RuntimeError(f"missing submission dir: {submission_dir}")
                if args.publish_each or args.sync_website_each or args.mirror_to_benchmark_submissions:
                    publish_result(
                        args=args,
                        submission_dir=submission_dir,
                        run_id=submission_dir.parent.name,
                        execute=execute,
                    )
                state["runs"][key] = {
                    **state["runs"][key],
                    "status": "completed",
                    "submission_dir": str(submission_dir),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                if execute:
                    write_state(state_file, state)
            except Exception as exc:
                state["runs"][key] = {
                    **state["runs"].get(key, {}),
                    "status": "failed",
                    "error": str(exc),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                if execute:
                    write_state(state_file, state)
                print(f"[backfill] failed: {target.label} / {spec.workload}: {exc}", file=sys.stderr)
                if execute:
                    return 1

    print("[backfill] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
