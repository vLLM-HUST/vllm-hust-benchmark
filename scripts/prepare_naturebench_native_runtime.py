"""Prepare verified original NatureBench software with a disclosed network profile.

No task data, package installation, Docker execution, model request or scoring.
The original source remains unchanged. Generated profiles are not native runs.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_jsonschemabench_author_source import verify_source


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/naturebench-author-source.json"
PROFILES = ("original", "owned-hostnet", "original-context", "owned-hostnet-context")


def network_profile(raw, profile):
    if profile not in PROFILES:
        raise ValueError("unknown NatureBench network profile")
    if profile in {"original", "original-context"}:
        return context_profile(raw) if profile.endswith("-context") else raw
    before = b'            "docker", "run",\n            "--name", container_name,'
    after = (
        b'            "docker", "run",\n'
        b'            "--network", "host",\n'
        b'            "--name", container_name,'
    )
    old_host = b'docker_cmd.extend(["--add-host=host.docker.internal:host-gateway"])'
    new_host = b'docker_cmd.extend(["--add-host=host.docker.internal:127.0.0.1"])'
    if raw.count(before) != 1 or raw.count(old_host) != 1:
        raise ValueError("original network construction differs; no heuristic patch")
    derived = raw.replace(before, after, 1).replace(old_host, new_host, 1)
    ast.parse(derived)
    return context_profile(derived) if profile.endswith("-context") else derived


def context_profile(raw):
    before = b'"ANTHROPIC_API_BASE", "ANTHROPIC_MODEL",\n'
    after = (
        b'"ANTHROPIC_API_BASE", "ANTHROPIC_MODEL", "CLAUDE_CODE_MAX_CONTEXT_TOKENS",\n'
    )
    if raw.count(before) != 1:
        raise ValueError("original task-container environment forwarding differs")
    derived = raw.replace(before, after, 1)
    ast.parse(derived)
    return derived


def original_files(author_root):
    root = verify_source(author_root, PINS)
    pins = json.loads(PINS.read_text("utf-8"))
    return root, pins


def profile_files(root, pins, profile):
    for item in pins["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe pinned source path")
        raw = (root / relative).read_bytes()
        yield (
            relative,
            (network_profile(raw, profile) if item["path"] == "solve.py" else raw),
        )


def prepare(author_root, output, profile="original"):
    if profile not in PROFILES:
        raise ValueError("unknown NatureBench network profile")
    root, pins = original_files(author_root)
    output = Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("runtime output must be separate from original source")
    if output.exists():
        raise FileExistsError(
            "fresh runtime output required; previous evidence retained"
        )
    # Validate the complete transformation before creating any output.
    files = list(profile_files(root, pins, profile))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = []
    for relative, raw in files:
        destination = output / "source" / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(raw)
        manifest.append(
            {
                "path": relative.as_posix(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    result = {
        "source_id": "NATUREBENCH",
        "status": "RUNTIME_SOFTWARE_PREPARED_NOT_NATIVE_DELIVERY",
        "code_revision": pins["revision"],
        "network_profile": profile,
        "changed_original_files": [] if profile == "original" else ["solve.py"],
        "network_scope": (
            "Unmodified original Docker networking"
            if profile in {"original", "original-context"}
            else "Dedicated owned pod only: task host network and loopback host alias; "
            "not a general untrusted-network isolation profile"
        ),
        "task_container_context_capacity_forwarded": profile.endswith("-context"),
        "tasks_prompts_budgets_and_scorers_changed": False,
        "files": manifest,
        "task_data_copied": False,
        "model_calls": 0,
        "scientific_task_executed": False,
        "source_checkbox_eligible": False,
    }
    with (output / "runtime-profile.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    verify_runtime(root, output)
    return result


def verify_runtime(author_root, output):
    root, pins = original_files(author_root)
    output = Path(output).resolve(strict=True)
    result = json.loads((output / "runtime-profile.json").read_text("utf-8"))
    profile = result["network_profile"]
    if (
        profile not in PROFILES
        or result["source_id"] != "NATUREBENCH"
        or result["code_revision"] != pins["revision"]
        or result["source_checkbox_eligible"] is not False
        or result["tasks_prompts_budgets_and_scorers_changed"] is not False
    ):
        raise ValueError("runtime identity or preparation boundary differs")
    expected = list(profile_files(root, pins, profile))
    manifest = [
        {
            "path": p.as_posix(),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for p, raw in expected
    ]
    if result["files"] != manifest or result["changed_original_files"] != (
        [] if profile == "original" else ["solve.py"]
    ):
        raise ValueError("runtime manifest differs from exact original/profile")
    for relative, raw in expected:
        candidate = output / "source" / relative
        if candidate.is_symlink() or candidate.read_bytes() != raw:
            raise ValueError("runtime file differs from the exact original/profile")
    python_files = {
        p.relative_to(output / "source").as_posix()
        for p in (output / "source").rglob("*.py")
    }
    if python_files != {p.as_posix() for p, _ in expected if p.suffix == ".py"}:
        raise ValueError("unexpected runtime Python module")
    return output / "source"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="original")
    parser.add_argument(
        "--write", action="store_true", help="Create fresh software profile"
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing profile")
    args = parser.parse_args()
    if args.write and args.verify:
        parser.error("choose write or verify, not both")
    if args.verify:
        source = verify_runtime(args.author_root, args.output)
        result = {
            "status": "RUNTIME_SOFTWARE_VERIFIED_NOT_NATIVE_RUN",
            "source_root": str(source),
        }
    elif args.write:
        result = prepare(args.author_root, args.output, args.profile)
    else:
        root, pins = original_files(args.author_root)
        files = list(profile_files(root, pins, args.profile))
        result = {
            "status": "OFFLINE_RUNTIME_PROFILE_PLAN",
            "files": len(files),
            "network_profile": args.profile,
            "output_created": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
