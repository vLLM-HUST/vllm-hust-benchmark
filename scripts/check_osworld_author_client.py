"""Verify pinned author code and optional offline client imports, never run tasks."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/osworld-author-source.json"
LOCK = ROOT / "configs/workload-native/osworld-client-requirements.lock"


def locked_packages(path):
    result = {}
    for line in path.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise ValueError("exact public package version required, no URLs or paths")
        name, version = match.groups()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in result:
            raise ValueError("duplicate normalized package name")
        result[key] = version
    if not result:
        raise ValueError("nonempty client software lock required")
    return result


def check_imports(author_root):
    original_path = list(sys.path)
    sys.path.insert(0, str(author_root))
    try:
        environment = importlib.import_module("desktop_env.desktop_env")
        setup = importlib.import_module("desktop_env.controllers.setup")
        metrics = importlib.import_module("desktop_env.evaluators.metrics")
        getters = importlib.import_module("desktop_env.evaluators.getters")
        agent = importlib.import_module("mm_agents.qwen3vl_agent")
        for module, names in [
            (environment, ["DesktopEnv"]),
            (setup, ["SetupController"]),
            (metrics, ["match_in_list", "is_cookie_deleted", "is_expected_tabs"]),
            (getters, ["get_default_search_engine"]),
            (agent, ["Qwen3VLAgent"]),
        ]:
            if not all(callable(getattr(module, name, None)) for name in names):
                raise ValueError("required original client or grader export absent")
    finally:
        sys.path[:] = original_path


def run(author_root, action):
    pins = json.loads(PROFILE.read_text("utf-8"))
    packages = locked_packages(LOCK)
    report = {
        "source_id": "OSWORLD",
        "status": "PLAN_ONLY_NO_AUTHOR_IMPORTS",
        "source_revision": pins["revision"],
        "locked_client_packages": len(packages),
        "qualified_platform_scope": "ARM64 CPython3.11 client software only",
        "observed_runtime": {
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "desktop_env_instantiated": False,
        "model_instantiated_or_invoked": False,
        "native_tasks": 0,
        "source_checkbox_eligible": False,
    }
    if action == "plan":
        return report
    author_root = Path(author_root).resolve(strict=True)
    if author_root.is_relative_to(ROOT):
        raise ValueError("original author source must remain outside the checkout")
    verify_source(author_root, PROFILE)
    report.update(status="ORIGINAL_SOURCE_VERIFIED_NO_IMPORTS")
    if action == "verify-source":
        return report
    for name, expected in packages.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"client package version mismatch: {name}")
    check_imports(author_root)
    report.update(status="ORIGINAL_CLIENT_AND_FIXED_THREE_GRADER_IMPORTS_PASSED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument(
        "--action", choices=["plan", "verify-source", "check-imports"], default="plan"
    )
    args = parser.parse_args()
    print(json.dumps(run(args.author_root, args.action), indent=2))


if __name__ == "__main__":
    main()
