"""Run the unchanged original Python tool only inside an offline nonroot container.

This is an execution-boundary adapter, not an Agent, scorer, or source delivery.
The caller must retain the original one-code-call and five-iteration budgets.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys


TOOL_SHA256 = "111faf2de81f3e597e33b9ffe4bd9b1b19cf05c374ecac622acbd6b0ce918e98"  # pragma: allowlist secret - original pinned publisher Python tool


def validate_isolation(*, container, uid, status, interfaces, readonly):
    if not container or uid <= 0 or not readonly:
        raise RuntimeError("offline nonroot read-only Docker required")
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    if (
        int(fields.get("CapEff", "1").strip(), 16) != 0
        or fields.get("NoNewPrivs", "0").strip() != "1"
        or set(interfaces) != {"lo"}
    ):
        raise RuntimeError(
            "no capabilities, no-new-privileges and network-none required"
        )


def check_container():
    validate_isolation(
        container=Path("/.dockerenv").is_file(),
        uid=os.geteuid(),
        status=Path("/proc/self/status").read_text(),
        interfaces=[name for _, name in socket.if_nameindex()],
        readonly=bool(os.statvfs("/").f_flag & os.ST_RDONLY),
    )


def original_tool(source):
    source = Path(source).resolve(strict=True)
    path = (source / "agents/coder_utils.py").resolve(strict=True)
    if (
        not path.is_relative_to(source)
        or hashlib.sha256(path.read_bytes()).hexdigest() != TOOL_SHA256
    ):
        raise ValueError("unchanged original coder tool required")
    if not os.statvfs(source).f_flag & os.ST_RDONLY:
        raise RuntimeError("original source must be mounted read-only")
    spec = importlib.util.spec_from_file_location(
        "discoverybench_original_coder_tool", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MAX_TURNS != 1:
        raise ValueError("original one-code-call limit required")
    return module.CustomPythonAstREPLTool()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    check_container()  # Must precede importing publisher code or reading code input.
    payload = json.load(sys.stdin)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"code"}
        or not isinstance(payload["code"], str)
    ):
        raise ValueError("one explicit original tool code string required")
    result = original_tool(args.source)._run(payload["code"])
    if not isinstance(result, str):
        raise TypeError("original Python tool string result required")
    print(json.dumps({"tool_output": result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
