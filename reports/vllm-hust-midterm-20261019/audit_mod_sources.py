"""Capture immutable public source receipts for same-workload MOD screening."""

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
REPOS = [
    "vllm-hust-prefill-microflow",
    "vllm-ascend-split-batch-hust",
    "vllm-hust-dla",
    "request-throttling-controller",
    "vllm-ascend-kvcompress-hust",
    "vllm-hust-bidkv",
    "pegaflow-hust",
    "vllm-ascend-pyramidkv-hust",
    "vllm-hust-hybrid-spec",
]
IMPLEMENTATIONS = {
    "vllm-hust-bidkv": ["src/bidkv/adapters/vllm_hust/selector.py"],
    "pegaflow-hust": ["python/pegaflow/connector/__init__.py"],
    "vllm-hust-dla": ["src/dla/policy.py"],
}


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def capture():
    rows = []
    for name in REPOS:
        repo = f"vLLM-HUST/{name}"
        meta = api(f"repos/{repo}")
        head = api(f"repos/{repo}/commits/{meta['default_branch']}")["sha"]
        tree = api(f"repos/{repo}/git/trees/{head}?recursive=1")["tree"]
        paths = [
            row["path"]
            for row in tree
            if row["type"] == "blob"
            and (
                row["path"] == "README.md"
                or row["path"] in IMPLEMENTATIONS.get(name, [])
                or (
                    row["path"].startswith("src/")
                    and ("manifest" in row["path"] or "host-contract" in row["path"])
                )
                and row["path"].endswith(".json")
            )
        ]
        sources = []
        for path in paths:
            data = api(f"repos/{repo}/contents/{path}?ref={head}")
            raw = base64.b64decode(data["content"])
            sources.append(
                {
                    "path": path,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "text": raw.decode(),
                    "url": f"https://github.com/{repo}/blob/{head}/{path}",
                }
            )
        rows.append(
            {
                "repository": repo,
                "default_branch": meta["default_branch"],
                "commit": head,
                "archived": meta["archived"],
                "sources": sources,
            }
        )
        print(repo, head, len(sources), flush=True)
    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Source admission screening, not runtime effectiveness",
        "repositories": rows,
    }
    evidence = Path("/root/gennova-evidence/20261019-midterm-aligned")
    local_paths = [
        "payload/betterscale/models/qwen35/seat_scheduler.py",
        "payload/betterscale/models/qwen35/fair_schedule.py",
    ]
    result["measured_host_sources"] = [
        {
            "path": name,
            "sha256": hashlib.sha256((evidence / name).read_bytes()).hexdigest(),
            "text": (evidence / name).read_text(),
        }
        for name in local_paths
    ]
    (ROOT / "b1-mod-source-audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    capture()
