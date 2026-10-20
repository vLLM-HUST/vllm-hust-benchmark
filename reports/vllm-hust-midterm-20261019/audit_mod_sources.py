"""Capture immutable public source receipts for same-workload MOD screening."""

import base64
import argparse
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

EXTENDED_REPOS = [
    "vllm-hust-vSpec",
    "vllm-hust-opset",
    "vllm-hust-opfuse",
    "vllm-hust-unified-comm",
    "vllm-ascend-layered-prefill-hust",
    "stateaxis-async-kv-transfer",
    "stateaxis-automatic-prefix-cache",
    "vllm-hust-qos-scheduler",
    "vllm-hust-activation-sparsity",
    "vllm-ascend-hust-LatchMoE",
    "ascend-distributed-metadata",
    "vllm-hust-kv-tiering",
    "vllm-ascend-simllm-hust",
]
EXTENDED_IMPLEMENTATIONS = {
    "vllm-hust-vSpec": ["src/vllm_hust_vspec/backends/mtp.py"],
    "vllm-hust-hybrid-spec": [
        "tools/gdn_prepare_fastpath.py",
        "tools/gdn_prepare_control.py",
    ],
}


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def capture(output=None, extended=False):
    rows = []
    for name in REPOS + (EXTENDED_REPOS if extended else []):
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
                or (
                    extended
                    and row["path"] in ("MOD_METADATA.json", "HOST_CONTRACT.md")
                )
                or row["path"] in IMPLEMENTATIONS.get(name, [])
                or (extended and row["path"] in EXTENDED_IMPLEMENTATIONS.get(name, []))
                or (
                    row["path"].startswith("src/")
                    and (
                        "manifest" in row["path"]
                        or "host-contract" in row["path"]
                        or "vllm-hust-extension" in row["path"]
                    )
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
    if extended:
        local_paths += [
            "payload/betterscale/models/qwen35/integration.py",
            "payload/betterscale/models/qwen35/draft_sampling.py",
            "payload/betterscale/models/qwen35/mamba_abi.py",
            "payload/betterscale/models/qwen35/device_apc.py",
        ]
    result["measured_host_sources"] = [
        {
            "path": name,
            "sha256": hashlib.sha256((evidence / name).read_bytes()).hexdigest(),
            "text": (evidence / name).read_text(),
        }
        for name in local_paths
    ]
    (output or ROOT / "b1-mod-source-audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    if args.extended and args.output is None:
        parser.error("--extended requires a new --output; preserve earlier receipts")
    capture(args.output, args.extended)
