"""Verify original Mooncake traces/tool and plan default storage replay only.

Never writes KV pages, runs inference, invents prompts, or reports performance.
"""

import argparse
import ast
import json
import math
from pathlib import Path
import shutil
import sys

from prepare_xsum_original_assets import sha

ROOT = Path(__file__).resolve().parents[1]
REVISION = "531790322667942f0368933d0d5960f6e4a4c87b"  # pragma: allowlist secret - public author revision
AUTHOR_FILES = {
    "__init__.py": "ff93cdadcd933874fb064dc81e4b2c8219a6b60189b39a73ef152916e8f44600",  # pragma: allowlist secret - public author checksum
    "__main__.py": "ae50cf37c3e673016a8c408c97197edfc9007fd580d655ce9c2270efb3f12618",  # pragma: allowlist secret - public author checksum
    "benchmark.py": "d4a1a2ca44cfcee9921938dfff0fa4d12fdb588ff2ef4c99d6268ed6221a10dc",  # pragma: allowlist secret - public author checksum
    "layout/__init__.py": "7fa600cfd35686cb0c13846f0958301438e360ea1b6ad0e99cdf15b977ba53ec",  # pragma: allowlist secret - public author checksum
    "layout/interface.py": "d2b3864ac5131f20212a0024f5d7130012bf9fa6f643b1b537a7cab00b4f633a",  # pragma: allowlist secret - public author checksum
    "layout/mla.py": "ea7ccfcedd99dcc07be0319fa09a4291b7c59fd81d7efaf35e4c9e5c3b827e58",  # pragma: allowlist secret - public author checksum
    "storage/__init__.py": "92f5fda01c4b7f09c4b6ef090bf8f50e9f76b1463ddd1f4339ddd11d3f94d04e",  # pragma: allowlist secret - public author checksum
    "storage/disk.py": "eac86c04c7b7367d2ecdb47505c88d663c020961bd0fe855c57bad0da3943527",  # pragma: allowlist secret - public author checksum
    "storage/interface.py": "2d7dbed632987de92b1c52744746f7f58cf14cbd6b6f334ba288ca68b6e6935e",  # pragma: allowlist secret - public author checksum
}


def audit_author(author):
    author = Path(author).resolve()
    if {p.relative_to(author).as_posix() for p in author.rglob("*.py")} != set(
        AUTHOR_FILES
    ):
        raise ValueError("exact pinned storage benchmark module inventory required")
    for relative, expected in AUTHOR_FILES.items():
        path = author / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(author)
            or sha(path) != expected
        ):
            raise ValueError("fixed official source identity mismatch")
    assignments = [
        node
        for node in ast.parse((author / "layout/mla.py").read_text("utf-8")).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MLA_MODEL_CONFIG"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("original model layout metadata required")
    config = ast.literal_eval(assignments[0].value)["glm5"]
    return (
        512
        * (
            config["kv_lora_rank"]
            + config["qk_rope_head_dim"]
            + config["index_head_dim"]
        )
        * config["kv_precision_bytes"]
        * config["num_layers"]
    )


def trace_summary(root, spec, count):
    root = Path(root).resolve()
    path = root / spec["path"]
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or path.stat().st_size != spec["bytes"]
        or sha(path) != spec["sha256"]
    ):
        raise ValueError("unchanged registered original trace required")
    total, maximum, selected_max, selected_ops, selected_tokens = 0, 0, 0, 0, 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or type(row.get("timestamp")) not in {int, float}
                or not math.isfinite(row["timestamp"])
                or row["timestamp"] < 0
                or any(
                    type(row.get(key)) is not int or row[key] < 0
                    for key in ("input_length", "output_length")
                )
                or not isinstance(row.get("hash_ids"), list)
                or any(type(index) is not int or index < 0 for index in row["hash_ids"])
            ):
                raise ValueError("original timing/length/hash trace contract required")
            current = max(row["hash_ids"], default=0)
            maximum = max(maximum, current)
            if count is None or total < count:
                selected_max = max(selected_max, current)
                selected_ops += len(row["hash_ids"])
                selected_tokens += row["input_length"] + row["output_length"]
            total += 1
    if (
        total != spec["rows"]
        or (count is not None and count > total)
        or sha(path) != spec["sha256"]
    ):
        raise ValueError("original trace rows/identity or requested count mismatch")
    return {
        "path": spec["path"],
        "sha256": spec["sha256"],
        "original_rows": total,
        "selected_original_rows": total if count is None else count,
        "full_trace_max_hash_id": maximum,
        "selected_max_hash_id": selected_max,
        "expected_selected_page_operations": selected_ops,
        "expected_selected_total_tokens": selected_tokens,
    }


def build_plan(author, traces, storage, *, source, count=3):
    if source.get("revision") != REVISION or source.get("source_id") != "MOONCAKE":
        raise ValueError("fixed original Mooncake source revision required")
    if count is not None and (type(count) is not int or count < 3):
        raise ValueError(
            "fixed sample requires at least three original requests per file"
        )
    author, traces, storage = [
        Path(path).resolve() for path in (author, traces, storage)
    ]
    if (
        storage.exists()
        or not storage.parent.is_dir()
        or any(
            storage.is_relative_to(root) or root.is_relative_to(storage)
            for root in (author, traces)
        )
    ):
        raise ValueError("fresh storage root separate from original assets required")
    page_bytes = audit_author(author)
    expected = {
        "conversation_trace.jsonl": "conversation",
        "toolagent_trace.jsonl": "toolagent",
    }
    specs = {item["path"]: item for item in source["files"] if item["role"] == "data"}
    if set(specs) != set(expected):
        raise ValueError("both registered original FAST25 trace files required")
    jobs = []
    for filename, scenario in expected.items():
        info = trace_summary(traces, specs[filename], count)
        command = [
            sys.executable,
            str(author / "benchmark.py"),
            "--scenario",
            scenario,
            "--trace-dir",
            str(traces),
            "--storage-dir",
            str(storage / scenario),
            "--model",
            "glm5",
            "--page-size-tokens",
            "512",
            "--max-pages",
            "2000",
            "--threads",
            "1",
            "--file-mode",
            "single",
            "--fsync-mode",
            "none",
            "--replay-scales",
            "0",
        ]
        if count is not None:
            command += ["--max-requests", str(count)]
        jobs.append(
            {
                **info,
                "scenario": scenario,
                "command": command,
                "full_trace_collision_free_bytes": (info["full_trace_max_hash_id"] + 1)
                * page_bytes,
                "selected_requests_use_modulo_mapping": info["selected_max_hash_id"]
                >= 2000,
            }
        )
    return {
        "source_id": "MOONCAKE",
        "status": "OFFICIAL_STORAGE_REPLAY_PLAN_ONLY",
        "author_revision": REVISION,
        "jobs": jobs,
        "page_bytes": page_bytes,
        "max_pages_per_scenario": 2000,
        "retained_storage_capacity_bytes": len(jobs) * 2000 * page_bytes,
        "model_preset_is_layout_not_deployed_model": True,
        "original_prompts_available": False,
        "model_calls": 0,
        "storage_writes_executed": False,
        "performance_claim": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-tool", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=3)
    selection.add_argument("--full", action="store_true")
    args = parser.parse_args()
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )["sources"]["MOONCAKE"]
    plan = build_plan(
        args.author_tool,
        args.trace_root,
        args.storage_root,
        source=source,
        count=None if args.full else args.count,
    )
    plan["currently_observed_free_bytes"] = shutil.disk_usage(
        args.storage_root.resolve().parent
    ).free
    plan["default_profile_capacity_fits_observed_free_space"] = (
        plan["currently_observed_free_bytes"] >= plan["retained_storage_capacity_bytes"]
    )
    output = args.output.resolve()
    if any(
        output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
        for path in (args.author_tool, args.trace_root, args.storage_root)
    ):
        parser.error(
            "fresh plan output separate from source and future storage roots required"
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / "plan-private.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "scenarios": len(plan["jobs"]),
                "storage_writes_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
