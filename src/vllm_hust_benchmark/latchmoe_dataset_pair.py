"""Offline LatchMoE evidence gate; not a server launcher or acceptance scorer.

Receipts must be collected from the effective server/worker, never copied from
desired launch settings. Hash checks bind receipts to raw evidence but are not
an attestation of the truth of a self-reported receipt. Human review, live MOD
qualification, dataset-specific scoring and V5.4 release gates remain separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from .mmlu_pro_pyramidkv import score_results
from .frontierscience_inputs import SOURCES as FRONTIER_SOURCES

SCHEMA = "latchmoe-dataset-pair-v1"
DATASETS = (
    "MMLU-Pro",
    "HLE-Verified",
    "SWE-bench-Pro",
    "FrontierScience",
    "Terminal-Bench 2.1",
)
COMMON_FIELDS = {
    "model": (
        "id",
        "revision",
        "config_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "dtype",
    ),
    "runtime": ("core_revision", "seam_revision", "mod_wheel_sha256", "packages"),
    "hardware": ("chip_model", "device_ids", "tp", "nodes"),
    "server": (
        "graph_mode",
        "prefix_caching",
        "cpu_offload_gb",
        "quantization",
        "argv",
    ),
    "dataset": ("provider", "source", "revision", "track", "manifest_sha256", "scope"),
    "evaluator": ("revision", "sha256", "metric"),
    "scaffold": ("id", "sha256"),
    "budgets": ("max_tokens", "concurrency", "tools"),
}

# Public canonical source revisions, not credentials. Changes require review.
SOURCE_IDENTITIES = {
    "MMLU-Pro": (
        "huggingface",
        "TIGER-Lab/MMLU-Pro",
        "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    ),  # pragma: allowlist secret
    "HLE-Verified": (
        "github",
        "SKYLENAGE-AI/HLE-Verified",
        "b705e0fb541c025a1532ce0d60d70ae2f53b00e0",
    ),  # pragma: allowlist secret
    "SWE-bench-Pro": (
        "huggingface",
        "ScaleAI/SWE-bench_Pro",
        "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50",
    ),  # pragma: allowlist secret
    "FrontierScience": (
        "huggingface",
        "openai/frontierscience",
        "25ed67db7da8f4591484e764008ff585544f5a30",
    ),  # pragma: allowlist secret
    "Terminal-Bench 2.1": (
        "github",
        "harbor-framework/terminal-bench-2-1",
        "7131e4375048a0e408a8fb404b5f499d726b695b",
    ),  # pragma: allowlist secret
}


def _validate_values(common: dict, dataset: str, track: str | None) -> None:
    source = common["dataset"]
    if (
        tuple(source[key] for key in ("provider", "source", "revision"))
        != SOURCE_IDENTITIES[dataset]
    ):
        raise ValueError("Dataset label disagrees with pinned source identity")
    if source["track"] != track:
        raise ValueError("Dataset track disagrees with source manifest")
    if (
        dataset == "FrontierScience"
        and source.get("source_sha256") != FRONTIER_SOURCES[track][1]
    ):
        raise ValueError("FrontierScience source SHA disagrees with track")
    packages = common["runtime"]["packages"]
    if (
        not isinstance(packages, dict)
        or not packages
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in packages.items()
        )
    ):
        raise ValueError("Explicit runtime packages required")
    for key, minimum in (("max_tokens", 1), ("concurrency", 1), ("tools", 0)):
        value = common["budgets"][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid frozen budget: {key}")
    if not all(
        isinstance(common[section][key], str) and common[section][key]
        for section, key in (
            ("dataset", "scope"),
            ("evaluator", "metric"),
            ("scaffold", "id"),
        )
    ):
        raise ValueError("Explicit scope, metric and scaffold required")
    hardware = common["hardware"]
    if (
        type(hardware["tp"]) is not int
        or type(hardware["nodes"]) is not int
        or not isinstance(hardware["device_ids"], list)
        or any(type(item) is not int or item < 0 for item in hardware["device_ids"])
    ):
        raise ValueError("Strict integer hardware topology required")
    server = common["server"]
    if (
        type(server["prefix_caching"]) is not bool
        or not isinstance(server["argv"], list)
        or not server["argv"]
        or not all(isinstance(arg, str) and arg for arg in server["argv"])
    ):
        raise ValueError("Explicit server argv and boolean prefix_caching required")
    if any(arg.split("=")[0] == "--enforce-eager" for arg in server["argv"]):
        raise ValueError("Server argv contradicts graph-mode contract")
    if type(server["cpu_offload_gb"]) not in (int, float):
        raise ValueError("Numeric native cpu_offload_gb required")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else _stream_digest(stream)
        )


def _stream_digest(stream) -> str:
    hasher = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(chunk)
    return hasher.hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_contract(contract: dict) -> None:
    if contract.get("schema_version") != SCHEMA:
        raise ValueError("Unknown contract schema")
    dataset, track = contract.get("dataset"), contract.get("track")
    if dataset not in DATASETS:
        raise ValueError("Exact primary dataset required; no near substitutes")
    if (dataset == "FrontierScience" and track not in ("olympiad", "research")) or (
        dataset != "FrontierScience" and track is not None
    ):
        raise ValueError("Separate FrontierScience track required")
    common = contract.get("common", {})
    for section, keys in COMMON_FIELDS.items():
        value = common.get(section)
        if not isinstance(value, dict) or any(key not in value for key in keys):
            raise ValueError(f"Incomplete common contract: {section}")
        for key in keys:
            if key.endswith("sha256") and not _sha(value[key]):
                raise ValueError(f"Invalid SHA-256 in common contract: {section}.{key}")
    _validate_values(common, dataset, track)
    if (
        common["model"]["id"] != "Qwen/Qwen3.5-35B-A3B"
        or common["model"]["dtype"] != "bfloat16"
    ):
        raise ValueError("Pinned Qwen3.5 BF16 model required")
    for section, key in (
        ("model", "revision"),
        ("runtime", "core_revision"),
        ("runtime", "seam_revision"),
        ("evaluator", "revision"),
        ("dataset", "revision"),
    ):
        if (
            not isinstance(common[section][key], str)
            or re.fullmatch(r"[0-9a-f]{40}", common[section][key]) is None
        ):
            raise ValueError("Missing source revision")
    hardware = common["hardware"]
    if (
        hardware["chip_model"] != "910B2"
        or hardware["tp"] != 2
        or hardware["nodes"] != 1
        or not isinstance(hardware["device_ids"], list)
        or len(hardware["device_ids"]) != 2
        or len(set(hardware["device_ids"])) != 2
    ):
        raise ValueError(
            "Qwen3.5 pair requires two distinct 910B2 devices, TP2, one node"
        )
    server = common["server"]
    if server["graph_mode"] not in (
        "PIECEWISE",
        "FULL_DECODE_ONLY",
        "FULL_AND_PIECEWISE",
    ):
        raise ValueError("Graph-mode contract required; eager is diagnostic only")
    if server["cpu_offload_gb"] != 0 or server["quantization"] is not None:
        raise ValueError(
            "Do not replace the native BF16 baseline with CPU offload or quantization"
        )
    requests = contract.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("Frozen requests required")
    ids = []
    for request in requests:
        if not isinstance(request.get("task_id"), str) or not request["task_id"]:
            raise ValueError("Nonempty immutable task ID required")
        if not _sha(request.get("prompt_sha256")) or not _sha(
            request.get("parameters_sha256")
        ):
            raise ValueError("Frozen requests must bind prompt and parameters hashes")
        ids.append(request["task_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate frozen requests")
    config = contract.get("mod_config", {})
    if (
        type(config.get("num_slots")) is not int
        or config["num_slots"] <= 0
        or config.get("indexed_mode") not in ("legacy", "eager", "graph")
        or type(config.get("offload_gb")) not in (int, float)
        or not math.isfinite(config["offload_gb"])
        or not config["offload_gb"] > 0
    ):
        raise ValueError("Explicit MOD configuration required")


def _artifact(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid artifact path")
    path = Path(relative)
    resolved = (root / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or not resolved.is_relative_to(root.resolve())
    ):
        raise ValueError("Artifact path escapes evidence root")
    return resolved


def _verify_arm(contract: dict, label: str, receipt: dict, root: Path) -> dict:
    if receipt.get("arm") != label or receipt.get("contract_sha256") != digest(
        contract
    ):
        raise ValueError(f"{label} label or frozen contract SHA mismatch")
    if digest(receipt.get("common")) != digest(contract["common"]):
        raise ValueError(f"{label} effective common contract differs")
    if digest(receipt.get("requests")) != digest(contract["requests"]):
        raise ValueError(f"{label} effective requests differ")
    expected = {**contract["mod_config"], "enabled": label == "B1"}
    if label == "B0":
        expected["offload_gb"] = 0
    effective = receipt.get("effective_mod")
    if digest(effective) != digest(expected):
        raise ValueError(f"{label} effective MOD activation differs")
    artifacts = receipt.get("artifacts", {})
    if not artifacts:
        raise ValueError("Hashed raw artifacts required")
    selected_bytes = {}
    selected_names = {receipt.get("outcomes_path"), receipt.get("telemetry_path")}
    for name, expected_sha in artifacts.items():
        path = _artifact(root, name)
        if name in selected_names:
            data = path.read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            selected_bytes[name] = data
        else:
            actual_sha = file_digest(path)
        if not _sha(expected_sha) or actual_sha != expected_sha:
            raise ValueError(f"Artifact SHA mismatch: {label}/{name}")
    selected = []
    for key in ("outcomes_path", "telemetry_path"):
        name = receipt.get(key)
        if name not in artifacts:
            raise ValueError(f"Hashed {key} required")
        selected.append(json.loads(selected_bytes[name]))
    outcomes, telemetry = selected
    ids = {item["task_id"] for item in contract["requests"]}
    if (
        not isinstance(outcomes, list)
        or len(outcomes) != len(ids)
        or {row.get("task_id") for row in outcomes} != ids
        or any(type(row.get("success")) is not bool for row in outcomes)
    ):
        raise ValueError("Missing, duplicate or invalid task outcomes")
    if any(not row["success"] and not row.get("error") for row in outcomes):
        raise ValueError("Failed outcomes must retain their errors")
    requests_by_id = {row["task_id"]: row for row in contract["requests"]}
    for outcome in outcomes:
        expected_request = requests_by_id[outcome["task_id"]]
        for key in ("prompt_sha256", "parameters_sha256"):
            if outcome.get(key) != expected_request[key]:
                raise ValueError("Raw outcomes contradict frozen request hashes")
    if (
        telemetry.get("phase") != "measured_requests_only"
        or not telemetry.get("worker_identity")
        or telemetry.get("request_manifest_sha256") != digest(contract["requests"])
    ):
        raise ValueError("telemetry must bind measured requests and worker identity")
    deltas = {}
    for key in ("calls", "waves"):
        before = telemetry.get("before", {}).get(key)
        after = telemetry.get("after", {}).get(key)
        if (
            type(before) is not int
            or type(after) is not int
            or not 0 <= before <= after
        ):
            raise ValueError("Invalid or reset telemetry counters")
        if label == "B0" and (before or after):
            raise ValueError("MOD path appeared in native baseline telemetry")
        deltas[key] = after - before
    if (deltas["calls"] == 0) != (deltas["waves"] == 0):
        raise ValueError("Inconsistent telemetry calls/waves")
    return {"deltas": deltas, "failures": sum(not row["success"] for row in outcomes)}


def verify_pair(contract: dict, arms: dict, roots: dict[str, Path]) -> dict:
    _validate_contract(contract)
    if set(arms) != {"B0", "B1"} or set(roots) != {"B0", "B1"}:
        raise ValueError("Both B0 and B1 evidence required")
    verified = {
        label: _verify_arm(contract, label, arms[label], roots[label])
        for label in ("B0", "B1")
    }
    exercised = verified["B1"]["deltas"]["calls"] > 0
    return {
        "schema_version": SCHEMA,
        "contract_sha256": digest(contract),
        "status": "matched-evidence" if exercised else "not-exercised",
        "mod_exercised": exercised,
        "measured_deltas": {key: value["deltas"] for key, value in verified.items()},
        "failures": {key: value["failures"] for key, value in verified.items()},
        "quality_scores": None,
        "formal_v5_4_accepted": False,
        "limitations": "Offline receipt/hash checks only; requires live qualification, scorer verification and human review.",
    }


def score_mmlu(results: list[dict], cases: list[dict]) -> dict:
    if not isinstance(results, list) or any(
        not isinstance(row, dict) or type(row.get("success")) is not bool
        for row in results
    ):
        raise ValueError("MMLU outcomes require boolean success")
    score = score_results(results, cases)
    for key in ("eligible", "ineligible"):
        score.pop(key)
    for row in score["outcomes"]:
        row.pop("eligible")
    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    verify = sub.add_parser("verify-pair")
    verify.add_argument("--contract", type=Path, required=True)
    for arm in ("b0", "b1"):
        verify.add_argument(
            f"--{arm}",
            type=Path,
            required=True,
            help="Receipt JSON beside its raw artifacts",
        )
    score = sub.add_parser("score-mmlu")
    score.add_argument("--cases", type=Path, required=True)
    score.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "verify-pair":
            contract = json.loads(args.contract.read_text())
            paths = {"B0": args.b0, "B1": args.b1}
            verdict = verify_pair(
                contract,
                {key: json.loads(path.read_text()) for key, path in paths.items()},
                {key: path.parent for key, path in paths.items()},
            )
        else:
            verdict = score_mmlu(
                json.loads(args.results.read_text()), json.loads(args.cases.read_text())
            )
        print(json.dumps(verdict, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(
            json.dumps(
                {"status": "rejected", "error": str(exc), "quality_scores": None}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
