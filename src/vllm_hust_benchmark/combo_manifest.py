from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "ascend-l1-perf-smoke/v1"
DEFAULT_BENCHMARK_PROFILE = "ascend-l1-qwen25-3b-dummy-v1"
DEFAULT_PAIRING_STRATEGY = "single-repo-pr-with-peer-main"
DEFAULT_REPO_SLUGS = {
    "vllm-hust": "vLLM-HUST/vllm-hust",
    "vllm-ascend-hust": "vLLM-HUST/vllm-ascend-hust",
}
DEFAULT_BENCHMARK_CONFIG: dict[str, Any] = {
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "load_format": "dummy",
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "server": {
        "max_model_len": 2048,
        "max_num_seqs": 8,
        "enforce_eager": True,
    },
    "latency": {
        "input_len": 1024,
        "output_len": 128,
        "warmup": 5,
        "measurement": 10,
    },
    "throughput": {
        "dataset": "random",
        "input_len": 1024,
        "output_len": 128,
        "num_prompts": 100,
        "random_batch_size": 1,
    },
    "serve": {
        "scenario": "random-online",
        "num_prompts": 100,
        "request_rate": 4,
        "max_concurrency": 4,
        "endpoint": "/v1/completions",
    },
}


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _normalize_json_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def fingerprint_payload(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _normalize_override_path(path: str) -> list[str]:
    return [segment.strip().replace("-", "_") for segment in path.split(".") if segment.strip()]


def apply_benchmark_config_overrides(
    benchmark_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = copy.deepcopy(dict(benchmark_config))
    for raw_path, raw_value in (overrides or {}).items():
        path = _normalize_override_path(raw_path)
        if not path:
            raise ValueError("Empty benchmark_config override path")
        cursor: dict[str, Any] = updated
        for key in path[:-1]:
            node = cursor.get(key)
            if not isinstance(node, dict):
                raise ValueError(
                    f"Invalid benchmark_config override path: {raw_path}"
                )
            cursor = node
        cursor[path[-1]] = raw_value
    return updated


def _build_repo_ref(
    *,
    repo: str,
    ref: str,
    commit: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "repo": repo,
        "ref": ref,
    }
    if commit:
        payload["commit"] = commit
    return payload


def build_source_combo(
    *,
    trigger_repo: str,
    trigger_ref: str,
    peer_ref: str,
    pairing_strategy: str = DEFAULT_PAIRING_STRATEGY,
    trigger_commit: str | None = None,
    peer_commit: str | None = None,
    trigger_repo_slug: str | None = None,
    peer_repo_slug: str | None = None,
) -> dict[str, Any]:
    if trigger_repo not in DEFAULT_REPO_SLUGS:
        raise ValueError(f"Unsupported trigger_repo: {trigger_repo}")

    peer_repo = "vllm-ascend-hust" if trigger_repo == "vllm-hust" else "vllm-hust"
    trigger_slug = trigger_repo_slug or DEFAULT_REPO_SLUGS[trigger_repo]
    peer_slug = peer_repo_slug or DEFAULT_REPO_SLUGS[peer_repo]

    source_combo = {
        "pairing_strategy": pairing_strategy,
        "trigger_repo": trigger_repo,
    }
    source_combo[trigger_repo.replace("-", "_")] = _build_repo_ref(
        repo=trigger_slug,
        ref=trigger_ref,
        commit=trigger_commit,
    )
    source_combo[peer_repo.replace("-", "_")] = _build_repo_ref(
        repo=peer_slug,
        ref=peer_ref,
        commit=peer_commit,
    )
    return source_combo


def build_combo_manifest(
    *,
    trigger_repo: str,
    trigger_ref: str,
    peer_ref: str,
    pairing_strategy: str = DEFAULT_PAIRING_STRATEGY,
    trigger_commit: str | None = None,
    peer_commit: str | None = None,
    trigger_repo_slug: str | None = None,
    peer_repo_slug: str | None = None,
    benchmark_profile: str = DEFAULT_BENCHMARK_PROFILE,
    benchmark_config: Mapping[str, Any] | None = None,
    benchmark_config_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = apply_benchmark_config_overrides(
        benchmark_config or DEFAULT_BENCHMARK_CONFIG,
        benchmark_config_overrides,
    )
    source_combo = build_source_combo(
        trigger_repo=trigger_repo,
        trigger_ref=trigger_ref,
        peer_ref=peer_ref,
        pairing_strategy=pairing_strategy,
        trigger_commit=trigger_commit,
        peer_commit=peer_commit,
        trigger_repo_slug=trigger_repo_slug,
        peer_repo_slug=peer_repo_slug,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_profile": benchmark_profile,
        "benchmark_config": config,
        "benchmark_config_fingerprint": fingerprint_payload(config),
        "source_combo": source_combo,
        "source_combo_fingerprint": fingerprint_payload(source_combo),
    }


def validate_combo_manifest(manifest: Mapping[str, Any]) -> None:
    required_top_level = {
        "schema_version",
        "benchmark_profile",
        "benchmark_config",
        "benchmark_config_fingerprint",
        "source_combo",
        "source_combo_fingerprint",
    }
    missing = sorted(required_top_level.difference(manifest))
    if missing:
        raise ValueError(f"combo manifest missing required keys: {', '.join(missing)}")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported combo manifest schema_version: {manifest.get('schema_version')}"
        )

    benchmark_config = manifest.get("benchmark_config")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("combo manifest benchmark_config must be a JSON object")

    source_combo = manifest.get("source_combo")
    if not isinstance(source_combo, Mapping):
        raise ValueError("combo manifest source_combo must be a JSON object")
    trigger_repo = source_combo.get("trigger_repo")
    if trigger_repo not in DEFAULT_REPO_SLUGS:
        raise ValueError(f"Unsupported combo manifest trigger_repo: {trigger_repo}")

    trigger_key = str(trigger_repo).replace("-", "_")
    peer_key = (
        "vllm_ascend_hust" if trigger_key == "vllm_hust" else "vllm_hust"
    )
    for key in (trigger_key, peer_key):
        repo_ref = source_combo.get(key)
        if not isinstance(repo_ref, Mapping):
            raise ValueError(f"combo manifest source_combo missing repo payload: {key}")
        if not repo_ref.get("repo") or not repo_ref.get("ref"):
            raise ValueError(f"combo manifest repo payload incomplete: {key}")

    expected_benchmark_config_fingerprint = fingerprint_payload(benchmark_config)
    if manifest.get("benchmark_config_fingerprint") != expected_benchmark_config_fingerprint:
        raise ValueError(
            "combo manifest benchmark_config_fingerprint does not match benchmark_config"
        )

    expected_source_combo_fingerprint = fingerprint_payload(source_combo)
    if manifest.get("source_combo_fingerprint") != expected_source_combo_fingerprint:
        raise ValueError(
            "combo manifest source_combo_fingerprint does not match source_combo"
        )


def load_combo_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"combo manifest must be a JSON object: {path}")
    validate_combo_manifest(manifest)
    return manifest


def write_combo_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    validate_combo_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_json_value(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_runtime_engine(manifest: Mapping[str, Any]) -> str:
    source_combo = manifest.get("source_combo")
    if not isinstance(source_combo, Mapping):
        raise ValueError("combo manifest source_combo must be a JSON object")
    trigger_repo = source_combo.get("trigger_repo")
    if trigger_repo not in DEFAULT_REPO_SLUGS:
        raise ValueError(f"Unsupported combo manifest trigger_repo: {trigger_repo}")
    return str(trigger_repo)
