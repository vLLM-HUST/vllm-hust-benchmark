from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "official-target-registry/v1"
REGISTRY_VERSION = "1.8.0"
EFFECTIVE_FROM = "2026-08-03"
PUBLIC_TEXT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
PUBLIC_CODE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
PUBLIC_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PUBLIC_TRACE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
PUBLIC_MODEL_REVISIONS = {
    PUBLIC_TEXT_MODEL: "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",  # pragma: allowlist secret
    PUBLIC_CODE_MODEL: "aedcc2d42b622764e023cf882b6652e646b95671",  # pragma: allowlist secret
    PUBLIC_VISION_MODEL: "cc594898137f460bfe9f0759e9844b3ce807cfb5",  # pragma: allowlist secret
    PUBLIC_TRACE_MODEL: "711ad2ea6aa40cfca18895e8aca02ab92df1a746",  # pragma: allowlist secret
}
V018_VLLM_COMMIT = (
    "bcf2be96120005e9aea171927f85055a6a5c0cf6"  # pragma: allowlist secret
)
V018_VLLM_ASCEND_COMMIT = (
    "e18643f8a4d5bd9990727654318ad069ea0b56e2"  # pragma: allowlist secret
)
V018_RUNTIME_CONFIG_DIGEST = (
    "sha256:9a50c7c633d52e2514593e5021a776572c35be734465ce02d41cd1481240fd31"
)
V018_RUNTIME_ARCHIVE_SHA256 = (
    "sha256:d28174deca2a8a28173cff1425a585b843421d7e43c8bb34b6c8a1622c289cc0"
)
V018_CONTAINERD_STORAGE_MANIFEST_DIGEST = (
    "sha256:5f80f602b9460f3a02f9e847edbe69576906e3dd60e200a095a763a4792f5c26"
)
V018_RUNTIME_PACKAGES = {
    "vllm": "0.18.0+empty",
    "vllm-ascend": "0.18.0",
    "datasets": "3.3.0",
    "xxhash": "3.6.0",
    "torch": "2.9.0+cpu",
    "torch-npu": "2.9.0.post1+gitee7ba04",
}
PUBLIC_TRACE_MODEL_REVISION = (
    "711ad2ea6aa40cfca18895e8aca02ab92df1a746"  # pragma: allowlist secret
)
PUBLIC_TRACE_VLLM_REF = "v0.22.1"
PUBLIC_TRACE_VLLM_ASCEND_REF = "v0.22.1rc1"
PUBLIC_TRACE_VLLM_COMMIT = (
    "0decac0d96c42b49572498019f0a0e3600f50398"  # pragma: allowlist secret
)
PUBLIC_TRACE_VLLM_ASCEND_COMMIT = (
    "5f6faa0cb8830f667266f3b8121cd1383606f2a1"  # pragma: allowlist secret
)
PUBLIC_TRACE_RUNTIME_IMAGE_DIGEST = (
    "sha256:bfc46fa57aedf933e6d6d4adcf42ce96aed956689018faf111bb01571891e092"
)
PUBLIC_TRACE_RUNTIME_IMAGE = (
    f"quay.io/ascend/vllm-ascend@{PUBLIC_TRACE_RUNTIME_IMAGE_DIGEST}"
)
PUBLIC_TRACE_RUNTIME_PACKAGES = {
    "transformers": "5.5.4",
    "huggingface-hub": "1.21.0",
    "click": "8.4.1",
    "vllm": "0.22.1+empty",
    "vllm-ascend": "0.22.1rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
}
PUBLIC_TRACE_RUNTIME_ENVIRONMENT = {"VLLM_BATCH_INVARIANT": "1"}
PUBLIC_TRACE_ADDITIONAL_CONFIG = {
    "ascend_compilation_config": {"fuse_norm_quant": False}
}
PUBLIC_TRACE_COMPILATION_CONFIG = {"cudagraph_mode": "PIECEWISE"}
PUBLIC_TRACE_SCENARIOS = {
    "burstgpt-production-replay",
    "tracelab-coding-agent-replay",
}
SIMLLM_WORKLOAD_IDS = {
    "simllm-random-online-warm-cache",
    "simllm-saturated-throughput-warm-cache",
}
SIMLLM_VLLM_HUST_COMMIT = (
    "f229ba7cad21a4dba58681af6738a9fd947388e2"  # pragma: allowlist secret
)
SIMLLM_VLLM_ASCEND_HUST_COMMIT = (
    "590855422839c1e885eee19339b7a015687215e5"  # pragma: allowlist secret
)
SIMLLM_RUNTIME_IMAGE_DIGEST = (
    "sha256:105834a38766a6b1b89a7eeb313a37351d098a69e8cdee87ad0ca3a6e090ce13"
)
SIMLLM_RUNTIME_IMAGE = f"quay.io/ascend/vllm-ascend@{SIMLLM_RUNTIME_IMAGE_DIGEST}"
SIMLLM_RUNTIME_PACKAGES = {
    "vllm": "0.21.0+empty",
    "vllm-ascend": "0.21.0rc1",
    "torch": "2.10.0+cpu",
    "torch-npu": "2.10.0",
    "transformers": "5.5.4",
    "huggingface-hub": "1.19.0",
    "click": "8.4.1",
}
CORE_PUBLIC_TARGET_VERSION = "1.4.0"
PUBLIC_TRACE_TARGET_VERSIONS = {
    "burstgpt-production-replay": "1.5.1",
    "tracelab-coding-agent-replay": "1.6.2",
}
SIMLLM_TARGET_VERSION = "1.6.2"
PUBLIC_TEXT_SCENARIOS = {
    "agent-research-online",
    "prefix-repetition-online",
    "random-latency",
    "random-online",
    "sharegpt-online",
    "sharegpt-throughput",
    "sonnet-throughput",
}
PACKAGE_REGISTRY_PATH = Path(__file__).parent / "data" / "official_targets.json"
VERSION_HISTORY_RELATIVE_PATH = (
    Path("src") / "vllm_hust_benchmark" / "data" / "official_target_versions.json"
)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"spec must be a JSON object: {path}")
    required = {
        "id",
        "baseline_target",
        "scenario",
        "model",
        "model_precision",
        "hardware_vendor",
        "hardware_chip_model",
        "chip_count",
        "node_count",
        "server_parameters",
        "client_parameters",
        "export",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"spec {path} is missing: {', '.join(missing)}")
    return payload


def _classify_spec(path: Path, spec: dict[str, Any]) -> tuple[str, str, str]:
    scenario = str(spec["scenario"])
    model = str(spec["model"])
    chip_count = int(spec["chip_count"])
    workload_id = str(spec.get("workload_id") or "")

    if workload_id in SIMLLM_WORKLOAD_IDS:
        return "specialty", "active", "simllm-warm-cache"

    if path.name.startswith("perfgate-"):
        if "Coder" in model:
            profile = "perfgate-code"
        elif "VL" in model:
            profile = "perfgate-multimodal"
        else:
            profile = "perfgate-text"
        return "perfgate", "provisional", profile

    if chip_count == 1 and model == PUBLIC_TEXT_MODEL:
        if scenario in PUBLIC_TEXT_SCENARIOS:
            return "public-leaderboard", "active", "core-text"
        return "specialty", "provisional", "specialty-text"
    if (
        chip_count == 2
        and model == PUBLIC_TRACE_MODEL
        and scenario in PUBLIC_TRACE_SCENARIOS
    ):
        return "public-leaderboard", "active", "production-trace"
    if (
        chip_count == 1
        and model == PUBLIC_CODE_MODEL
        and scenario == "instructcoder-online"
    ):
        return "public-leaderboard", "active", "code"
    if (
        chip_count == 1
        and model == PUBLIC_VISION_MODEL
        and scenario == "visionarena-online"
    ):
        return "public-leaderboard", "active", "multimodal"
    if chip_count > 1:
        return "specialty", "provisional", "multi-chip"
    return "specialty", "provisional", "specialty"


def _validate_public_target(spec: dict[str, Any], path: Path) -> None:
    server = spec["server_parameters"]
    expected_model_revision = PUBLIC_MODEL_REVISIONS[spec["model"]]
    if spec.get("model_revision") != expected_model_revision:
        raise ValueError(
            f"public target must pin model_revision={expected_model_revision}: {path}"
        )
    if server.get("revision") != expected_model_revision:
        raise ValueError(
            f"public target server must pin revision={expected_model_revision}: {path}"
        )
    data_identity = spec.get("data_identity")
    if not isinstance(data_identity, dict) or not data_identity.get("kind"):
        raise ValueError(f"public target must pin data_identity: {path}")
    if spec["scenario"] in PUBLIC_TRACE_SCENARIOS:
        expected = {
            "tensor_parallel_size": 2,
            "gpu_memory_utilization": 0.92,
            "max_model_len": 131072,
        }
    else:
        expected = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 32768,
        }
    for name, value in expected.items():
        if server.get(name) != value:
            raise ValueError(
                f"public target {path} requires {name}={value!r}, "
                f"got {server.get(name)!r}"
            )
    expected_chip_count = 2 if spec["scenario"] in PUBLIC_TRACE_SCENARIOS else 1
    if int(spec["chip_count"]) != expected_chip_count or int(spec["node_count"]) != 1:
        raise ValueError(
            f"public target must be single-node/{expected_chip_count}-chip: {path}"
        )
    expected_precision = (
        "BF16" if spec["scenario"] in PUBLIC_TRACE_SCENARIOS else "FP16"
    )
    if str(spec["model_precision"]) != expected_precision:
        raise ValueError(f"public target must use {expected_precision}: {path}")

    baseline = spec["baseline_target"]
    if spec["scenario"] in PUBLIC_TRACE_SCENARIOS:
        if baseline.get("vllm_ref") != PUBLIC_TRACE_VLLM_REF:
            raise ValueError(
                f"production-trace target must use vLLM {PUBLIC_TRACE_VLLM_REF}: {path}"
            )
        if baseline.get("vllm_ascend_ref") != PUBLIC_TRACE_VLLM_ASCEND_REF:
            raise ValueError(
                "production-trace target must use vLLM-Ascend "
                f"{PUBLIC_TRACE_VLLM_ASCEND_REF}: {path}"
            )
        if server.get("revision") != PUBLIC_TRACE_MODEL_REVISION:
            raise ValueError(
                f"production-trace target must pin the DeepSeek revision: {path}"
            )
        if server.get("additional_config") != PUBLIC_TRACE_ADDITIONAL_CONFIG:
            raise ValueError(
                "production-trace target must pin the audited Ascend compilation "
                f"configuration: {path}"
            )
        if server.get("compilation_config") != PUBLIC_TRACE_COMPILATION_CONFIG:
            raise ValueError(
                f"production-trace target must pin the audited compilation mode: {path}"
            )
        if spec["client_parameters"].get("cohort_context_cap") != server.get(
            "max_model_len"
        ):
            raise ValueError(
                f"production-trace cohort cap must match max_model_len: {path}"
            )
        if baseline.get("vllm_commit") != PUBLIC_TRACE_VLLM_COMMIT:
            raise ValueError(
                f"production-trace target must pin the vLLM commit: {path}"
            )
        if baseline.get("vllm_ascend_commit") != PUBLIC_TRACE_VLLM_ASCEND_COMMIT:
            raise ValueError(
                f"production-trace target must pin the vLLM-Ascend commit: {path}"
            )
        if baseline.get("runtime_packages") != PUBLIC_TRACE_RUNTIME_PACKAGES:
            raise ValueError(
                f"production-trace target must pin the runtime packages: {path}"
            )
        if baseline.get("runtime_environment") != PUBLIC_TRACE_RUNTIME_ENVIRONMENT:
            raise ValueError(
                f"production-trace target must pin the audited runtime environment: {path}"
            )
        if baseline.get("runtime_image") != PUBLIC_TRACE_RUNTIME_IMAGE:
            raise ValueError(
                f"production-trace target must pin the official runtime image: {path}"
            )
        digest = str(baseline.get("runtime_image_digest") or "")
        if digest != PUBLIC_TRACE_RUNTIME_IMAGE_DIGEST or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", digest
        ):
            raise ValueError(
                f"production-trace target must pin the runtime image digest: {path}"
            )
        if not str(baseline["runtime_image"]).endswith(f"@{digest}"):
            raise ValueError(f"production-trace runtime image/digest mismatch: {path}")
    else:
        if baseline.get("vllm_ref") != "v0.18.0":
            raise ValueError(f"public target must use vLLM v0.18.0: {path}")
        if baseline.get("vllm_ascend_ref") != "v0.18.0":
            raise ValueError(f"public target must use vLLM-Ascend v0.18.0: {path}")
        expected_runtime = {
            "vllm_commit": V018_VLLM_COMMIT,
            "vllm_ascend_commit": V018_VLLM_ASCEND_COMMIT,
            "runtime_transport": "docker-archive",
            "runtime_image": None,
            # Compatibility field used by existing attestation/orchestration.
            # It identifies the runnable image config, not a registry manifest.
            "runtime_image_digest": V018_RUNTIME_CONFIG_DIGEST,
            "runtime_config_digest": V018_RUNTIME_CONFIG_DIGEST,
            "runtime_archive_sha256": V018_RUNTIME_ARCHIVE_SHA256,
            "containerd_storage_manifest_digest": (
                V018_CONTAINERD_STORAGE_MANIFEST_DIGEST
            ),
            "runtime_packages": V018_RUNTIME_PACKAGES,
        }
        for name, expected_value in expected_runtime.items():
            if baseline.get(name) != expected_value:
                raise ValueError(
                    f"public v0.18 target must pin {name}={expected_value!r}: {path}"
                )


def _validate_simllm_target(spec: dict[str, Any], path: Path) -> None:
    workload_id = str(spec.get("workload_id") or "")
    if workload_id not in SIMLLM_WORKLOAD_IDS:
        return
    if spec.get("scenario") != "random-online":
        raise ValueError(
            f"SimLLM target must use random-online as its base scenario: {path}"
        )
    if spec.get("model") != PUBLIC_TEXT_MODEL or spec.get("model_precision") != "FP16":
        raise ValueError(f"SimLLM target must use Qwen2.5-14B FP16: {path}")
    if int(spec.get("chip_count") or 0) != 1 or int(spec.get("node_count") or 0) != 1:
        raise ValueError(f"SimLLM target must be single-node/single-chip: {path}")
    if spec.get("hardware_chip_model") != "910B2":
        raise ValueError(f"SimLLM target must use Ascend 910B2: {path}")
    server = spec.get("server_parameters") or {}
    if (
        server.get("tensor_parallel_size") != 1
        or server.get("gpu_memory_utilization") != 0.6
    ):
        raise ValueError(
            f"SimLLM target must pin TP1 and gpu_memory_utilization=0.6: {path}"
        )
    baseline = spec.get("baseline_target") or {}
    if baseline.get("engine") != "vllm-hust":
        raise ValueError(f"SimLLM target must use vllm-hust: {path}")
    if baseline.get("vllm_commit") != SIMLLM_VLLM_HUST_COMMIT:
        raise ValueError(f"SimLLM target must pin the vllm-hust commit: {path}")
    if baseline.get("vllm_ascend_commit") != SIMLLM_VLLM_ASCEND_HUST_COMMIT:
        raise ValueError(f"SimLLM target must pin the vllm-ascend-hust commit: {path}")
    if baseline.get("runtime_image") != SIMLLM_RUNTIME_IMAGE:
        raise ValueError(f"SimLLM target must pin the verified runtime image: {path}")
    if baseline.get("runtime_image_digest") != SIMLLM_RUNTIME_IMAGE_DIGEST:
        raise ValueError(f"SimLLM target must pin the verified image digest: {path}")
    if baseline.get("runtime_packages") != SIMLLM_RUNTIME_PACKAGES:
        raise ValueError(
            f"SimLLM target must pin the verified runtime packages: {path}"
        )
    protocol = spec.get("ab_protocol") or {}
    required_protocol = {
        "schema_version": "simllm-ab-protocol/v1",
        "baseline_variant": "simllm-disabled",
        "candidate_variant": "simllm-enabled-warm-cache",
        "baseline_engine": "vllm-hust",
        "candidate_engine": "vllm-hust-simllm",
        "minimum_independent_repetitions": 3,
        "aggregation": "median",
        "maximum_primary_metric_cv_percent": 5,
        "setting_signature_required": True,
        "local_reference_results_are_official": False,
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"SimLLM A/B protocol requires {key}={expected!r}: {path}")
    expected_cache_size = (
        200 if workload_id == "simllm-random-online-warm-cache" else 32
    )
    expected_simllm_config = {
        "cosine_threshold": 0.8,
        "lsh_num_bits": 64,
        "lsh_batch_threshold": 32,
        "kv_cache_size": expected_cache_size,
        "sandwich_bottom": 3,
        "sandwich_top": 3,
        "unmatched_store_mode": "top",
    }
    if protocol.get("simllm_config") != expected_simllm_config:
        raise ValueError(
            f"SimLLM target must pin simllm_config={expected_simllm_config!r}: {path}"
        )
    if spec["server_parameters"].get("prefix_caching_hash_algo") != "sha256":
        raise ValueError(
            "SimLLM immutable runtime requires prefix_caching_hash_algo='sha256': "
            f"{path}"
        )
    warmup = protocol.get("candidate_warmup") or {}
    if (
        warmup.get("passes") != 1
        or warmup.get("restart_before_measurement") is not False
    ):
        raise ValueError(
            f"SimLLM candidate must use one in-process warm-cache pass: {path}"
        )
    if warmup.get("same_requests_as_measurement") is not True:
        raise ValueError(f"SimLLM warmup must reuse the measured request set: {path}")


def _source_set_sha256(targets: list[dict[str, Any]]) -> str:
    sources = sorted(
        (target["source_spec"] for target in targets), key=lambda source: source["path"]
    )
    canonical = json.dumps(sources, sort_keys=True, separators=(",", ":")) + "\n"
    return _sha256_bytes(canonical.encode("utf-8"))


def _load_version_history(repo_root: Path) -> dict[str, Any]:
    path = repo_root / VERSION_HISTORY_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "official-target-version-history/v1":
        raise ValueError(f"invalid official target version history: {path}")
    versions = payload.get("versions")
    if not isinstance(versions, list) or not versions:
        raise ValueError(f"official target version history is empty: {path}")
    return payload


def _validate_version_history(
    repo_root: Path, *, source_set_sha256: str
) -> dict[str, Any]:
    history = _load_version_history(repo_root)
    current = history["versions"][-1]
    if current.get("version") != REGISTRY_VERSION:
        raise ValueError("REGISTRY_VERSION must match the latest version history entry")
    if current.get("effective_from") != EFFECTIVE_FROM:
        raise ValueError("EFFECTIVE_FROM must match the latest version history entry")
    if current.get("source_set_sha256") != source_set_sha256:
        raise ValueError(
            "official specs changed without a new registry version; add a version "
            "history entry and update REGISTRY_VERSION"
        )
    return history


def build_registry(repo_root: Path) -> dict[str, Any]:
    spec_root = repo_root / "docs" / "official-baselines"
    spec_paths = sorted(
        path
        for path in spec_root.glob("*.json")
        if path.name != "official-ascend-constraints.stub.json"
    )
    if not spec_paths:
        raise ValueError(f"no official target specs found under {spec_root}")

    targets: list[dict[str, Any]] = []
    for path in spec_paths:
        spec = _load_spec(path)
        intended_use, status, profile = _classify_spec(path, spec)
        _validate_simllm_target(spec, path)
        if status == "active" and intended_use == "public-leaderboard":
            _validate_public_target(spec, path)

        baseline = spec["baseline_target"]
        export = spec["export"]
        relative_path = path.relative_to(repo_root).as_posix()
        targets.append(
            {
                "target_id": spec["id"],
                # Target contracts are versioned independently from the registry
                # container. Adding or hardening the production-trace profile must
                # not invalidate unchanged, already-attested core baselines.
                "target_version": (
                    PUBLIC_TRACE_TARGET_VERSIONS[spec["scenario"]]
                    if profile == "production-trace"
                    else SIMLLM_TARGET_VERSION
                    if profile == "simllm-warm-cache"
                    else CORE_PUBLIC_TARGET_VERSION
                    if status == "active"
                    else REGISTRY_VERSION
                ),
                "status": status,
                "effective_from": EFFECTIVE_FROM,
                "supersedes": [],
                "profile": profile,
                "intended_use": intended_use,
                "baseline_runtime": {
                    "engine": baseline["engine"],
                    "engine_version": baseline["engine_version"],
                    "github_repository": baseline["github_repository"],
                    "vllm_ref": baseline["vllm_ref"],
                    "vllm_ascend_ref": baseline["vllm_ascend_ref"],
                    "git_commit": export.get("git_commit"),
                    "core_commit": baseline.get("vllm_commit"),
                    "backend_commit": baseline.get("vllm_ascend_commit"),
                    "runtime_image": baseline.get("runtime_image"),
                    "runtime_image_digest": baseline.get("runtime_image_digest"),
                    "runtime_transport": baseline.get("runtime_transport"),
                    "runtime_config_digest": baseline.get("runtime_config_digest"),
                    "runtime_archive_sha256": baseline.get("runtime_archive_sha256"),
                    "containerd_storage_manifest_digest": baseline.get(
                        "containerd_storage_manifest_digest"
                    ),
                    "runtime_packages": baseline.get("runtime_packages"),
                    "runtime_environment": baseline.get("runtime_environment"),
                },
                "model": {
                    "id": spec["model"],
                    "parameters": spec.get("model_parameters"),
                    "precision": spec["model_precision"],
                    **(
                        {"revision": spec["model_revision"]}
                        if spec.get("model_revision")
                        else {}
                    ),
                },
                "hardware": {
                    "vendor": spec["hardware_vendor"],
                    "chip_model": spec["hardware_chip_model"],
                    "chip_count": spec["chip_count"],
                    "node_count": spec["node_count"],
                },
                "server_parameters": spec["server_parameters"],
                "workload": {
                    "name": spec.get("workload_id") or spec["scenario"],
                    "base_scenario": spec["scenario"],
                    "client_parameters": spec["client_parameters"],
                    "data_identity": spec.get("data_identity"),
                    "protocol": spec.get("ab_protocol"),
                },
                "source_spec": {
                    "path": relative_path,
                    "sha256": _sha256_bytes(path.read_bytes()),
                },
                "compatibility_policy": {
                    "model": "exact",
                    "hardware": "exact",
                    "server_parameters": "exact",
                    "workload_parameters": "exact",
                    "exceptions": "new-target-version-required",
                },
            }
        )

    use_order = {"public-leaderboard": 0, "perfgate": 1, "specialty": 2}
    targets.sort(
        key=lambda target: (
            use_order[target["intended_use"]],
            target["profile"],
            target["workload"]["name"],
            target["hardware"]["chip_count"],
        )
    )
    source_set_sha256 = _source_set_sha256(targets)
    _validate_version_history(repo_root, source_set_sha256=source_set_sha256)
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "effective_from": EFFECTIVE_FROM,
        "source_set_sha256": source_set_sha256,
        "targets": targets,
    }


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_document(registry: dict[str, Any], *, language: str) -> str:
    import mdformat

    chinese = language == "zh-CN"
    title = "vLLM-HUST 官方固定靶" if chinese else "vLLM-HUST Official Targets"
    generated = (
        "本文件由 official specs 自动生成，请勿手工修改。"
        if chinese
        else "This file is generated from the official specs. Do not edit it manually."
    )
    public_note = (
        "公开排行榜固定靶与 3B 快速门禁是不同契约；provisional 记录不得作为公开成果对比。"
        if chinese
        else "Public leaderboard targets and 3B perfgate profiles are separate contracts. "
        "Provisional entries are not valid public-result comparison targets."
    )
    headers = (
        [
            "用途",
            "状态",
            "Profile",
            "Workload",
            "模型",
            "硬件",
            "精度",
            "显存比例",
            "最大长度",
            "Spec",
        ]
        if chinese
        else [
            "Use",
            "Status",
            "Profile",
            "Workload",
            "Model",
            "Hardware",
            "Precision",
            "Memory util.",
            "Max length",
            "Spec",
        ]
    )
    lines = [
        f"# {title}",
        "",
        f"> {generated}",
        "",
        f"- Registry version: `{registry['registry_version']}`",
        f"- Effective from: `{registry['effective_from']}`",
        "",
        public_note,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for target in registry["targets"]:
        server = target["server_parameters"]
        hardware = target["hardware"]
        row = [
            target["intended_use"],
            target["status"],
            target["profile"],
            target["workload"]["name"],
            target["model"]["id"],
            f"{hardware['chip_model']} × {hardware['chip_count']}",
            target["model"]["precision"],
            _format_value(server.get("gpu_memory_utilization")),
            _format_value(server.get("max_model_len")),
            f"[`{Path(target['source_spec']['path']).name}`](../{target['source_spec']['path']})",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            (
                "机器可读快照：[`leaderboard-data/official-targets.json`](../leaderboard-data/official-targets.json)"
                if chinese
                else "Machine-readable snapshot: [`leaderboard-data/official-targets.json`](../leaderboard-data/official-targets.json)"
            ),
            "",
        ]
    )
    return mdformat.text("\n".join(lines), options={"wrap": 100})


def render_changelog(repo_root: Path) -> str:
    import mdformat

    history = _load_version_history(repo_root)
    lines = [
        "# Official target registry changelog",
        "",
        "> Generated from `official_target_versions.json`. Do not edit manually.",
        "",
    ]
    for version in reversed(history["versions"]):
        lines.extend(
            [
                f"## {version['version']} — {version['effective_from']}",
                "",
                f"- English: {version['summary_en']}",
                f"- 中文：{version['summary_zh']}",
                f"- Source set: `{version['source_set_sha256']}`",
                f"- Supersedes: `{version['supersedes'] or 'none'}`",
                "",
            ]
        )
    return mdformat.text("\n".join(lines), options={"wrap": 100})


def generated_outputs(repo_root: Path) -> dict[Path, str]:
    registry = build_registry(repo_root)
    registry_text = _json_text(registry)
    digest = _sha256_bytes(registry_text.encode("utf-8"))
    return {
        repo_root / "leaderboard-data" / "official-targets.json": registry_text,
        repo_root / "leaderboard-data" / "official-targets.sha256": (
            f"{digest}  official-targets.json\n"
        ),
        repo_root
        / "src"
        / "vllm_hust_benchmark"
        / "data"
        / "official_targets.json": registry_text,
        repo_root / "docs" / "OFFICIAL_TARGETS.md": render_document(
            registry, language="en"
        ),
        repo_root / "docs" / "OFFICIAL_TARGETS.zh-CN.md": render_document(
            registry, language="zh-CN"
        ),
        repo_root / "docs" / "OFFICIAL_TARGETS_CHANGELOG.md": render_changelog(
            repo_root
        ),
    }


def write_generated_outputs(repo_root: Path, *, check: bool) -> None:
    stale: list[str] = []
    for path, expected in generated_outputs(repo_root).items():
        if check:
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            if actual != expected:
                stale.append(path.relative_to(repo_root).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="\n")
    if stale:
        raise SystemExit(
            "official target outputs are stale; regenerate: " + ", ".join(stale)
        )


def load_packaged_registry() -> dict[str, Any]:
    return json.loads(PACKAGE_REGISTRY_PATH.read_text(encoding="utf-8"))


def render_active_targets(registry: dict[str, Any]) -> str:
    active = [target for target in registry["targets"] if target["status"] == "active"]
    lines = [
        "USE                PROFILE      WORKLOAD                    MODEL",
    ]
    for target in active:
        lines.append(
            f"{target['intended_use']:<18} "
            f"{target['profile']:<12} "
            f"{target['workload']['name']:<27} "
            f"{target['model']['id']}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or generate official targets."
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generate or args.check:
        repo_root = (args.repo_root or Path.cwd()).resolve()
        write_generated_outputs(repo_root, check=args.check)
        print(
            "official target outputs: ok"
            if args.check
            else "official target outputs: generated"
        )
        return 0

    registry = load_packaged_registry()
    if args.as_json:
        print(_json_text(registry), end="")
    else:
        print(render_active_targets(registry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
