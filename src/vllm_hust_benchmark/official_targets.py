from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.same_spec import PREFIX_REPETITION_DEFAULT_NUM_PREFIXES

SCHEMA_VERSION = "official-target-registry/v1"
REGISTRY_VERSION = "1.3.5"
EFFECTIVE_FROM = "2026-08-16"
PUBLIC_TEXT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
PUBLIC_CODE_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"
PUBLIC_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
PUBLIC_TEXT_SCENARIOS = {
    "agent-research-online",
    "prefix-repetition-online",
    "random-latency",
    "random-online",
    "sharegpt-online",
    "sharegpt-throughput",
    "sonnet-throughput",
}
# Public-leaderboard targets are defined on the official Ascend 910B2 fleet.
# Records measured on other hardware (e.g. 910B3) must be classified as
# specialty/provisional even when the scenario/model/chip-count match a
# public text/code/vision target shape.
PUBLIC_TEXT_HARDWARE = "910B2"
PUBLIC_CODE_HARDWARE = "910B2"
PUBLIC_VISION_HARDWARE = "910B2"
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

    if path.name.startswith("perfgate-"):
        if "Coder" in model:
            profile = "perfgate-code"
        elif "VL" in model:
            profile = "perfgate-multimodal"
        else:
            profile = "perfgate-text"
        return "perfgate", "provisional", profile

    if chip_count == 1 and model == PUBLIC_TEXT_MODEL:
        if (
            scenario in PUBLIC_TEXT_SCENARIOS
            and spec.get("hardware_chip_model") == PUBLIC_TEXT_HARDWARE
        ):
            return "public-leaderboard", "active", "core-text"
        return "specialty", "provisional", "specialty-text"
    if (
        chip_count == 1
        and model == PUBLIC_CODE_MODEL
        and scenario == "instructcoder-online"
        and spec.get("hardware_chip_model") == PUBLIC_CODE_HARDWARE
    ):
        return "public-leaderboard", "active", "code"
    if (
        chip_count == 1
        and model == PUBLIC_VISION_MODEL
        and scenario == "visionarena-online"
        and spec.get("hardware_chip_model") == PUBLIC_VISION_HARDWARE
    ):
        return "public-leaderboard", "active", "multimodal"
    if chip_count > 1:
        return "specialty", "provisional", "multi-chip"
    return "specialty", "provisional", "specialty"


def _validate_public_target(spec: dict[str, Any], path: Path) -> None:
    server = spec["server_parameters"]
    public_hardware = {
        PUBLIC_TEXT_HARDWARE,
        PUBLIC_CODE_HARDWARE,
        PUBLIC_VISION_HARDWARE,
    }
    if str(spec["hardware_chip_model"]) not in public_hardware:
        raise ValueError(
            f"public target {path} must run on 910B2 hardware, "
            f"got {spec['hardware_chip_model']!r}"
        )
    expected = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": (
            0.9 if spec.get("scenario") == "prefix-repetition-online" else 0.6
        ),
        "max_model_len": 32768,
    }
    for name, value in expected.items():
        if server.get(name) != value:
            raise ValueError(
                f"public target {path} requires {name}={value!r}, "
                f"got {server.get(name)!r}"
            )
    if int(spec["chip_count"]) != 1 or int(spec["node_count"]) != 1:
        raise ValueError(f"public target must be single-node/single-chip: {path}")
    if str(spec["model_precision"]) != "FP16":
        raise ValueError(f"public target must use FP16: {path}")

    fixed_output_scenarios = {"agent-research-online", "prefix-repetition-online"}
    if (
        spec.get("scenario") in fixed_output_scenarios
        and spec["client_parameters"].get("ignore_eos") is not True
    ):
        raise ValueError(
            f"fixed-output public target {path} must set ignore_eos=true so "
            "the declared output length is an executable measurement contract"
        )

    baseline = spec["baseline_target"]
    if baseline.get("vllm_ref") != "v0.18.0":
        raise ValueError(f"public target must use vLLM v0.18.0: {path}")
    if baseline.get("vllm_ascend_ref") != "v0.18.0":
        raise ValueError(f"public target must use vLLM-Ascend v0.18.0: {path}")


def _validate_prefix_repetition_workload(spec: dict[str, Any], path: Path) -> None:
    client = spec["client_parameters"]
    if client.get("dataset_name") != "prefix_repetition":
        return

    num_prompts = int(client.get("num_prompts", 0))
    num_prefixes = int(
        client.get(
            "prefix_repetition_num_prefixes",
            PREFIX_REPETITION_DEFAULT_NUM_PREFIXES,
        )
    )
    if num_prompts < 1 or num_prefixes < 1 or num_prompts // num_prefixes < 2:
        raise ValueError(
            f"prefix-repetition target {path} must reuse each prefix at least twice: "
            f"num_prompts={num_prompts}, num_prefixes={num_prefixes}"
        )


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


def _target_contract_metadata(
    history: dict[str, Any], target_id: str
) -> dict[str, Any]:
    """Resolve the immutable contract metadata for one target.

    ``official_target_versions.json`` historically versioned the whole source
    set.  New entries may add a ``target_contracts`` mapping containing only
    changed targets plus an optional ``default`` migration entry. Walking the
    history backwards lets unchanged targets inherit their previous contract
    while a changed target records its own version and supersession.
    """
    for version in reversed(history["versions"]):
        contracts = version.get("target_contracts")
        if not isinstance(contracts, dict):
            continue
        metadata = contracts.get(target_id) or contracts.get("default")
        if not isinstance(metadata, dict):
            continue
        resolved = dict(metadata)
        resolved.setdefault("version", version["version"])
        resolved.setdefault("effective_from", version["effective_from"])
        supersedes = resolved.get("supersedes")
        if supersedes is None:
            resolved["supersedes"] = []
        elif isinstance(supersedes, str):
            resolved["supersedes"] = [supersedes]
        elif not isinstance(supersedes, list) or not all(
            isinstance(item, str) for item in supersedes
        ):
            raise ValueError(f"invalid supersedes metadata for target {target_id!r}")
        return resolved
    return {
        "version": REGISTRY_VERSION,
        "effective_from": EFFECTIVE_FROM,
        "supersedes": [],
    }


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
    history_path = repo_root / VERSION_HISTORY_RELATIVE_PATH
    history = (
        _load_version_history(repo_root) if history_path.exists() else {"versions": []}
    )
    for path in spec_paths:
        spec = _load_spec(path)
        _validate_prefix_repetition_workload(spec, path)
        intended_use, status, profile = _classify_spec(path, spec)
        if status == "active":
            _validate_public_target(spec, path)

        baseline = spec["baseline_target"]
        export = spec["export"]
        relative_path = path.relative_to(repo_root).as_posix()
        source_sha256 = _sha256_bytes(path.read_bytes())
        contract = _target_contract_metadata(history, str(spec["id"]))
        targets.append(
            {
                "target_id": spec["id"],
                "target_version": str(contract["version"]),
                "status": status,
                "effective_from": str(contract["effective_from"]),
                "supersedes": contract["supersedes"],
                "profile": profile,
                "intended_use": intended_use,
                "baseline_runtime": {
                    "engine": baseline["engine"],
                    "engine_version": baseline["engine_version"],
                    "github_repository": baseline["github_repository"],
                    "vllm_ref": baseline["vllm_ref"],
                    "vllm_ascend_ref": baseline["vllm_ascend_ref"],
                    "git_commit": export.get("git_commit"),
                },
                "model": {
                    "id": spec["model"],
                    "parameters": spec.get("model_parameters"),
                    "precision": spec["model_precision"],
                },
                "hardware": {
                    "vendor": spec["hardware_vendor"],
                    "chip_model": spec["hardware_chip_model"],
                    "chip_count": spec["chip_count"],
                    "node_count": spec["node_count"],
                },
                "server_parameters": spec["server_parameters"],
                "workload": {
                    "name": spec["scenario"],
                    "client_parameters": spec["client_parameters"],
                },
                "source_spec": {
                    "path": relative_path,
                    "sha256": source_sha256,
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
        "registry_generation": {
            "version": REGISTRY_VERSION,
            "effective_from": EFFECTIVE_FROM,
            "source_set_sha256": source_set_sha256,
        },
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
        "本 registry 只描述工程健康、公开排行榜和专项覆盖，不定义 V4.6 A1-A4 正式验收。"
        if chinese
        else "This registry covers engineering health, public leaderboard, and specialty runs. "
        "It does not define formal V4.6 A1-A4 acceptance."
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
        (
            "`registry_version` identifies this generated registry snapshot. Each target's "
            "`target_version` is the immutable execution-contract version for that target; "
            "an unrelated target update must not change it. Producers should record the "
            "canonical `target_contract_id` and `target_contract_version` metadata fields."
            if not chinese
            else "`registry_version` 表示本次生成的 registry 快照版本；每个 target 的 "
            "`target_version` 表示该 target 不可变的执行契约版本。无关 target 的更新不应 "
            "改变它。生产端应记录 canonical 的 `target_contract_id` 和 "
            "`target_contract_version` 字段。"
        ),
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
        path.write_text(expected, encoding="utf-8")
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
