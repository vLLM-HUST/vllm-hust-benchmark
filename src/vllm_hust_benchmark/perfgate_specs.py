from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.registry import get_scenario


@dataclass(frozen=True)
class PerfgateSpecEntry:
    scenario: str
    hardware_chip_model: str
    spec_file: str


def _normalize_scenario(value: str) -> str:
    return str(value or "").strip()


def _normalize_chip_model(value: str) -> str:
    return str(value or "").strip().upper()


def _load_default_registry_payload() -> dict[str, Any]:
    with (
        resources.files("vllm_hust_benchmark.data")
        .joinpath("perfgate_spec_registry.json")
        .open("r", encoding="utf-8") as handle
    ):
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("perfgate spec registry must be a JSON object")
    return payload


def _load_registry_payload(registry_file: Path | None = None) -> dict[str, Any]:
    if registry_file is None:
        return _load_default_registry_payload()
    payload = json.loads(registry_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{registry_file}: perfgate spec registry must be a JSON object")
    return payload


def _validate_spec_file_path(spec_file: str) -> None:
    path = Path(spec_file)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"perfgate spec file must be repo-relative: {spec_file}")
    if not spec_file.endswith(".json"):
        raise ValueError(f"perfgate spec file must be a JSON file: {spec_file}")


def load_perfgate_spec_registry(
    registry_file: Path | None = None,
) -> tuple[PerfgateSpecEntry, ...]:
    payload = _load_registry_payload(registry_file)
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("perfgate spec registry must contain an entries array")

    entries: list[PerfgateSpecEntry] = []
    seen: set[tuple[str, str]] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("perfgate spec registry entries must be objects")

        scenario = _normalize_scenario(raw_entry.get("scenario", ""))
        hardware_chip_model = _normalize_chip_model(
            str(raw_entry.get("hardware_chip_model", ""))
        )
        spec_file = str(raw_entry.get("spec_file") or "").strip()
        if not scenario:
            raise ValueError("perfgate spec registry entry is missing scenario")
        if not hardware_chip_model:
            raise ValueError(
                "perfgate spec registry entry is missing hardware_chip_model"
            )
        if not spec_file:
            raise ValueError("perfgate spec registry entry is missing spec_file")

        try:
            get_scenario(scenario)
        except KeyError as exc:
            raise ValueError(
                f"perfgate spec registry references unknown scenario: {scenario}"
            ) from exc

        _validate_spec_file_path(spec_file)

        key = (scenario, hardware_chip_model)
        if key in seen:
            raise ValueError(
                "duplicate perfgate spec registry entry for "
                f"scenario={scenario}, hardware_chip_model={hardware_chip_model}"
            )
        seen.add(key)
        entries.append(
            PerfgateSpecEntry(
                scenario=scenario,
                hardware_chip_model=hardware_chip_model,
                spec_file=spec_file,
            )
        )

    return tuple(entries)


def _load_spec_payload(spec_path: Path) -> dict[str, Any]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_path}: perfgate spec must be a JSON object")
    return payload


def _validate_resolved_spec(
    entry: PerfgateSpecEntry,
    *,
    repo_root: Path,
) -> Path:
    spec_path = (repo_root / entry.spec_file).resolve()
    resolved_repo_root = repo_root.resolve()
    if not spec_path.is_relative_to(resolved_repo_root):
        raise ValueError(f"resolved perfgate spec escapes repo root: {entry.spec_file}")
    if not spec_path.is_file():
        raise ValueError(f"perfgate spec file not found: {spec_path}")

    spec = _load_spec_payload(spec_path)
    spec_id = str(spec.get("id") or "").strip()
    spec_scenario = _normalize_scenario(str(spec.get("scenario") or ""))
    spec_chip = _normalize_chip_model(str(spec.get("hardware_chip_model") or ""))
    if not spec_id:
        raise ValueError(f"{spec_path}: perfgate spec is missing required field: id")
    if spec_scenario != entry.scenario:
        raise ValueError(
            f"{spec_path}: scenario mismatch; registry={entry.scenario}, "
            f"spec={spec_scenario}"
        )
    if spec_chip != entry.hardware_chip_model:
        raise ValueError(
            f"{spec_path}: hardware_chip_model mismatch; "
            f"registry={entry.hardware_chip_model}, spec={spec_chip}"
        )
    return spec_path


def format_supported_pairs(entries: tuple[PerfgateSpecEntry, ...]) -> str:
    pairs = sorted(
        f"{entry.scenario}/{entry.hardware_chip_model}" for entry in entries
    )
    return ", ".join(pairs)


def resolve_perfgate_spec_file(
    *,
    scenario: str,
    hardware_chip_model: str,
    repo_root: Path | None = None,
    registry_file: Path | None = None,
) -> Path:
    normalized_scenario = _normalize_scenario(scenario)
    normalized_chip = _normalize_chip_model(hardware_chip_model)
    if not normalized_scenario:
        raise ValueError("scenario is required")
    if not normalized_chip:
        raise ValueError("hardware_chip_model is required")

    entries = load_perfgate_spec_registry(registry_file)
    for entry in entries:
        if (
            entry.scenario == normalized_scenario
            and entry.hardware_chip_model == normalized_chip
        ):
            if repo_root is None:
                return Path(entry.spec_file)
            return _validate_resolved_spec(entry, repo_root=repo_root)

    supported = format_supported_pairs(entries)
    raise ValueError(
        "No perfgate spec registered for "
        f"scenario={normalized_scenario}, hardware_chip_model={normalized_chip}. "
        f"Supported pairs: {supported}"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a perfgate spec file from scenario and hardware chip.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("--scenario", required=True)
    resolve_parser.add_argument("--hardware-chip-model", required=True)
    resolve_parser.add_argument("--repo-root", type=Path)
    resolve_parser.add_argument("--registry-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "resolve":
            spec_file = resolve_perfgate_spec_file(
                scenario=args.scenario,
                hardware_chip_model=args.hardware_chip_model,
                repo_root=args.repo_root,
                registry_file=args.registry_file,
            )
            print(spec_file)
            return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(f"unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
