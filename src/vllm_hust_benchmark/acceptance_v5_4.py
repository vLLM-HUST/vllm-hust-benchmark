from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[2]
DECLARATION_PATH = REPO_ROOT / "src/vllm_hust_benchmark/data/acceptance_v5_4.json"
SCHEMA_PATH = REPO_ROOT / "schemas/acceptance_v5_4.schema.json"


def load_and_validate(root: Path = REPO_ROOT) -> dict[str, Any]:
    declaration = json.loads(
        (root / DECLARATION_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    )
    schema = json.loads(
        (root / SCHEMA_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(declaration)

    source = root / declaration["source_document"]
    if not source.is_file():
        raise ValueError(f"missing V5.4 semantic mirror: {source}")
    if source.stat().st_size != declaration["source_size_bytes"]:
        raise ValueError("V5.4 semantic mirror size mismatch")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != declaration["source_sha256"]:
        raise ValueError("V5.4 semantic mirror checksum mismatch")

    if len({item["id"] for item in declaration["optional_deliveries"]}) != 3:
        raise ValueError("optional delivery IDs must be unique")
    def contains_b2(value: Any) -> bool:
        if isinstance(value, dict):
            return any(key == "B2" or contains_b2(item) for key, item in value.items())
        if isinstance(value, list):
            return any(contains_b2(item) for item in value)
        return value == "B2"

    if contains_b2(declaration):
        raise ValueError("V5.4 permits only B0 and B1")
    return declaration


def build_measurement_queue(declaration: dict[str, Any]) -> list[dict[str, str]]:
    """Build only mandatory B0/B1 placeholders; inactive optional work never queues."""
    if any(item["active"] for item in declaration["optional_deliveries"]):
        raise ValueError("V5.4 optional deliveries must remain inactive")
    return [
        {"dataset": dataset, "baseline_role": role, "test_plan_version": "V5.4"}
        for dataset in declaration["primary_datasets"]
        for role in declaration["baseline_roles"]["allowed"]
    ]
