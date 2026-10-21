#!/usr/bin/env python3
"""Validate leaderboard data against JSON Schema (MVP)."""

import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:
    print("❌ Error: jsonschema library not found")
    print("Install with: pip install jsonschema")
    sys.exit(1)


def load_schema() -> dict:
    """Load the JSON Schema file."""
    schema_path = Path(__file__).parent / "schemas" / "leaderboard_v1.schema.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(file_path: str) -> dict | list[dict]:
    """Load data file to validate."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_data(data: dict | list[dict], schema: dict) -> tuple[bool, str]:
    """Validate data against schema and return readable diagnostics."""
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.path))
    if not errors:
        return True, "✅ Validation passed"

    first = errors[0]
    return False, f"❌ Validation failed: {first.message}\nPath: {list(first.path)}"


def count_entries(data: dict | list[dict]) -> int:
    if isinstance(data, list):
        return len(data)
    return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <data_file> [data_file ...]")
        print("\nExamples:")
        print("  python validate_schema.py data/examples/single_node_example.json")
        print(
            "  python validate_schema.py data/leaderboard_single.json data/leaderboard_multi.json"
        )
        sys.exit(1)

    data_files = sys.argv[1:]

    print("📦 Loading schema...")
    schema = load_schema()

    has_error = False
    for data_file in data_files:
        print(f"\n📄 Loading data: {data_file}")
        data = load_data(data_file)

        print("🔍 Validating...")
        is_valid, message = validate_data(data, schema)
        print(message)

        if is_valid:
            print(f"   entries: {count_entries(data)}")
        else:
            has_error = True

    sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
