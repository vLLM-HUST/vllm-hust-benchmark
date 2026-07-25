import json
from pathlib import Path

from jsonschema import Draft7Validator


ROOT = Path(__file__).parent
SCHEMA_PATH = ROOT.parent / "schemas" / "leaderboard_trend_v1.schema.json"
FIXTURE_ROOT = ROOT / "fixtures" / "trend_coverage"


def _validator() -> Draft7Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schema_accepts_all_supported_trend_states_and_legacy_payload() -> None:
    validator = _validator()
    for path in sorted((FIXTURE_ROOT / "valid").glob("*.json")):
        errors = list(validator.iter_errors(_payload(path)))
        assert errors == [], f"{path.name}: {errors}"


def test_schema_rejects_missing_or_inconsistent_conditional_fields() -> None:
    validator = _validator()
    for path in sorted((FIXTURE_ROOT / "invalid").glob("*.json")):
        errors = list(validator.iter_errors(_payload(path)))
        assert errors, f"expected {path.name} to fail schema validation"


def test_schema_accepts_a_payload_array() -> None:
    validator = _validator()
    payloads = [_payload(FIXTURE_ROOT / "valid" / name) for name in ("full-matrix.json", "experimental.json")]
    assert list(validator.iter_errors(payloads)) == []


def test_schema_requires_aggregate_for_non_experimental_repeated_entry() -> None:
    validator = _validator()
    payload = _payload(FIXTURE_ROOT / "valid" / "full-matrix.json")
    del payload["canonical_aggregate"]
    assert list(validator.iter_errors(payload))

