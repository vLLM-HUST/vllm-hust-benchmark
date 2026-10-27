from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "source_tools/validate_paio_assets.py"
SPEC = importlib.util.spec_from_file_location("paio_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def spec(count=1, format="wrapped_request"):
    return {
        "dataset_id": "FIXTURE",
        "path": "fixture.jsonl",
        "format": format,
        "expected_count": count,
    }


def wrapped(body):
    return {
        "model_name": "fixture-model",
        "request_type": "chat",
        "request_body": json.dumps(body),
    }


def write_fixture(tmp_path, rows):
    path = tmp_path / "fixture.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_duplicate_identity_is_request_body_not_wrapper(tmp_path):
    body = {"messages": [{"role": "user", "content": "fixture words"}], "max_tokens": 4}
    rows = [wrapped(body), {**wrapped(body), "model_name": "other-wrapper"}]
    path = write_fixture(tmp_path, rows)
    result = audit.validate_asset(tmp_path, spec(2))
    assert result["record_count"] == result["valid_record_count"] == 2
    assert result["duplicate_request_count"] == 1
    assert result["duplicate_rate"] == 0.5
    assert result["sha256"] == audit.digest(path.read_bytes())
    assert result["byte_size"] == path.stat().st_size


def test_constructed_prompt_hash_and_group_aggregation(tmp_path):
    prompt = "fixture content"
    row = {
        "prompt": prompt,
        "prompt_hash": audit.digest(prompt.encode())[:16],
        "messages": [{"role": "user", "content": prompt}],
        "group_id": "private-group",
        "metadata": {"output_tokens_target": 8},
        "created_at_ms": 12,
    }
    write_fixture(tmp_path, [row])
    result = audit.validate_asset(tmp_path, spec(format="constructed_prompt"))
    assert result["prompt_hash_checked_count"] == 1
    assert result["prompt_hash_mismatch_count"] == 0
    assert result["output_target_distribution"]["max"] == 8
    assert result["session_or_group_structure"]["group_id"]["unique_groups"] == 1
    assert "private-group" not in json.dumps(result)
    assert prompt not in json.dumps(result)


def test_hash_mismatch_is_not_admitted(tmp_path):
    write_fixture(
        tmp_path,
        [
            {
                "prompt": "x",
                "prompt_hash": "incorrect",
                "messages": [{"role": "user", "content": "x"}],
            }
        ],
    )
    result = audit.validate_asset(tmp_path, spec(format="generated_long_text"))
    assert result["prompt_hash_mismatch_count"] == 1
    assert result["admission_status"] == "REVIEW_REQUIRED"


def test_parse_errors_and_blank_lines_are_explicit(tmp_path):
    (tmp_path / "fixture.jsonl").write_text(
        "not json\n\n"
        + json.dumps(
            {"model_name": "x", "request_type": "x", "request_body": "not json"}
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit.validate_asset(tmp_path, spec(2))
    assert result["outer_parse_error_count"] == 1
    assert result["inner_parse_error_count"] == 1
    assert result["blank_line_count"] == 1
    assert result["valid_record_count"] == 0
    assert result["admission_status"] == "REVIEW_REQUIRED"


def test_nonfinite_json_is_rejected(tmp_path):
    (tmp_path / "fixture.jsonl").write_text('{"request_body":NaN}\n', encoding="utf-8")
    result = audit.validate_asset(tmp_path, spec())
    assert result["outer_parse_error_count"] == 1


def test_private_values_and_unknown_field_names_not_exported(tmp_path):
    secret = (
        "private.fixture@example.com"  # pragma: allowlist secret - invented fixture
    )
    write_fixture(
        tmp_path,
        [
            wrapped(
                {
                    "messages": [{"role": "user", "content": secret}],
                    "private-field-name": "private-value",
                }
            )
        ],
    )
    result = audit.validate_asset(tmp_path, spec())
    encoded = json.dumps(result)
    assert secret not in encoded
    assert "private-field-name" not in encoded
    assert "private-value" not in encoded
    assert (
        result["sensitive_information_scan"]["record_match_counts"]["email_like"] == 1
    )


def test_escape_from_explicit_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="under the explicit data root"):
        audit.validate_asset(tmp_path, {**spec(), "path": "../outside.jsonl"})


def test_asset_catalog_covers_exactly_eight_named_payloads():
    catalog = json.loads(
        (SCRIPT.parents[1] / "data/catalogs/dataset_assets_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(catalog["assets"]) == 8
    assert len({a["dataset_id"] for a in catalog["assets"]}) == 8
    assert sum(a["expected_count"] for a in catalog["assets"]) == 26500
    assert all(not Path(a["path"]).name.startswith("._") for a in catalog["assets"])
