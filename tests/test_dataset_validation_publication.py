from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.dataset_validation import (
    DatasetValidationError,
    validate_artifact,
    validate_index,
    validate_publication,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "leaderboard-data" / "dataset-validation"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_checked_in_publication_is_valid_and_complete() -> None:
    assert validate_publication(PUBLICATION) == {"scenarios": 2, "results": 175}


def test_publication_documents_match_json_schemas() -> None:
    index_schema = load_json(
        ROOT / "schemas" / "dataset_validation_index_v1.schema.json"
    )
    artifact_schema = load_json(ROOT / "schemas" / "dataset_validation_v1.schema.json")
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    jsonschema.Draft202012Validator(index_schema).validate(index)
    for scenario in index["scenarios"]:
        jsonschema.Draft202012Validator(artifact_schema).validate(
            load_json(PUBLICATION / scenario["data_file"])
        )


def test_index_rejects_duplicate_scenarios_and_consumer_urls() -> None:
    index = load_json(PUBLICATION / "dataset_validation_index_v1.json")
    duplicate = copy.deepcopy(index)
    duplicate["scenarios"].append(copy.deepcopy(duplicate["scenarios"][0]))
    with pytest.raises(DatasetValidationError, match="duplicate scenario"):
        validate_index(duplicate)

    website_path = copy.deepcopy(index)
    website_path["scenarios"][0]["data_file"] = "./data/result.json"
    with pytest.raises(DatasetValidationError, match="data_file"):
        validate_index(website_path)


def test_artifact_rejects_scenario_drift_duplicate_cells_and_missing_evidence() -> None:
    artifact = load_json(PUBLICATION / "dataset_validation_qwen35_bidkv.json")
    with pytest.raises(DatasetValidationError, match="does not match"):
        validate_artifact(artifact, expected_scenario_id="another-scenario")

    duplicate = copy.deepcopy(artifact)
    duplicate["results"].append(copy.deepcopy(duplicate["results"][0]))
    with pytest.raises(DatasetValidationError, match="duplicate result cell"):
        validate_artifact(duplicate, expected_scenario_id=artifact["scenario"]["id"])

    missing_evidence = copy.deepcopy(artifact)
    missing_evidence["results"][0]["provenance"].pop("artifact")
    missing_evidence["results"][0]["provenance"].pop("report_url")
    with pytest.raises(DatasetValidationError, match="evidence URL"):
        validate_artifact(
            missing_evidence, expected_scenario_id=artifact["scenario"]["id"]
        )

    nullable_current_value = copy.deepcopy(artifact)
    nullable_current_value["results"][0]["current_value"] = None
    nullable_current_value["results"][0].pop("provenance")
    with pytest.raises(DatasetValidationError, match="lacks provenance"):
        validate_artifact(
            nullable_current_value,
            expected_scenario_id=artifact["scenario"]["id"],
        )
