"""Synthetic schema fixtures; no downloaded schemas or model outputs."""

from copy import deepcopy
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.registry import source_spec


def test_fixed_acquisition_urls_match_original_version_and_paths():
    manifest = source_spec("JSONSCHEMABENCH")
    assert len(manifest["files"]) == 3
    for file in manifest["files"]:
        assert file["url"] == (
            "https://raw.githubusercontent.com/guidance-ai/jsonschemabench/"
            + manifest["revision"]
            + "/data/"
            + file["path"]
        )
        assert file["rows"] == 1


@pytest.mark.parametrize(
    "schema",
    [
        True,
        False,
        {},
        {
            "type": "object",
            "properties": {"fixture": {"maxLength": 3}},
            "$ref": "https://invalid.example/do-not-fetch",
            "x-fixture-extension": {"unchanged": [1, False]},
        },
    ],
)
def test_schema_asset_is_one_unchanged_input_without_network(tmp_path, schema):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    manifest = deepcopy(source_spec("JSONSCHEMABENCH"))
    manifest["revision"] = "synthetic-unit-fixture"
    manifest["files"] = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 1,
            "role": "data",
        }
    ]
    rows = list(iter_workload("JSONSCHEMABENCH", data_root=tmp_path, manifest=manifest))
    assert len(rows) == 1
    assert rows[0].task["input"] == {"schema": schema}
    assert not rows[0].outcome_record()
    assert rows[0].replay["synthetic_fallback"] is False
    binding = bind_source(
        "JSONSCHEMABENCH",
        data_root=tmp_path,
        files=[path.name],
        revision=manifest["revision"],
    )
    assert binding["files"][0]["rows"] == 1
    assert json.loads(path.read_text(encoding="utf-8")) == schema


@pytest.mark.parametrize(
    "text", ["[]", "null", "12", '"PRIVATE_SENTINEL"', '{"a": NaN}']
)
def test_non_schema_and_invalid_json_rejected_without_payload(tmp_path, text):
    from vllm_hust_workload_data.adapters import iter_source_rows

    path = tmp_path / "fixture.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(iter_source_rows(path, {"mapping": "whole_json_schema"}))
    assert "PRIVATE_SENTINEL" not in str(error.value)
