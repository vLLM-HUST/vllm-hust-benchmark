"""Inert relation fixtures, not real SQLite contents or native execution."""

import hashlib
import json

import pytest

from vllm_hust_workload_data import spider


@pytest.fixture
def original(tmp_path, monkeypatch):
    schema = [{"db_id": "same", "table_names_original": ["original"]}]
    for name in spider.SCHEMAS:
        path = tmp_path / name
        path.write_text(json.dumps(schema))
        monkeypatch.setitem(
            spider.SCHEMAS, name, hashlib.sha256(path.read_bytes()).hexdigest()
        )
    row = {
        "db_id": "same",
        "question": "original question",
        "question_toks": ["question"],
        "query": "reference SQL",
        "query_toks": ["reference"],
        "query_toks_no_value": ["reference"],
        "sql": {"reference": True},
    }
    for name, (schema_name, namespace, _) in list(spider.SPLITS.items()):
        monkeypatch.setitem(spider.SPLITS, name, (schema_name, namespace, 1))
        (tmp_path / name).write_text(json.dumps([row]))
        folder = tmp_path / namespace / "same"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "same.sqlite").write_bytes(
            b"SQLite format 3\0inert-not-a-real-database-" + namespace.encode()
        )
    return tmp_path, row


def test_same_db_id_preserves_distinct_original_namespaces(original):
    root, original_row = original
    dev = list(spider.original_rows(root / "dev.json"))[0]
    test = list(spider.original_rows(root / "test.json"))[0]
    assert {k: dev[k] for k in original_row} == original_row
    assert dev["database_asset"]["namespace"] == "database"
    assert test["database_asset"]["namespace"] == "test_database"
    assert dev["database_asset"]["sha256"] != test["database_asset"]["sha256"]
    assert dev["schema"] == test["schema"]


def test_missing_test_namespace_does_not_fall_back_to_train(original):
    root, _ = original
    (root / "test_database/same/same.sqlite").unlink()
    with pytest.raises(FileNotFoundError):
        list(spider.original_rows(root / "test.json"))


def test_changed_schema_is_rejected(original):
    root, _ = original
    (root / "test_tables.json").write_text("[]")
    with pytest.raises(ValueError, match="schema identity"):
        list(spider.original_rows(root / "test.json"))


def test_unified_reader_keeps_references_out_of_inputs(original):
    from vllm_hust_workload_data.adapters import iter_workload

    root, row = original
    path = root / "dev.json"
    manifest = {
        "source_id": "SPIDER",
        "revision": "fixture-revision",
        "unit": "source_rows",
        "provenance_kind": "text_to_sql_corpus",
        "files": [
            {
                "path": "dev.json",
                "role": "data",
                "format": "json",
                "rows": 1,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    records = list(iter_workload("SPIDER", data_root=root, manifest=manifest))
    assert len(records) == 1
    record = records[0]
    inputs = record.input_record()
    assert inputs["task"]["input"]["question"] == row["question"]
    assert "reference SQL" not in json.dumps(inputs)
    assert record.outcome_record()["query"] == row["query"]
    assert inputs["replay"]["annotations"]["database_asset"]["namespace"] == "database"
