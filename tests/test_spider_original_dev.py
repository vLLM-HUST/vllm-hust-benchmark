"""Inert file-link fixtures only; no real SQLite database, SQL or model run."""

import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "spider_prepare", ROOT / "scripts/prepare_spider_original_dev.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data, suites = tmp_path / "data", tmp_path / "suites"
    data.mkdir()
    (suites / "fixture_db").mkdir(parents=True)
    row = {
        "db_id": "fixture_db",
        "question": "inert original question",
        "query": "SELECT secret_reference",
        "query_toks": ["reference"],
        "query_toks_no_value": ["reference"],
        "question_toks": ["question"],
        "sql": {"reference_ast": True},
    }
    (data / "dev.json").write_text(json.dumps([row] * 3))
    (data / "tables.json").write_text(
        json.dumps([{"db_id": "fixture_db", "table_names_original": ["table"]}])
    )
    archive = tmp_path / "suite.zip"
    with zipfile.ZipFile(archive, "w") as zip_out:
        for name in ("fixture_db.sqlite", "fixture_db_variant.sqlite"):
            raw = b"SQLite format 3\0inert fixture, not an executable database"
            (suites / "fixture_db" / name).write_bytes(raw)
            zip_out.writestr("database/fixture_db/" + name, raw)
    monkeypatch.setattr(
        module,
        "EXPECTED_INPUTS",
        {n: module.digest(data / n) for n in ("dev.json", "tables.json")},
    )
    monkeypatch.setattr(module, "SUITE_ARCHIVE_SHA", module.digest(archive))
    monkeypatch.setattr(module, "DEV_ROWS", 3)
    kwargs = dict(
        data_root=data,
        suite_root=suites,
        suite_archive=archive,
        output=tmp_path / "out",
        count=3,
    )
    return module, kwargs, row


def test_preserves_original_fields_suites_and_reference_isolation(prepared):
    module, kwargs, row = prepared
    result = module.prepare(**kwargs)
    assert result["selected_original_dev_indices"] == [0, 1, 2]
    assert len(result["suite_files"]) == 2
    assert result["model_calls"] == result["sql_statements_executed"] == 0
    assert not result["source_checkbox_eligible"]
    output = kwargs["output"]
    inputs = [
        json.loads(line)
        for line in (output / "inputs.private.jsonl").read_text().splitlines()
    ]
    refs = [
        json.loads(line)
        for line in (output / "references.private.jsonl").read_text().splitlines()
    ]
    assert all(ref["original_record"] == row for ref in refs)
    assert all(item["question"] == row["question"] for item in inputs)
    assert "secret_reference" not in json.dumps(inputs)
    assert "reference_ast" not in json.dumps(inputs)
    assert (output / "gold.private.txt").read_text() == (
        row["query"] + "\tfixture_db\n"
    ) * 3
    with pytest.raises(ValueError, match="fresh"):
        module.prepare(**kwargs)


@pytest.mark.parametrize(
    "kind", ["missing", "changed", "extra", "changed_dev", "invalid_count"]
)
def test_rejects_missing_or_rewritten_assets_without_output(prepared, kind):
    module, kwargs, _ = prepared
    directory = kwargs["suite_root"] / "fixture_db"
    if kind == "missing":
        (directory / "fixture_db_variant.sqlite").unlink()
    elif kind == "changed":
        (directory / "fixture_db_variant.sqlite").write_bytes(b"changed")
    elif kind == "extra":
        (directory / "invented.sqlite").write_bytes(b"invented")
    elif kind == "changed_dev":
        (kwargs["data_root"] / "dev.json").write_text("[]")
    else:
        kwargs["count"] = True
    with pytest.raises(ValueError):
        module.prepare(**kwargs)
    assert not kwargs["output"].exists()
