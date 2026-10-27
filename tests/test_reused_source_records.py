"""Tiny software fixtures only: no publisher data or native benchmark claims."""

import hashlib
import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reuse_records", ROOT / "scripts/prepare_reused_source_records.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_docvqa_public_prepare_preserves_original_fields_and_media(tmp_path):
    from vllm_hust_workload_data.consumer import resolve_input

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    sid = "DOCVQA"
    filename = "DocVQA/validation-00000-of-00006.parquet"
    original = {
        "questionId": "fixture-question",
        "question": "fixture input",
        "question_types": ["fixture type"],
        "docId": 7,
        "image": {"bytes": b"\x89PNG\r\n\x1a\nfixture", "path": "fixture.png"},
        "ucsf_document_id": "fixture-document",
        "ucsf_document_page_no": "001",
        "answers": ["fixture reference"],
        "data_split": "val",
    }
    data = tmp_path / "original"
    path = data / filename
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([original]), path)
    raw = path.read_bytes()
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    for name in (
        "source_manifests_v1.json",
        "adapter_registry_v1.json",
        "adapter_extensions_v1.json",
    ):
        payload = json.loads(
            (ROOT / "workload-preparation/data/catalogs" / name).read_text("utf-8")
        )
        if name == "source_manifests_v1.json":
            payload["sources"][sid]["files"] = [
                {
                    "path": filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "rows": 1,
                    "role": "raw_source",
                }
            ]
        (catalog / "data/catalogs" / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    output = tmp_path / "prepared"
    result = runner.prepare(sid, data_root=data, output=output, catalog_root=catalog)
    db = output / "prepared/workload.sqlite"
    with sqlite3.connect(db) as connection:
        ids = connection.execute("select record_id from records").fetchall()
    assert len(ids) == 1
    record = resolve_input(db, ids[0][0], data_root=data)
    assert {
        **record.task["input"],
        **record.replay["annotations"],
        **record.outcome_record(),
    } == original
    assert "answers" not in record.task["input"]
    assert path.read_bytes() == raw
    assert result["source_rows_fully_bound"] == 1
    assert not result["native_execution_complete"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_copilot_public_prepare_index_and_resolve_preserve_original_nested_metadata(
    tmp_path,
):
    from vllm_hust_workload_data.consumer import resolve_input

    sid = "COPILOT_AGENT_2026"
    rows = [
        {
            "session_id": "same-label",
            "date": "2026-06-01",
            "turns": [
                {
                    "turn_id": "turn",
                    "llm_calls": [{"message_id": "call", "extension": [index, None]}],
                    "tool_batches": [
                        {"request_id": "call", "function_calls": [{"status": 2}]}
                    ],
                }
            ],
        }
        for index in range(2)
    ]
    filename = "date=2026-06-01/shard-0000.jsonl.gz"
    data = tmp_path / "original"
    (data / filename).parent.mkdir(parents=True)
    raw = gzip.compress(("\n".join(json.dumps(row) for row in rows) + "\n").encode())
    (data / filename).write_bytes(raw)
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    for name in (
        "source_manifests_v1.json",
        "adapter_registry_v1.json",
        "adapter_extensions_v1.json",
    ):
        payload = json.loads(
            (ROOT / "workload-preparation/data/catalogs" / name).read_text("utf-8")
        )
        if name == "source_manifests_v1.json":
            payload["sources"][sid]["files"] = [
                {
                    "path": filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "rows": 2,
                    "role": "raw_source",
                }
            ]
        (catalog / "data/catalogs" / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    output = tmp_path / "prepared"
    result = runner.prepare(sid, data_root=data, output=output, catalog_root=catalog)
    db = output / "prepared/workload.sqlite"
    with sqlite3.connect(db) as connection:
        ids = [
            item[0]
            for item in connection.execute(
                "select record_id from records order by rowid"
            )
        ]
    assert len(set(ids)) == 2  # Same session label must not collapse original rows.
    for record_id, original in zip(ids, rows, strict=True):
        record = resolve_input(db, record_id, data_root=data)
        assert record.replay["annotations"] == original
        assert record.task["input"] == {} and record.outcome_record() is None
        assert record.replay["unmapped_field_names"] == []
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()
    assert result["source_rows_fully_bound"] == 2
    assert not result["native_execution_complete"]
    assert not result["research_observation_or_feature_views_migrated"]


@pytest.fixture(params=["LONGALPACA", "NATURALQUESTIONS", "DUREADER"])
def original_fixture(request, tmp_path):
    sid = request.param
    row = (
        {
            "instruction": "fixture instruction",
            "input": "fixture context",
            "output": "fixture answer",
            "file": "original-document-identity-fixture.pdf",
        }
        if sid == "LONGALPACA"
        else {"question": "fixture question", "answer": ["fixture answer", "alternate"]}
    )
    if sid == "DUREADER":
        row = {
            "question_id": 7,
            "question": "fixture question",
            "question_type": "DESCRIPTION",
            "fact_or_opinion": "FACT",
            "segmented_question": ["fixture", "question"],
            "segmented_answers": [["fixture", "answer"]],
            "documents": [
                {
                    "title": "fixture title",
                    "paragraphs": ["fixture context\u2028same physical JSONL record"],
                    "is_selected": True,
                    "most_related_para": 0,
                    "bs_rank_pos": 2,
                    "segmented_title": ["fixture", "title"],
                    "segmented_paragraphs": [["fixture", "context"]],
                }
            ],
            "answers": ["fixture answer"],
            "answer_docs": [0],
            "answer_spans": [[0, 1]],
        }
    name = "fixture.json" if sid == "LONGALPACA" else "fixture.jsonl"
    raw = (
        json.dumps([row] if sid == "LONGALPACA" else row, ensure_ascii=False).encode()
        + b"\n"
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / name).write_bytes(raw)
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    source["sources"][sid]["files"] = [
        {
            "path": name,
            "role": "raw_source",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    ]
    (catalog / "data/catalogs/source_manifests_v1.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    return sid, row, name, data, catalog


def test_prepare_and_resolve_preserve_fields_without_copying_data(
    original_fixture, tmp_path
):
    from vllm_hust_workload_data.consumer import resolve_input

    sid, row, name, data, catalog = original_fixture
    output = tmp_path / "prepared"
    receipt = runner.prepare(
        sid, data_root=data, output=output, catalog_root=catalog, limit=1
    )
    assert receipt["source_rows_fully_bound"] == 1
    assert receipt["preparation"]["status"] == "PREPARED_REFERENCES_ONLY"
    assert not receipt["native_execution_complete"]
    assert not receipt["source_checkbox_eligible"]
    database = output / "prepared/workload.sqlite"
    with sqlite3.connect(database) as connection:
        record_id = connection.execute("SELECT record_id FROM records").fetchone()[0]
    record = resolve_input(database, record_id, data_root=data)
    assert b"fixture answer" not in database.read_bytes()
    assert not (output / name).exists()
    if sid == "LONGALPACA":
        assert record.task["input"]["instruction"] == row["instruction"]
        assert record.outcome["reference_output"] == row["output"]
        assert record.outcome["original_source_file"] == row["file"]
        assert "original-document-identity-fixture.pdf" not in json.dumps(
            record.task["input"]
        )
    elif sid == "NATURALQUESTIONS":
        assert record.task["input"]["question"] == row["question"]
        assert record.outcome["answers"] == row["answer"]
    else:
        assert record.task["input"]["documents"] == [
            {
                "title": "fixture title",
                "paragraphs": ["fixture context\u2028same physical JSONL record"],
            }
        ]
        assert record.outcome["document_annotations"] == [
            {
                "is_selected": True,
                "most_related_para": 0,
                "bs_rank_pos": 2,
                "segmented_title": ["fixture", "title"],
                "segmented_paragraphs": [["fixture", "context"]],
            }
        ]
        assert record.outcome["answer_docs"] == [0]
        assert record.outcome["answer_spans"] == [[0, 1]]
        assert record.outcome["answers"] == row["answers"]
        for name in (
            "question_type",
            "fact_or_opinion",
            "segmented_question",
            "segmented_answers",
        ):
            assert record.outcome[name] == row[name]
            assert name not in record.task["input"]
    assert (
        "answers" not in record.task["input"]
        and "reference_output" not in record.task["input"]
    )
    with pytest.raises(ValueError, match="fresh output"):
        runner.prepare(sid, data_root=data, output=output, catalog_root=catalog)


def test_unknown_files_and_changed_identity_fail_closed(original_fixture, tmp_path):
    sid, _, name, data, catalog = original_fixture
    with pytest.raises(ValueError, match="registered"):
        runner.prepare(
            sid,
            data_root=data,
            output=tmp_path / "out",
            files=["unknown"],
            catalog_root=catalog,
        )
    (data / name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from pinned"):
        runner.prepare(
            sid, data_root=data, output=tmp_path / "out", catalog_root=catalog
        )
    assert not (tmp_path / "out").exists()


def test_data_only_profiles_do_not_imply_nq_retrieval():
    profiles = {
        sid: json.loads(path.read_text("utf-8"))
        for sid, path in runner.MAPPINGS.items()
    }
    assert all(profile["features"] == {} for profile in profiles.values())
    assert profiles["NATURALQUESTIONS"]["families"] == ["general-single-turn-assistant"]


@pytest.mark.parametrize(
    "sid,field,id_field",
    [
        ("WILDCHAT", "conversation", "conversation_hash"),
        ("ULTRACHAT", "messages", "prompt_id"),
        ("LMSYS_CHAT", "conversation", "conversation_id"),
    ],
)
@pytest.mark.parametrize("file_format", ["jsonl", "parquet"])
def test_existing_dialogue_adapters_preserve_original_history(
    tmp_path, sid, field, id_field, file_format
):
    from vllm_hust_workload_data.consumer import resolve_input

    messages = [
        {"role": "user", "content": "first fixture"},
        {"role": "assistant", "content": "historical fixture"},
        {"role": "user", "content": "second fixture"},
        {"role": "assistant", "content": "final historical fixture"},
    ]
    row = {field: messages, id_field: "fixture-id"}
    data = tmp_path / "data"
    data.mkdir()
    filename = "fixture." + file_format
    if file_format == "parquet":
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        pq.write_table(pa.Table.from_pylist([row]), data / filename)
    else:
        (data / filename).write_text(json.dumps(row) + "\n", encoding="utf-8")
    raw = (data / filename).read_bytes()
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    for name in (
        "source_manifests_v1.json",
        "adapter_registry_v1.json",
        "adapter_extensions_v1.json",
    ):
        payload = json.loads(
            (ROOT / "workload-preparation/data/catalogs" / name).read_text("utf-8")
        )
        if name == "source_manifests_v1.json":
            payload["sources"][sid]["files"] = [
                {
                    "path": filename,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "role": "raw_source",
                }
            ]
        (catalog / "data/catalogs" / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    output = tmp_path / "output"
    receipt = runner.prepare(sid, data_root=data, output=output, catalog_root=catalog)
    with sqlite3.connect(output / "prepared/workload.sqlite") as connection:
        rid = connection.execute("SELECT record_id FROM records").fetchone()[0]
    record = resolve_input(output / "prepared/workload.sqlite", rid, data_root=data)
    assert record.task["input"][field] == messages
    assert record.runtime["kind"] == "historical_dialogue"
    assert not receipt["native_execution_complete"]
    assert not (output / "source-mapping.private.json").exists()
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_revision_drift_never_downgrades_to_unpinned_local_snapshot(
    original_fixture, tmp_path
):
    sid, _, _, data, catalog = original_fixture
    path = catalog / "data/catalogs/source_manifests_v1.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["sources"][sid]["revision"] = "different-revision"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="re-review required"):
        runner.prepare(
            sid, data_root=data, output=tmp_path / "out", catalog_root=catalog
        )
    assert not (tmp_path / "out").exists()


def test_infinitebench_file_groups_are_explicit_complete_and_disjoint():
    assert len(runner.INFINITE_NUMERIC_FILES) == 3
    assert len(runner.INFINITE_STRING_FILES) == 9
    assert not runner.INFINITE_NUMERIC_FILES & runner.INFINITE_STRING_FILES
    catalog = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    assert {
        item["path"] for item in catalog["sources"]["INFINITEBENCH"]["files"]
    } == runner.INFINITE_NUMERIC_FILES | runner.INFINITE_STRING_FILES
    for files in (None, [], ["unknown.jsonl"], ["math_calc.jsonl", "passkey.jsonl"]):
        with pytest.raises(ValueError):
            runner.mapping_path("INFINITEBENCH", files)


@pytest.mark.parametrize(
    "filename,references",
    [
        ("math_calc.jsonl", [1, 2.5]),
        ("code_debug.jsonl", ["fixture answer", "A"]),
        ("number_string.jsonl", ["0007"]),
    ],
)
def test_infinitebench_original_reference_types_survive_prepare_resolve(
    tmp_path, filename, references
):
    from vllm_hust_workload_data.consumer import resolve_input

    row = {
        "id": 0,
        "context": "fixture complete context",
        "input": "fixture question",
        "options": ["fixture answer", "other"],
        "answer": references,
    }
    raw = (json.dumps(row) + "\n").encode()
    data = tmp_path / "data"
    data.mkdir()
    (data / filename).write_bytes(raw)
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    source = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text("utf-8")
    )
    source["sources"]["INFINITEBENCH"]["files"] = [
        {
            "path": filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "role": "raw_source",
        }
    ]
    (catalog / "data/catalogs/source_manifests_v1.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    out = tmp_path / "out"
    result = runner.prepare(
        "INFINITEBENCH",
        data_root=data,
        output=out,
        files=[filename],
        catalog_root=catalog,
    )
    with sqlite3.connect(out / "prepared/workload.sqlite") as connection:
        rid = connection.execute("SELECT record_id FROM records").fetchone()[0]
    record = resolve_input(out / "prepared/workload.sqlite", rid, data_root=data)
    assert record.outcome["references"] == references
    assert [type(value) for value in record.outcome["references"]] == [
        type(value) for value in references
    ]
    assert record.task["input"]["context"] == row["context"]
    assert record.task["input"]["feature_inputs"] == {}
    assert "answer" not in record.task["input"]
    assert not result["native_execution_complete"]
