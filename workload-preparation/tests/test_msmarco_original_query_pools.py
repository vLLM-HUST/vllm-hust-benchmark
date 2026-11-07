"""Fixtures test complete native query pools, never actual benchmark execution."""

from copy import deepcopy
import hashlib
import io
import json
import tarfile

import pytest

from vllm_hust_workload_data.adapters import iter_workload
from vllm_hust_workload_data.registry import source_spec


def assets(tmp_path, rows=None):
    rows = rows or [
        '10\t1\t original query \tpassage with "quotes"\n',
        "10\t2\t original query \tpassage\twith literal tab\n",
        "20\t4\tsecond query\tsecond passage\n",
    ]
    raw = "".join(rows).encode()
    path = tmp_path / "top1000.dev.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo("top1000.dev")
        item.size = len(raw)
        archive.addfile(item, io.BytesIO(raw))
    (tmp_path / "qrels.dev.tsv").write_bytes(b"10\t0\t99\t1\n")
    manifest = deepcopy(source_spec("MSMARCO_PASSAGE"))
    manifest["revision"] = "fixture-only-not-runtime-evidence"
    for item in manifest["files"]:
        data = (tmp_path / item["path"]).read_bytes()
        item.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        if item["role"] == "data":
            item["archive_member_bytes"] = len(raw)
            item["rows"] = len({row.split("\t", 1)[0] for row in rows})
    return manifest


def test_original_query_text_complete_pool_and_hidden_qrels(tmp_path):
    manifest = assets(tmp_path)
    records = list(
        iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest)
    )
    assert [x.task["task_id"] for x in records] == ["10", "20"]
    inputs = records[0].task["input"]
    assert set(inputs) == {"query", "candidates"}
    assert inputs["query"] == " original query "
    assert [x["passage_id"] for x in inputs["candidates"]] == ["1", "2"]
    assert inputs["candidates"][1]["passage"] == "passage\twith literal tab"
    assert [x["original_position_within_query"] for x in inputs["candidates"]] == [0, 1]
    assert records[0].outcome["references"]["original_qrel_rows"] == [
        ["10", "0", "99", "1"]
    ]
    assert records[1].outcome["references"]["original_qrel_rows"] == []
    assert records[0].runtime["execution"] == "not_executed"
    with pytest.raises(ValueError, match="not a ready model request"):
        records[0].request_body()


def test_diagnostic_limit_is_query_count_not_candidate_truncation(tmp_path):
    manifest = assets(tmp_path)
    (record,) = list(
        iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest, limit=1)
    )
    assert len(record.task["input"]["candidates"]) == 2


def test_qrels_identity_is_checked_even_when_only_archive_selected(tmp_path):
    manifest = assets(tmp_path)
    (tmp_path / "qrels.dev.tsv").write_bytes(b"10\t0\t1\t1\n")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        list(
            iter_workload(
                "MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest, limit=1
            )
        )


def test_interleaved_pool_is_complete_and_keeps_original_order(tmp_path):
    manifest = assets(tmp_path, ["10\t1\tq\tp\n", "20\t2\tu\tv\n", "10\t3\tq\tw\n"])
    records = list(
        iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest)
    )
    assert [record.task["task_id"] for record in records] == ["10", "20"]
    assert [
        candidate["passage_id"] for candidate in records[0].task["input"]["candidates"]
    ] == ["1", "3"]
    (first,) = list(
        iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest, limit=1)
    )
    assert first.task == records[0].task


def test_inconsistent_query_is_not_rewritten(tmp_path):
    manifest = assets(tmp_path, ["10\t1\tq\tp\n", "10\t2\tchanged\tw\n"])
    with pytest.raises(ValueError, match="query text differs"):
        list(iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest))


def test_crlf_framing_is_not_field_content_and_spaces_are_preserved(tmp_path):
    manifest = assets(tmp_path, ["10\t1\t query \t passage \r\n"])
    reference = tmp_path / "qrels.dev.tsv"
    reference.write_bytes(b"10\t0\t1\t1\r\n")
    item = next(x for x in manifest["files"] if x["role"] == "reference")
    item.update(
        bytes=reference.stat().st_size,
        sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
    )
    (record,) = list(
        iter_workload("MSMARCO_PASSAGE", data_root=tmp_path, manifest=manifest)
    )
    assert record.task["input"]["query"] == " query "
    assert record.task["input"]["candidates"][0]["passage"] == " passage "
    assert record.outcome["references"]["original_qrel_rows"] == [["10", "0", "1", "1"]]


def test_prepared_index_preserves_qrels_binding_and_complete_pool(
    tmp_path, monkeypatch
):
    from dataclasses import asdict
    from vllm_hust_workload_data import consumer
    from vllm_hust_workload_data.pipeline import run, prepared_rows

    data = tmp_path / "data"
    data.mkdir()
    manifest = assets(data, ["10\t1\tq\tp\n", "20\t2\tu\tv\n", "10\t3\tq\tw\n"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    run(
        {
            "sources": [
                {
                    "source_id": "MSMARCO_PASSAGE",
                    "data_root": str(data),
                    "manifest": str(manifest_path),
                    "limit": 1,
                }
            ]
        },
        output=output,
    )
    (entry,) = list(prepared_rows(output / "workload.sqlite"))
    observed = []
    original = consumer.iter_source_rows

    def spy(path, spec):
        observed.append(spec)
        yield from original(path, spec)

    monkeypatch.setattr(consumer, "iter_source_rows", spy)
    restored = consumer.resolve_input(
        output / "workload.sqlite", entry["record_id"], data_root=data
    )
    (direct,) = list(
        iter_workload("MSMARCO_PASSAGE", data_root=data, manifest=manifest, limit=1)
    )
    assert asdict(restored) == asdict(direct)
    assert observed[0]["original_qrels"]["sha256"] == next(
        item["sha256"] for item in manifest["files"] if item["role"] == "reference"
    )
    assert observed[0]["diagnostic_query_limit"] == 1
    import sqlite3

    with sqlite3.connect(output / "workload.sqlite") as connection:
        stored = json.loads(
            connection.execute("SELECT adapter_json FROM sources").fetchone()[0]
        )
    assert stored["diagnostic_query_limit"] == 1
    assert stored["original_qrels"] == observed[0]["original_qrels"]
    (data / "qrels.dev.tsv").write_bytes(b"10\t0\t98\t1\n")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        consumer.resolve_input(
            output / "workload.sqlite", entry["record_id"], data_root=data
        )
