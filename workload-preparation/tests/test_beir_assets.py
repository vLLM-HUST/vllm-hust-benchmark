import json
import hashlib
from zipfile import ZipFile

import pytest

from vllm_hust_workload_data.beir import adapt_asset, original_asset_rows
from vllm_hust_workload_data import beir


@pytest.mark.parametrize(
    "filename,raw,unit,fields",
    [
        (
            "corpus.jsonl",
            {
                "_id": "doc-7",
                "title": "Title",
                "text": "Original",
                "metadata": {"x": 1},
            },
            "corpus_document",
            {"title", "text"},
        ),
        (
            "queries.jsonl",
            {"_id": "query-3", "text": "Original question", "metadata": {}},
            "retrieval_query",
            {"text"},
        ),
    ],
)
def test_original_text_and_metadata_are_lossless(tmp_path, filename, raw, unit, fields):
    path = tmp_path / filename
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    (row,) = original_asset_rows(path)
    record = adapt_asset(row, source={"file_sha256": "fixture"}, index=0)
    assert record.task["task_id"] == raw["_id"]
    assert record.task["native_unit"] == unit
    assert set(record.task["input"]) == fields
    assert {**record.task["input"], **record.replay["annotations"]} == raw


def test_qrels_never_become_model_input_or_generated_pairs(tmp_path):
    directory = tmp_path / "qrels"
    directory.mkdir()
    path = directory / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\td2\t-1\n", encoding="utf-8")
    (row,) = original_asset_rows(path)
    record = adapt_asset(row, source={"file_sha256": "fixture"}, index=0)
    assert record.task["input"] == {}
    assert record.runtime["asset_role"] == "reference_only"
    assert record.outcome == {
        "original_judgment": {"query-id": "q1", "corpus-id": "d2", "score": "-1"}
    }


def test_duplicate_native_ids_are_rejected(tmp_path):
    path = tmp_path / "queries.jsonl"
    raw = json.dumps({"_id": "same", "text": "Original"}) + "\n"
    path.write_text(raw * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate original BEIR ID"):
        list(original_asset_rows(path))


@pytest.mark.parametrize(
    "content", ["wrong\theader\n", "query-id\tcorpus-id\tscore\nq\td\tx\n"]
)
def test_malformed_judgments_fail_without_rewriting(tmp_path, content):
    directory = tmp_path / "qrels"
    directory.mkdir()
    path = directory / "test.tsv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        list(original_asset_rows(path))


def test_unknown_asset_is_not_fabricated_as_chat(tmp_path):
    with pytest.raises(ValueError, match="no prompt fallback"):
        list(original_asset_rows(tmp_path / "unknown.jsonl"))


def fixture_archive(tmp_path, monkeypatch, *, missing_document=False):
    values = {
        "corpus.jsonl": b'{"_id":"d","title":"Title","text":"Text","metadata":{}}\n',
        "queries.jsonl": b'{"_id":"q","text":"Question","metadata":{}}\n',
        "qrels/test.tsv": b"query-id\tcorpus-id\tscore\nq\td\t1\n"
        if not missing_document
        else b"query-id\tcorpus-id\tscore\nq\tmissing\t1\n",
    }
    archive = tmp_path / "original.zip"
    files = []
    with ZipFile(archive, "w") as zipped:
        for name, raw in values.items():
            member = "original/" + name
            zipped.writestr(member, raw)
            files.append(
                {
                    "role": "data",
                    "path": name,
                    "archive_member": member,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "rows": 1,
                }
            )
    source = {
        "revision": "fixture",
        "files": [
            {
                "role": "source_archive",
                "bytes": archive.stat().st_size,
                "sha256": beir.sha256(archive),
            },
            *files,
        ],
    }
    monkeypatch.setattr(beir, "source_spec", lambda source_id: source)
    return archive, values, source


def test_archive_preparation_preserves_original_bytes_and_every_edge(
    tmp_path, monkeypatch
):
    archive, values, _ = fixture_archive(tmp_path, monkeypatch)
    output = tmp_path / "fresh"
    report = beir.prepare_original_archive(
        source_id="BEIR_SCIFACT", data_archive=archive, output=output
    )
    assert (
        report["original_corpus_documents"]
        == report["original_queries_all_splits"]
        == 1
    )
    assert report["original_qrels_splits"][0]["original_judgment_rows"] == 1
    assert not report["native_execution_or_scoring_verified"]
    for name, raw in values.items():
        assert (output / name).read_bytes() == raw
    with pytest.raises(ValueError, match="fresh BEIR output"):
        beir.prepare_original_archive(
            source_id="BEIR_SCIFACT", data_archive=archive, output=output
        )


def test_missing_original_relation_is_not_filtered_away(tmp_path, monkeypatch):
    archive, _, _ = fixture_archive(tmp_path, monkeypatch, missing_document=True)
    with pytest.raises(ValueError, match="missing query or document"):
        beir.prepare_original_archive(
            source_id="BEIR_SCIFACT", data_archive=archive, output=tmp_path / "fresh"
        )


def test_wrong_member_digest_never_creates_output(tmp_path, monkeypatch):
    archive, _, source = fixture_archive(tmp_path, monkeypatch)
    source["files"][1]["sha256"] = "0" * 64
    output = tmp_path / "fresh"
    with pytest.raises(ValueError, match="member identity mismatch"):
        beir.prepare_original_archive(
            source_id="BEIR_SCIFACT", data_archive=archive, output=output
        )
    assert not output.exists()


def test_registered_path_cannot_escape_output(tmp_path, monkeypatch):
    archive, _, source = fixture_archive(tmp_path, monkeypatch)
    source["files"][1]["path"] = "../escape.jsonl"
    with pytest.raises(ValueError, match="escapes fresh output"):
        beir.prepare_original_archive(
            source_id="BEIR_SCIFACT", data_archive=archive, output=tmp_path / "fresh"
        )
    assert not (tmp_path / "escape.jsonl").exists()
