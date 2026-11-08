"""Original fields and dialogue/reference relations, not synthetic chat cases."""

import hashlib
import json
import sqlite3
from dataclasses import asdict

import pytest

from vllm_hust_workload_data.adapters import iter_workload
from vllm_hust_workload_data.dailytalk import original_rows
from vllm_hust_workload_data.registry import source_spec


RAW = (
    b"0_1_d23|1|{HH AH}|Original first.|neutral\r\n"
    b"2_1_d23|1|{T UW}|Original third.|happiness\r\n"
    b"1_0_d23|0|{W AH N}|Original second.|neutral\r\n"
)


def test_original_file_order_and_numeric_dialogue_edges(tmp_path):
    path = tmp_path / "val_phone.txt"
    path.write_bytes(RAW)
    rows = list(original_rows(path))
    assert [row["turn_index"] for row in rows] == [0, 2, 1]
    assert [row["previous_turn_id"] for row in rows] == [None, "1_0_d23", "0_1_d23"]
    assert b"".join(row["original_row"].encode() for row in rows) == RAW
    assert (
        rows[0]["reference_audio"]["relative_path"] == "dailytalk/data/23/0_1_d23.wav"
    )
    assert not rows[0]["reference_audio"]["local_media_verified"]


def test_provider_preserves_all_columns_reference_side_and_native_contract(tmp_path):
    path = tmp_path / "val_phone.txt"
    path.write_bytes(RAW)
    manifest = {
        "source_id": "DAILYTALK",
        "revision": "fixture-original-v1",
        "unit": "source_rows",
        "provenance_kind": "recorded_dialogue_corpus",
        "files": [
            {
                "path": path.name,
                "role": "data",
                "bytes": len(RAW),
                "sha256": hashlib.sha256(RAW).hexdigest(),
            }
        ],
    }
    records = list(iter_workload("DAILYTALK", data_root=tmp_path, manifest=manifest))
    assert len(records) == 3
    first, third, second = records
    assert first.task["input"] == {
        "phonemes": "{HH AH}",
        "text": "Original first.",
        "speaker": "1",
        "emotion": "neutral",
    }
    assert first.task["session_id"] == "d23"
    assert first.task["parent_task_id"] is None
    assert third.task["parent_task_id"] == second.task["task_id"]
    assert "reference_audio" not in first.task["input"]
    assert "messages" not in first.task["input"]
    assert first.outcome_record()["reference_audio"]["format"] == "wav"
    assert first.replay["unmapped_field_names"] == []
    assert (
        b"".join(
            record.replay["annotations"]["original_row"].encode() for record in records
        )
        == RAW
    )
    with pytest.raises(ValueError, match="not a ready model request"):
        first.request_body()
    assert not first.replay["synthetic_fallback"]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"bad|row\n",
        b"0_0_d1|1|{P}|text|neutral\n",
        b"0_0_d1|0|{P}|text|neutral\n" * 2,
    ],
)
def test_malformed_duplicate_or_speaker_mismatch_rejected(tmp_path, raw):
    path = tmp_path / "train_phone.txt"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        list(original_rows(path))


def test_missing_turn_is_not_invented(tmp_path):
    path = tmp_path / "train_phone.txt"
    path.write_bytes(b"2_0_d3|0|{P}|Only original turn.|neutral\n")
    assert list(original_rows(path))[0]["previous_turn_id"] is None


def test_registered_original_splits_match_preparation_pins():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    pins = json.loads(
        (
            root
            / "workload-preparation/data/catalogs/dailytalk_original_metadata_files_v1.json"
        ).read_text("utf-8")
    )
    source = source_spec("DAILYTALK")
    assert source["revision"] == pins["revision"]
    original = {item["path"]: item for item in pins["files"]}
    assert len(source["files"]) == 2
    for item in source["files"]:
        assert item["sha256"] == original[item["path"]]["sha256"]
        assert item["bytes"] == original[item["path"]]["bytes"]


def test_sqlite_reference_and_public_consumer_roundtrip(tmp_path):
    from vllm_hust_workload_data.pipeline import run
    from vllm_hust_workload_data.consumer import resolve_input

    data = tmp_path / "data"
    data.mkdir()
    (data / "val_phone.txt").write_bytes(RAW)
    manifest = {
        "source_id": "DAILYTALK",
        "revision": "fixture-original-v1",
        "unit": "source_rows",
        "provenance_kind": "recorded_dialogue_corpus",
        "files": [
            {
                "path": "val_phone.txt",
                "role": "data",
                "bytes": len(RAW),
                "sha256": hashlib.sha256(RAW).hexdigest(),
            }
        ],
    }
    path = tmp_path / "source-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    report = run(
        {
            "sources": [
                {
                    "source_id": "DAILYTALK",
                    "data_root": str(data),
                    "manifest": str(path),
                }
            ]
        },
        output=tmp_path / "prepared",
    )
    assert report["sources"][0]["records"] == 3
    database = tmp_path / "prepared/workload.sqlite"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    with sqlite3.connect(database) as connection:
        ids = [
            row[0]
            for row in connection.execute(
                "select record_id from records order by rowid"
            )
        ]
    originals = list(iter_workload("DAILYTALK", data_root=data, manifest=manifest))
    for record_id, original in zip(ids, originals):
        reread = resolve_input(database, record_id, data_root=data)
        assert asdict(reread) == asdict(original)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def test_declared_original_protocol_is_batch_inference_not_quality_score():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    profile = json.loads(
        (root / "configs/workload-native/dailytalk-guo-batch-protocol.json").read_text(
            "utf-8"
        )
    )
    pins = json.loads(
        (root / "configs/workload-native/dailytalk-author-source.json").read_text(
            "utf-8"
        )
    )
    original = {item["path"]: item for item in pins["files"]}
    assert profile["publisher_revision"] == pins["revision"]
    assert (
        profile["publisher_protocol"]["readme_sha256"]
        == original["README.md"]["sha256"]
    )
    assert (
        profile["publisher_protocol"]["evaluation_source_sha256"]
        == original["evaluate.py"]["sha256"]
    )
    assert (
        profile["seed"],
        profile["batch_size"],
        profile["history_window_turns"],
    ) == (1234, 8, 10)
    assert profile["acoustic_checkpoint_step"] == 900000
    assert profile["pitch_energy_duration_controls"] == [1.0, 1.0, 1.0]
    assert profile["fixed_native_validation_selection"]["utterances"] == 29
    assert not profile["evaluation_boundary"]["quality_benchmark_score_claimed"]
    assert not profile["evaluation_boundary"]["full_split_synthesis_claimed"]
