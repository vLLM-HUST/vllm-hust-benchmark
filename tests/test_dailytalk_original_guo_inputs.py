"""Preparation fixtures are not native TTS or original-data execution evidence."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "dailytalk_guo", ROOT / "scripts/prepare_dailytalk_original_guo_inputs.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_plan_never_reads_data_or_creates_output(tmp_path):
    report = runner.prepare(
        SimpleNamespace(
            metadata_root=tmp_path / "missing",
            audio_root=tmp_path / "missing-audio",
            output=tmp_path / "absent",
            action="plan",
        )
    )
    assert report["history_window_turns"] == 10
    assert report["history_encoder_embedding_dimension"] == 512
    assert report["model_calls"] == report["embedding_arrays_created"] == 0
    assert not report["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())


def test_rows_preserve_original_bytes_and_order():
    raw = b"0_1_d7|1|{HH}|hello|neutral\r\n0_0_d8|0|{AY}|other|neutral\n"
    rows = runner.select_rows(raw, {"0_1_d7"})
    assert rows[0][0] == raw.splitlines(keepends=True)[0]


@pytest.mark.parametrize(
    "raw",
    [
        b"0_1_d7|1|x|a|n\n0_1_d7|1|x|a|n\n",
        b"0_1_d7|1|missing-fields\n",
        b"0_0_d8|0|x|b|n\n",
    ],
)
def test_reject_duplicate_malformed_or_missing_original_rows(raw):
    with pytest.raises(ValueError):
        runner.select_rows(raw, {"0_1_d7"})


def test_history_is_original_last_ten_not_all_or_zero_history():
    raw = b"".join(f"{i}_0_d7|0|x|turn{i}|n\n".encode() for i in range(12))
    rows = runner.select_rows(raw, {f"{i}_0_d7" for i in range(12)})
    plan = runner.history_plan(rows)
    assert plan[0]["history_ids"] == []
    assert plan[1]["history_ids"] == ["0_0_d7"]
    assert plan[-1]["history_ids"] == [f"{i}_0_d7" for i in range(1, 11)]
    assert plan[-1]["feature_path"] == "text_emb/0-text_emb-11_0_d7.npy"


def test_missing_history_turn_rejected():
    rows = runner.select_rows(b"1_0_d7|0|x|a|n\n", {"1_0_d7"})
    with pytest.raises(ValueError, match="contiguous"):
        runner.history_plan(rows)


def test_fixture_materialization_verify_and_tamper(tmp_path, monkeypatch):
    import prepare_dailytalk_original_audio as audio
    import prepare_dailytalk_original_metadata as metadata

    source = tmp_path / "metadata"
    processed = source / "preprocessed_data/DailyTalk"
    processed.mkdir(parents=True)
    original = b"0_1_d7|1|{HH}|hello|neutral\r\n"
    (processed / "val_phone.txt").write_bytes(original)
    for name in runner.MAPS:
        (processed / name).write_bytes(b'{"original": 1}\n')
    monkeypatch.setattr(audio, "prepare", lambda args: None)
    monkeypatch.setattr(audio, "selected_ids", lambda *args: {"0_1_d7"})
    monkeypatch.setattr(metadata, "prepare", lambda args: None)
    args = SimpleNamespace(
        metadata_root=source,
        audio_root=tmp_path / "audio",
        output=tmp_path / "output",
        action="prepare",
    )
    report = runner.prepare(args)
    assert report["original_utterances"] == 1 and report["model_calls"] == 0
    assert (args.output / "val_phone.fixed3.private.txt").read_bytes() == original
    contract = json.loads((args.output / "guo-input-contract.private.json").read_text())
    assert not contract["embeddings_generated"] and not contract["phonemes_recomputed"]
    with pytest.raises(ValueError, match="fresh"):
        runner.prepare(args)
    args.action = "verify"
    runner.prepare(args)
    (args.output / "val_phone.fixed3.private.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        runner.prepare(args)


def test_output_cannot_overlap_original_inputs(tmp_path):
    with pytest.raises(ValueError, match="separate external"):
        runner.prepare(
            SimpleNamespace(
                metadata_root=tmp_path / "metadata",
                audio_root=tmp_path / "audio",
                output=tmp_path,
                action="prepare",
            )
        )
