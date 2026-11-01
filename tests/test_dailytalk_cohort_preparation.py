"""General complete-dialogue preparation fixtures, never original benchmark runs."""

import hashlib
import json
import importlib.util
from pathlib import Path
import sys
import zipfile
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "dailytalk_cohort", ROOT / "scripts/prepare_dailytalk_original_guo_inputs.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)
RAW = b"0_0_d7|0|{P}|first|neutral\r\n0_1_d8|1|{P}|other|neutral\n1_1_d7|1|{P}|second|neutral\n"


def test_select_any_complete_dialogue_or_entire_split_without_reordering():
    chosen = ENTRY.cohort_rows(RAW, ["7"])
    assert (
        b"".join(raw for raw, _ in chosen)
        == RAW.splitlines(keepends=True)[0] + RAW.splitlines(keepends=True)[2]
    )
    assert ENTRY.history_plan(chosen)[1]["history_ids"] == ["0_0_d7"]
    assert b"".join(raw for raw, _ in ENTRY.cohort_rows(RAW)) == RAW


@pytest.mark.parametrize("selection", [[], ["7", "7"], ["unknown"]])
def test_unknown_duplicate_or_empty_dialogue_selection_rejected(selection):
    with pytest.raises(ValueError):
        ENTRY.cohort_rows(RAW, selection)


def test_missing_original_history_is_not_reconstructed():
    with pytest.raises(ValueError, match="contiguous"):
        ENTRY.cohort_rows(b"1_0_d7|0|{P}|missing previous|neutral\n", ["7"])


@pytest.fixture
def audio(tmp_path):
    root = tmp_path / "audio"
    original = tmp_path / "original-fixture.zip"
    rows = ENTRY.cohort_rows(RAW, ["7"])
    items = []
    with zipfile.ZipFile(original, "w") as archive:
        for _, fields in rows:
            name = f"dailytalk/data/7/{fields[0]}.wav"
            raw = b"fixture original member bytes: " + fields[0].encode()
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            archive.writestr(name, raw)
            items.append(
                {
                    "path": name,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "crc32": zlib.crc32(raw),
                }
            )
    return (
        root,
        original,
        rows,
        {"files": items, "archive": {"bytes": original.stat().st_size}},
    )


def test_existing_pinned_media_can_be_reused_without_archive_or_network(audio):
    root, _, rows, profile = audio
    assert (
        ENTRY.verify_cohort_audio(root, rows, profile)
        == "publisher-pinned-selected-member-sha256-and-crc"
    )


def test_non_pinned_complete_dialogue_requires_actual_archive_bytes(audio):
    root, archive, rows, profile = audio
    profile["files"] = []
    with pytest.raises(ValueError, match="original publisher ZIP"):
        ENTRY.verify_cohort_audio(root, rows, profile)
    assert "not-published-full-archive-digest" in ENTRY.verify_cohort_audio(
        root, rows, profile, archive
    )
    target = root / "dailytalk/data/7/0_0_d7.wav"
    raw = target.read_bytes()
    target.write_bytes(b"X" + raw[1:])
    with pytest.raises(ValueError, match="differs from original archive"):
        ENTRY.verify_cohort_audio(root, rows, profile, archive)


def test_extra_or_missing_audio_turn_is_not_accepted(audio):
    root, _, rows, profile = audio
    (root / "dailytalk/data/7/9_0_d7.wav").write_bytes(b"not an original turn")
    with pytest.raises(ValueError, match="missing or extra"):
        ENTRY.verify_cohort_audio(root, rows, profile)


def test_actual_original_selected_cohort_receipt_matches_public_entry():
    report = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dailytalk-general-cohort-preparation-20261101.json"
        ).read_text("utf-8")
    )
    assert (
        hashlib.sha256((ROOT / report["entry"]).read_text("utf-8").encode()).hexdigest()
        == report["entry_sha256"]
    )
    assert report["exit_codes"] == [0, 0, 0]
    assert report["original_utterances"] == 12
    assert report["original_rows_and_history_equal_prior_native_cohort_subset"]
    assert report["full_original_split_metadata_and_history_plan_rows"] == {
        "train": 22479,
        "val": 1197,
    }
    assert not report["torch_imported"] and report["new_model_calls"] == 0
    assert not report["non_pinned_full_archive_branch_actual_original_archive_verified"]
    assert not report["source_checkbox_eligible"]
