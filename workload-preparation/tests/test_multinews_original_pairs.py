"""Small field fixtures, not native model or corpus-quality evidence."""

import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.assets import sha256
from vllm_hust_workload_data.multinews import original_rows
from vllm_hust_workload_data.registry import adapter_spec


def pair_fixture(tmp_path, monkeypatch, *, target_lines=2):
    root = tmp_path / "data"
    root.mkdir()
    source = root / "test.src.cleaned"
    target = root / "test.tgt"
    source.write_text("  First NEWLINE_CHAR second ||||| Third  \n\n", encoding="utf-8")
    target.write_text(
        "- DO_NOT_LEAK_REFERENCE\n"
        + ("Other reference\n" if target_lines == 2 else ""),
        encoding="utf-8",
    )
    files = [
        {
            "path": "data/test.src.cleaned",
            "role": "data",
            "paired_target": "data/test.tgt",
            "rows": 2,
            "sha256": sha256(source),
            "bytes": source.stat().st_size,
        },
        {
            "path": "data/test.tgt",
            "role": "reference",
            "rows": target_lines,
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
        },
    ]
    monkeypatch.setattr(
        "vllm_hust_workload_data.multinews.source_spec", lambda _: {"files": files}
    )
    return source, target


def test_original_loader_fields_and_pairing_are_preserved(tmp_path, monkeypatch):
    source, target = pair_fixture(tmp_path, monkeypatch)
    first, second = list(iter_source_rows(source, adapter_spec("MULTINEWS")))
    assert first["document"] == "First \n second ||||| Third"
    assert first["summary"] == "- DO_NOT_LEAK_REFERENCE"
    assert first["source_line_index"] == 0
    assert first["target_file"] == "data/test.tgt"
    assert first["target_sha256"] == sha256(target)
    assert second["document"] == ""  # Original blank lines are not filtered.
    record = adapt_row(
        "MULTINEWS",
        first,
        source={"source_id": "MULTINEWS", "file_sha256": sha256(source)},
        spec=adapter_spec("MULTINEWS"),
        index=0,
    )
    assert record.task["input"] == {"document": first["document"]}
    assert record.outcome_record() == {"summary": first["summary"]}
    assert record.replay["unmapped_field_names"] == []
    assert "DO_NOT_LEAK_REFERENCE" not in json.dumps(record.input_record())
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_original_reference_line_is_not_silently_dropped(tmp_path, monkeypatch):
    source, _ = pair_fixture(tmp_path, monkeypatch, target_lines=1)
    with pytest.raises(ValueError, match="line counts differ"):
        list(original_rows(source))


def test_changed_reference_bytes_fail_before_adaptation(tmp_path, monkeypatch):
    source, target = pair_fixture(tmp_path, monkeypatch)
    target.write_text("changed reference", encoding="utf-8")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        list(original_rows(source))
