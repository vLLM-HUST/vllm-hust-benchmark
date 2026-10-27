import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "chartqa_view",
    Path(__file__).parents[1] / "scripts/prepare_chartqa_original_view.py",
)
view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view)


def rows():
    return [
        {
            "query": f'原题\t"{i}"\n完整问题',
            "label": ["0", "否\n是"],
            "image": {"bytes": b"original image bytes\x00" + bytes([i])},
            "human_or_machine": i % 2,
        }
        for i in range(3)
    ]


def test_lossless_original_fields_and_image_bytes(tmp_path):
    target = tmp_path / "view.tsv"
    view.write_view(rows(), target)
    assert view.verify_view(rows(), target) == 3
    with pytest.raises(FileExistsError):
        view.write_view(rows(), target)


@pytest.mark.parametrize("field", ["query", "label", "image", "human_or_machine"])
def test_changed_original_relation_rejected(tmp_path, field):
    target = tmp_path / "view.tsv"
    original = rows()
    view.write_view(original, target)
    changed = rows()
    changed[0][field] = {
        "query": "different",
        "label": ["different"],
        "image": {"bytes": b"different"},
        "human_or_machine": 9,
    }[field]
    with pytest.raises(ValueError, match="changed"):
        view.verify_view(changed, target)


def test_missing_and_extra_rows_rejected(tmp_path):
    target = tmp_path / "view.tsv"
    view.write_view(rows(), target)
    with pytest.raises(ValueError, match="additional"):
        view.verify_view(rows()[:2], target)
    with pytest.raises(ValueError, match="count lost"):
        view.verify_view(rows() + rows()[:1], target)


def test_original_view_pins_match_shared_public_source_catalog():
    from vllm_hust_workload_data.registry import source_spec

    source = source_spec("CHARTQA")
    original = next(
        item
        for item in source["files"]
        if item["path"] == "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"
    )
    assert source["revision"] == view.REVISION
    assert original["sha256"] == view.SHA256
    assert original["bytes"] == view.BYTES
    assert original["rows"] == view.ROWS
