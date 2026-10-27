"""Original task selection/database contracts; fixtures do not validate an Agent."""

import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "deepplanning_shopping", ROOT / "scripts/run_deepplanning_original_shopping.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_original_order_fixed_three_or_all():
    rows = [{"id": value, "query": "original fixture"} for value in [7, 2, 9, 0]]
    assert MODULE.select_tasks(rows) == [7, 2, 9]
    assert MODULE.select_tasks(rows, full=True) == [7, 2, 9, 0]
    with pytest.raises(ValueError):
        MODULE.select_tasks([{"id": 1}, {"id": 1}])


def test_original_selected_database_bytes_and_unselected_exclusion(tmp_path):
    original = tmp_path / "database.tar.gz"
    with tarfile.open(original, "w:gz") as archive:
        for case in [7, 2]:
            member = tarfile.TarInfo(f"database_level1/case_{case}/products.jsonl")
            content = str(case).encode()
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    output = tmp_path / "out"
    MODULE.extract_cases(original, output, 1, [7])
    assert (output / "case_7/products.jsonl").read_bytes() == b"7"
    assert not (output / "case_2").exists()
    with pytest.raises(ValueError, match="missing"):
        MODULE.extract_cases(original, tmp_path / "missing", 1, [3])


def test_no_task_or_token_budget_replacement():
    source = (ROOT / "scripts/run_deepplanning_original_shopping.py").read_text("utf-8")
    assert "max_llm_calls=400" in source
    assert "rerun_ids=ids" in source
    assert 'native["failed"]' in source
    assert "generate_summary_report(scores, reports)" in source
    assert "enable_thinking" not in source
    assert "max_tokens" not in source
