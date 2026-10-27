"""CPU original-history/database fixtures, not an actual Agent run."""

import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "travel_saved", ROOT / "scripts/verify_deepplanning_travel_saved.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def history():
    return {
        "messages": [
            {"role": "system", "content": "original"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "tool_calls": [{"id": "call1"}]},
            {"role": "tool", "tool_call_id": "call1", "content": "native result"},
            {"role": "assistant", "content": "report", "tool_calls": None},
        ]
    }


def test_complete_original_tool_relationship():
    assert MODULE.verify_history(history(), "question", "original") == {
        "model_messages": 2,
        "tool_pairs": 1,
    }


@pytest.mark.parametrize("mutation", ["prompt", "missing", "wrong_id", "injected"])
def test_changed_or_truncated_original_history_rejected(mutation):
    payload = history()
    if mutation == "prompt":
        payload["messages"][0]["content"] = "changed"
    elif mutation == "missing":
        del payload["messages"][3]
    elif mutation == "wrong_id":
        payload["messages"][3]["tool_call_id"] = "other"
    else:
        payload["messages"].append({"role": "user", "content": "replacement question"})
    with pytest.raises(ValueError):
        MODULE.verify_history(payload, "question", "original")


def test_original_database_bytes_and_missing_members(tmp_path):
    archive = tmp_path / "database.zip"
    database = tmp_path / "database_en"
    case = database / "id_0"
    case.mkdir(parents=True)
    (case / "table.csv").write_bytes(b"original")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("database_en/id_0/table.csv", b"original")
    assert MODULE.verify_database(archive, database, "en", "0") == 1
    (case / "table.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        MODULE.verify_database(archive, database, "en", "0")
    with pytest.raises(ValueError, match="absent"):
        MODULE.verify_database(archive, database, "en", "1")


def test_running_native_task_cannot_be_audited_as_complete(tmp_path, monkeypatch):
    assets, native, output = (
        tmp_path / name for name in ("assets", "native", "fresh-audit")
    )
    assets.mkdir()
    native.mkdir()
    (native / "native-public.json").write_text(
        json.dumps(
            {"status": "RUNNING_ORIGINAL_TRAVEL_TASKS", "native_max_llm_calls": 400}
        )
    )
    monkeypatch.setattr(MODULE, "prepare", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="not terminal"):
        MODULE.audit(assets, native, output)
    assert not output.exists()
