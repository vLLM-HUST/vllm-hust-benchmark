"""Original task workspace preparation fixtures, never execute task programs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "mlagent_workspace", ROOT / "scripts/prepare_mlagentbench_cifar_workspace.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_plan_does_not_open_data_or_import_runtime(tmp_path):
    result = MODULE.prepare(SimpleNamespace(action="plan"))
    assert result["model_calls"] == result["native_tasks_run"] == 0
    assert not result["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())


@pytest.fixture
def task(tmp_path, monkeypatch):
    author, data = tmp_path / "author", tmp_path / "data"
    author.mkdir()
    data.mkdir()
    entries = []
    for name in ["scripts/prepare.py", "scripts/eval.py", "env/train.py"]:
        relative = MODULE.TASK_PREFIX + name
        path = author / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"# fixture source only, never executed\n"
        path.write_bytes(raw)
        entries.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"files": entries}), encoding="utf-8")
    raw = b"fixture only, not a real CIFAR archive or pickle"
    (data / "original.tar.gz").write_bytes(raw)
    profile = tmp_path / "assets.json"
    profile.write_text(
        json.dumps(
            {
                "data": {
                    "path": "original.tar.gz",
                    "bytes": len(raw),
                    "md5": hashlib.md5(raw).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(MODULE, "PROFILE", profile)
    monkeypatch.setattr(MODULE, "SOURCE_PINS", pins)
    monkeypatch.setattr(MODULE, "verify_source", lambda root, pins: root.resolve())
    return SimpleNamespace(
        action="prepare",
        author_root=author,
        data_root=data,
        output=tmp_path / "workspace",
    )


def test_original_files_and_archive_preserved_without_extraction(task):
    result = MODULE.prepare(task)
    assert result["task_files"] == 3
    assert not result["archive_extracted"]
    assert not result["scientific_training_executed"]
    assert (task.output / "env/data/original.tar.gz").read_bytes() == (
        task.data_root / "original.tar.gz"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        MODULE.prepare(task)
    task.action = "verify"
    assert MODULE.prepare(task) == result


def test_wrong_original_archive_rejected_before_workspace_creation(task):
    (task.data_root / "original.tar.gz").write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="exact original"):
        MODULE.prepare(task)
    assert not task.output.exists()


def test_changed_source_rejected(task):
    MODULE.prepare(task)
    (task.output / "env/train.py").write_text("# modified", encoding="utf-8")
    task.action = "verify"
    with pytest.raises(ValueError, match="differs"):
        MODULE.prepare(task)


def test_extra_program_rejected(task):
    MODULE.prepare(task)
    (task.output / "env/new.py").write_text("# not original", encoding="utf-8")
    task.action = "verify"
    with pytest.raises(ValueError, match="only original"):
        MODULE.prepare(task)


def test_original_asset_root_cannot_be_modified(task):
    task.output = task.data_root / "workspace"
    with pytest.raises(ValueError, match="separate"):
        MODULE.prepare(task)
    assert not task.output.exists()


def test_actual_preparation_receipt_does_not_claim_agent_delivery():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/mlagentbench-cifar-original-preparation-20261101.json"
        ).read_text("utf-8")
    )
    public = receipt["public_workspace"]
    assert (
        public["entry_sha256"]
        == hashlib.sha256(
            (ROOT / public["entry"]).read_text("utf-8").encode()
        ).hexdigest()
    )
    assert public["exit_codes"] == [0, 0, 0]
    assert receipt["original_prepare"]["exit_code"] == 0
    assert receipt["original_prepare"]["entry_modified"] is False
    assert receipt["original_prepare"]["network"] == "none"
    assert receipt["native_agent_tasks_run"] == 0
    assert receipt["source_checkbox_eligible"] is False
