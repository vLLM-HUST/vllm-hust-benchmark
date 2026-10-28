"""Offline protocol/manifest fixtures, not original model evaluation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "temporalwiki_contract", ROOT / "scripts/verify_temporalwiki_original_contract.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_pinned_software_manifest_contains_no_datasets_or_checkpoints():
    pins = json.loads(runner.PINS.read_text("utf-8"))
    assert pins["source_id"] == "TEMPORALWIKI"
    assert len(pins["files"]) == 177
    assert sum(item["bytes"] for item in pins["files"]) == 382667
    assert sum(item["path"].endswith(".py") for item in pins["files"]) == 13
    assert sum(item["path"].endswith(".json") for item in pins["files"]) == 162
    assert len({item["path"] for item in pins["files"]}) == 177
    assert not any(
        item["path"].endswith((".zip", ".csv", ".pt", ".ckpt"))
        for item in pins["files"]
    )


def fixture(tmp_path):
    files = []
    for name, raw in (
        ("evaluation.py", b"raise RuntimeError('never import author code')\n"),
        ("configs/original.json", b'{"mode":"evaluate_ppl"}'),
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        )
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps({"revision": "fixture", "files": files}), encoding="utf-8"
    )
    return pins


def test_verifier_reads_without_importing_or_executing_author_code(
    tmp_path, monkeypatch
):
    pins = fixture(tmp_path)
    monkeypatch.setattr(runner, "PINS", pins)
    result = runner.inspect_contract(tmp_path)
    assert result["configuration_modes"] == {"evaluate_ppl": 1}
    assert result["python_ast_parsed_not_imported"] == 1
    assert not result["source_checkbox_eligible"]
    assert not result["task_data_acquired"]
    assert result["model_calls"] == result["native_tasks_run"] == 0


def test_verifier_rejects_changed_original_configuration(tmp_path, monkeypatch):
    pins = fixture(tmp_path)
    monkeypatch.setattr(runner, "PINS", pins)
    (tmp_path / "configs/original.json").write_text(
        '{"mode":"evaluate"}', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        runner.inspect_contract(tmp_path)


def test_source_plan_does_not_download_or_mark_delivery(tmp_path):
    import subprocess

    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "temporalwiki",
            "--author-root",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 177 and report["bytes"] == 382667
    assert not report["source_checkbox_eligible"]
    assert not target.exists()


def test_readiness_is_not_native_completion():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    item = ledger["delivery_inventory"]["TEMPORALWIKI"]
    receipt = json.loads((ROOT / item["preparation_receipt"]).read_text("utf-8"))
    assert not item["checked"] and item["delivery_receipt"] is None
    assert receipt["original_source_verification"]["files"] == 177
    assert (
        sum(receipt["original_source_verification"]["configuration_modes"].values())
        == 162
    )
    assert not receipt["dataset_acquisition"]["all_original_archives_acquired"]
    probe = receipt["dataset_acquisition"]["original_probe_archive"]
    assert probe["whole_original_archive_verified"] and probe["bytes"] == 620459
    assert not receipt["native_protocol"]["qwen_chat_qa_substitution"]
    assert receipt["model_calls"] == receipt["native_tasks_run"] == 0
