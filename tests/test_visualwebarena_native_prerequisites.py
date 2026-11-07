"""Invented integrity controls, not original site or Agent execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vwa_prerequisites", ROOT / "scripts/check_visualwebarena_native_prerequisites.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(root, *, ids=(0, 1, 2), launcher=None):
    texts = {
        "run.py": launcher
        or (
            "if DATASET == 'visualwebarena':\n"
            "    eval_caption_image_fn = image_utils.get_captioning_fn(\n"
            'choices=["cpu", "cuda"]\n'
        ),
        "environment_docker/README.md": "invented fixture deployment only",
    }
    for name in runner.TASK_FILES:
        texts[name] = json.dumps(
            [
                {
                    "task_id": i,
                    "sites": ["classifieds"],
                    "eval": {"eval_types": ["url_match"]},
                    "image": None,
                    "require_reset": False,
                }
                for i in ids
            ]
        )
    entries = []
    for name, text in texts.items():
        raw = text.encode()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(
            dict(
                path=name,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                git_blob=hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            )
        )
    pins = root / "fixture-pins.json"
    pins.write_text(
        json.dumps(
            dict(source_id="VISUALWEBARENA", revision="invented-fixture", files=entries)
        )
    )
    return pins


def test_url_metrics_do_not_hide_original_eager_captioner_requirement(tmp_path):
    result = runner.inspect_original(tmp_path, pins_path=fixture(tmp_path))
    assert result["original_task_definitions"] == 9
    assert [r["task_id"] for r in result["fixed_original_tasks"]] == [0, 1, 2]
    assert all(r["eval_types"] == ["url_match"] for r in result["fixed_original_tasks"])
    assert result["original_launcher_initializes_eval_captioner_even_for_url_match"]
    assert result["original_cli_eval_captioner_devices"] == ["cpu", "cuda"]
    assert not result["npu_captioner_execution_qualified"]
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]


def test_tampered_original_launcher_rejected(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / "run.py").write_text("modified")
    with pytest.raises(ValueError):
        runner.inspect_original(tmp_path, pins_path=pins)


def test_changed_upstream_captioner_contract_requires_review(tmp_path):
    pins = fixture(tmp_path, launcher="invented changed source")
    with pytest.raises(ValueError, match="captioner/device contract"):
        runner.inspect_original(tmp_path, pins_path=pins)


@pytest.mark.parametrize("ids", [(0, 0, 2), (1, 2, 3)])
def test_no_duplicate_or_reselected_fixed_tasks(tmp_path, ids):
    pins = fixture(tmp_path, ids=ids)
    with pytest.raises(ValueError, match="distinct original|fixed original"):
        runner.inspect_original(tmp_path, pins_path=pins)


def test_npu_captioner_component_does_not_mark_browser_source_delivered():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    entry = ledger["delivery_inventory"]["VISUALWEBARENA"]
    component = entry["original_caption_helper_npu_component_20261107"]
    receipt = json.loads((ROOT / component["receipt"]).read_text("utf-8"))
    assert not entry["checked"]
    assert not component["source_checkbox_eligible"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["native_browser_tasks_run"] == receipt["native_scores_produced"] == 0
    assert receipt["control"]["model_calls"] == 1
    assert receipt["control"]["original_max_new_tokens"] == 32
    assert receipt["control"]["prompt_argument"] is None
    assert receipt["original_helper_unmodified"]
    assert receipt["compatibility_profile"]["all_parameter_devices"] == ["npu:0"]
    assert not receipt["compatibility_profile"]["cpu_neural_fallback"]
    assert len(receipt["weight_shards"]) == 2
    assert sum(row["bytes"] for row in receipt["weight_shards"]) == 15769954544
