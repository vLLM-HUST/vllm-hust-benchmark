"""Pinned official software/plan fixtures only, not speech or NPU synthesis."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_preparation_receipt_does_not_claim_native_synthesis():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/ljspeech-vits-preparation-20261101.json"
        ).read_text("utf-8")
    )
    assert receipt["checkpoint"]["bytes"] == 145599717
    assert receipt["checkpoint"]["safe_weights_only_cpu_deserialization_verified"]
    assert receipt["checkpoint"]["state_tensors"] == 838
    assert not receipt["checkpoint"]["model_executed"]
    assert not receipt["native_tts_verified"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["model_calls"] == 0
    assert {"phonemizer", "espeak"} <= set(receipt["cpu_dependency_probe"]["missing"])


def test_original_vits_source_and_inference_defaults_are_pinned(tmp_path):
    profile = json.loads(
        (ROOT / "configs/workload-native/ljspeech-vits-author-source.json").read_text(
            "utf-8"
        )
    )
    assert profile["source_id"] == "LJSPEECH"
    assert profile["repository"] == "jaywalnut310/vits"
    assert len(profile["revision"]) == 40
    files = profile["files"]
    assert len(files) == len({item["path"] for item in files}) == 23
    assert sum(item["bytes"] for item in files) == 127230
    assert {
        "models.py",
        "modules.py",
        "commons.py",
        "utils.py",
        "inference.ipynb",
        "requirements.txt",
        "configs/ljs_base.json",
        "text/cleaners.py",
        "text/symbols.py",
        "monotonic_align/setup.py",
        "monotonic_align/core.pyx",
        "LICENSE",
    } <= {item["path"] for item in files}
    assert not any(
        item["path"].startswith("filelists/") or item["path"].endswith((".pth", ".wav"))
        for item in files
    )
    inference = profile["original_inference"]
    assert inference["noise_scale"] == 0.667 and inference["noise_scale_w"] == 0.8
    assert inference["length_scale"] == 1 and inference["sampling_rate"] == 22050
    assert inference["cleaners"] == ["english_cleaners2"]
    assert not inference["npu_runtime_qualified"]
    assert not profile["official_checkpoint_candidate"][
        "checkpoint_acquisition_performed_by_source_getter"
    ]
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "ljspeech-vits",
            "--author-root",
            str(target),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["source_id"] == "LJSPEECH" and plan["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert plan["files"] == 23 and plan["model_calls"] == 0
    assert not plan["source_checkbox_eligible"] and not target.exists()
