"""Inert sampling/plan fixtures; not NPU execution or source delivery evidence."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "cceval_native_generation", SCRIPTS / "run_crosscodeeval_original.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_plan_does_not_need_model_or_create_outputs(tmp_path):
    output = tmp_path / "absent-output"
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_crosscodeeval_original.py"),
            "--author-root",
            str(tmp_path / "absent-source"),
            "--data-archive",
            str(tmp_path / "absent-archive"),
            "--model-root",
            str(tmp_path / "absent-model"),
            "--output",
            str(output),
            "--start",
            "3",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    plan = json.loads(process.stdout)
    assert plan["count"] == 3 and plan["start"] == 3
    assert plan["model_max_tokens"] == 16384
    assert plan["crossfile_max_tokens"] == 12800
    assert plan["generation_max_tokens"] == 50
    assert plan["temperature"] == 0.2 and plan["top_p"] == 0.95
    assert not output.exists()


def test_fixed_samples_and_full_keep_exact_original_lines(tmp_path):
    archive = tmp_path / "fixture.tar.xz"
    rows = [
        {"prompt": f"inert{index}", "metadata": {"task_id": str(index)}}
        for index in range(7)
    ]
    lines = [json.dumps(row).encode() + b"\n" for row in rows]
    with tarfile.open(archive, "w:xz") as bundle:
        item = tarfile.TarInfo("java/line_completion_rg1_unixcoder_cosine_sim.jsonl")
        item.size = len(b"".join(lines))
        bundle.addfile(item, io.BytesIO(b"".join(lines)))
    args = (archive, "java", "line_completion_rg1_unixcoder_cosine_sim")
    assert MODULE.select_rows(*args, 3, 3) == (lines[3:6], rows[3:6])
    assert MODULE.select_rows(*args, 0, None) == (lines, rows)
    with pytest.raises(ValueError, match="beyond"):
        MODULE.select_rows(*args, 6, 3)


def test_model_artifact_is_original_unquantized_checkpoint():
    pins = json.loads(MODULE.MODEL_PINS.read_text(encoding="utf-8"))
    assert pins["repo_id"] == "bigcode/starcoder2-3b"
    assert pins["storage_bytes"] == sum(item["bytes"] for item in pins["files"])
    assert len(pins["files"]) == 8
    assert pins["files"][-1]["path"] == "model.safetensors"
    assert pins["files"][-1]["bytes"] == 12121539344
