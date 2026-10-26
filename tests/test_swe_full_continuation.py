"""Keep valid negative task results distinct from coverage and infrastructure."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/swe-full-continuation-20261026"


def test_public_agreement():
    subprocess.run([sys.executable, str(REPORT / "verify.py")], check=True)


def test_complete_v1_source_coverage():
    lock = json.loads(
        (ROOT / "src/vllm_hust_benchmark/data/swe_bench_pro_v1_lock.json").read_text()
    )
    source = lock["official_v1"]
    pins = (
        source["sha256"]
        | source["nodebb_task_source_sha256"]
        | source.get("grading_dockerfiles_sha256", {})
    )
    task_ids = {
        p.split("/")[1]
        for p in pins
        if p.startswith("run_scripts/") and p.endswith("/run_script.sh")
    }
    complete = []
    for identifier in task_ids:
        paths = [
            f"run_scripts/{identifier}/{name}"
            for name in ["run_script.sh", "parser.py"]
        ] + [
            f"dockerfiles/{kind}/{identifier}/Dockerfile"
            for kind in ["base_dockerfile", "instance_dockerfile"]
        ]
        if all(p in pins for p in paths):
            complete.append(identifier)
    assert len(complete) == 731
    assert (
        source["commit"]
        == "66f92766bba642462d4bbe5479e83f91f9211862"  # pragma: allowlist secret
    )
    assert (
        lock["dataset"]["parquet_sha256"]
        == "c8cd7115496ad4e9a8b21d088cef576a65bf821bb542b24336f13f714cef13f8"  # pragma: allowlist secret
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_calls", 99),
        ("resolved", True),
        ("unresolved", 1),
        ("trajectory_sha256", "0" * 64),
    ],
)
def test_reject_changed_observation(tmp_path, field, value):
    for name in ["verify.py", "independent-review.json", "storage-review.json"]:
        shutil.copyfile(REPORT / name, tmp_path / name)
    p = tmp_path / "independent-review.json"
    value_dict = json.loads(p.read_text())
    value_dict[field] = value
    p.write_text(json.dumps(value_dict))
    result = subprocess.run(
        [sys.executable, str(tmp_path / "verify.py")], capture_output=True
    )
    assert result.returncode != 0
