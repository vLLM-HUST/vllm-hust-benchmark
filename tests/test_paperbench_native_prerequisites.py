"""Invented offline recipe controls, not original PaperBench execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pb_prerequisites", ROOT / "scripts/check_paperbench_native_prerequisites.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path):
    texts = [
        "Miniconda3-py313_25.5.1-0-Linux-x86_64.sh\nconda tos accept\n",
        "FROM invented-control-only\n",
        "--platform=linux/amd64\n--platform=linux/amd64\n",
        "time_limit: int | None = chz.field(default=5 * 60\n",
        "paperbench.reproduction.timeout=60\n",
    ]
    entries = []
    for name, text in zip(runner.RECIPES, texts, strict=True):
        raw = text.encode()
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        )
    pins = tmp_path / "fixture-pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "PAPERBENCH",
                "revision": "invented-control",
                "files": entries,
            }
        )
    )
    return pins


@pytest.mark.parametrize(
    "architecture,match", [("aarch64", False), ("x86_64", True), ("AMD64", True)]
)
def test_architecture_never_promotes_preflight_to_execution(
    tmp_path, architecture, match
):
    pins = fixture(tmp_path)
    result = runner.inspect_original(
        tmp_path, pins_path=pins, host_architecture=architecture
    )
    assert result["native_host_architecture_matches"] is match
    assert result["verified_original_files"] == 5
    assert result["original_recipe_auto_accepts_anaconda_terms"]
    assert not result["vendor_terms_accepted_by_this_check"]
    assert not result["cross_arch_execution_qualified"]
    assert not result["docker_built_or_run"]
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]


def test_modified_source_is_rejected_before_recipe_inspection(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / runner.RECIPES[0]).write_text("changed")
    with pytest.raises(ValueError):
        runner.inspect_original(tmp_path, pins_path=pins)


def test_missing_required_recipe_is_rejected(tmp_path):
    pins = fixture(tmp_path)
    value = json.loads(pins.read_text())
    value["files"].pop()
    pins.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pins required"):
        runner.inspect_original(tmp_path, pins_path=pins)
