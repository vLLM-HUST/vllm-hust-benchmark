"""Plan/identity fixtures only, not original model or perplexity validation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "temporal_native", ROOT / "scripts/run_temporalwiki_original_npu.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture(tmp_path, monkeypatch):
    author = tmp_path / "author"
    (author / "models").mkdir(parents=True)
    (author / "models/GPT2_Model.py").write_text(
        "raise RuntimeError('never execute fixture')"
    )
    (author / "evaluation_ppl.py").write_text(
        "raise RuntimeError('never execute fixture')"
    )
    data = tmp_path / "data"
    data.mkdir()
    files = []
    for i in range(8):
        raw = b"subject,relation,object\ns,r,o\n"
        (data / f"{i}.csv").write_bytes(raw)
        files.append(
            {
                "path": f"{i}.csv",
                "role": "data",
                "rows": 1,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    profile = tmp_path / "data-pins.json"
    profile.write_text(json.dumps({"files": files}))
    monkeypatch.setattr(runner, "DATA_PINS", profile)
    monkeypatch.setattr(runner, "verify_source", lambda *_: author)
    monkeypatch.setattr(
        runner, "baseline_factory_tree", lambda _: (None, ["training-only"])
    )
    monkeypatch.setattr(
        runner,
        "npu_evaluator_tree",
        lambda *_: (None, {"tensor_cuda_to_npu": 14, "model_cuda_to_npu": 1}),
    )
    return SimpleNamespace(
        author_root=author,
        data_root=data,
        model_root=tmp_path / "not-acquired",
        output=tmp_path / "not-created",
        device="npu:0",
        seed=42,
        allocator_budget_gib=None,
        execute=False,
        full=False,
    )


def test_plan_never_loads_model_or_creates_output(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "verify_model",
        lambda *_: pytest.fail("plan must not require or load weights"),
    )
    result = runner.run(args)
    assert result["status"] == "PLAN_ONLY"
    assert result["original_probe_files"] == result["selected_rows"] == 8
    assert result["original_input_length"] == result["original_output_length"] == 25
    assert result["original_batch_size"] == 32
    assert result["model_calls"] == 0 and not result["native_execution_complete"]
    assert not result["source_checkbox_eligible"]
    assert not args.output.exists()


def test_changed_original_data_is_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    (args.data_root / "0.csv").write_text("different")
    with pytest.raises(ValueError, match="asset identity"):
        runner.run(args)


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu:-1"])
def test_no_other_neural_device(tmp_path, monkeypatch, device):
    args = fixture(tmp_path, monkeypatch)
    args.device = device
    with pytest.raises(ValueError, match="explicit NPU"):
        runner.run(args)


def test_original_model_size_and_digest_required(tmp_path):
    raw = b"fixture-weight-not-a-real-model"
    path = tmp_path / "model.safetensors"
    path.write_bytes(raw)
    pins = {
        "model": "openai-community/gpt2-large",
        "files": [
            {
                "path": path.name,
                "bytes": len(raw),
                "lfs_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    runner.verify_model(tmp_path, pins)
    path.write_bytes(b"X" * len(raw))
    with pytest.raises(ValueError, match="identity differs"):
        runner.verify_model(tmp_path, pins)


def test_model_profile_is_explicit_base_probe_not_trained_checkpoint():
    pins = json.loads(runner.MODEL_PINS.read_text("utf-8"))
    assert pins["model"] == "openai-community/gpt2-large"
    assert len(pins["files"]) == 8
    assert sum(item["bytes"] for item in pins["files"]) == 3250026117
    assert "empty checkpoint_path" in pins["protocol"]
