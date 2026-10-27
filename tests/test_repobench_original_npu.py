"""Native port contract fixtures, not original model/NPU execution evidence."""

from contextlib import nullcontext
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "repobench_npu", ROOT / "scripts/run_repobench_original_npu.py"
)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_fixed_first_rows_per_stratum_preserve_original_positions():
    rows = [
        {"level": "2k", "next_line": "a"},
        {"level": "4k", "next_line": "b"},
        {"level": "2k", "next_line": "c"},
    ]
    assert RUN.select_rows(rows, 1) == [(0, rows[0]), (1, rows[1])]
    assert RUN.select_rows(rows, None) == list(enumerate(rows))
    assert "next_line" not in RUN.INPUT_FIELDS and "all_code" not in RUN.INPUT_FIELDS
    assert "gold_snippet_index" not in RUN.INPUT_FIELDS
    with pytest.raises(ValueError):
        RUN.select_rows(rows, 0)


def test_asset_verifier_rejects_same_size_changed_file(tmp_path):
    payload = b"fixture original"
    path = tmp_path / "model.bin"
    path.write_bytes(payload)
    specs = {path.name: (len(payload), hashlib.sha256(payload).hexdigest())}
    RUN.verify_files(tmp_path, specs)
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        RUN.verify_files(tmp_path, specs)


@pytest.mark.parametrize("device_type", ["npu", "cpu"])
def test_generation_keeps_native_sampling_budget_and_only_decodes_suffix(device_type):
    calls = {}

    class Tokens:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values),)

        def __getitem__(self, key):
            return Tokens(self.values[key])

        def tolist(self):
            return self.values

    class DeviceList(list):
        device = SimpleNamespace(type=device_type)

    class Inputs(dict):
        def to(self, device):
            calls["device"] = device
            return self

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert prompts == ["unchanged original prompt"]
            assert kwargs == {"return_tensors": "pt", "padding": True}
            return Inputs(input_ids=DeviceList([Tokens([1, 2])]))

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [3, 4] and skip_special_tokens is True
            return "original next line"

    def generate(**kwargs):
        calls["generation"] = {k: v for k, v in kwargs.items() if k != "input_ids"}
        return DeviceList([Tokens([1, 2, 3, 4])])

    if device_type != "npu":
        with pytest.raises(ValueError, match="no silent fallback"):
            RUN.generate_one(
                SimpleNamespace(inference_mode=nullcontext),
                SimpleNamespace(generate=generate),
                Tokenizer(),
                "unchanged original prompt",
                "npu:1",
            )
        assert "generation" not in calls
        return
    tokens, text = RUN.generate_one(
        SimpleNamespace(inference_mode=nullcontext),
        SimpleNamespace(generate=generate),
        Tokenizer(),
        "unchanged original prompt",
        "npu:1",
    )
    assert calls["device"] == "npu:1"
    assert calls["generation"] == {
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
    }
    assert tokens == [3, 4] and text == "original next line"


def test_execute_requires_explicit_resource_window_before_any_files():
    with pytest.raises(ValueError, match="idle research/NPU window"):
        RUN.execute(SimpleNamespace(execute=True, resource_window_confirmed=False))


def test_author_scorer_runs_in_fresh_fixed_hash_process(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("PYTHONHASHSEED", "123")

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="official report")

    monkeypatch.setattr(RUN.subprocess, "run", run)
    assert (
        RUN.score_original(tmp_path, tmp_path / "results", "python")
        == "official report"
    )
    command, options = calls[0]
    assert command[1] == str(tmp_path / "eval.py")
    assert command[-2:] == ["--language", "python"]
    assert options["env"]["PYTHONHASHSEED"] == "0"
