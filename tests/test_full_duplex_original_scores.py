"""Score-entry orchestration fixtures, not benchmark/quality evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duplex_scores", ROOT / "scripts/run_full_duplex_original_scores.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.mark.parametrize(
    "phase", ["v10-nonjudge", "v10-backchannel-npu", "v15-timing-npu", "v15-summary"]
)
def test_offline_score_plan_does_not_import_model_runtime(tmp_path, phase):
    command = [sys.executable, str(SPEC.origin), "--phase", phase]
    for flag in ["omni-root", "reference-source", "tree", "output", "scores"]:
        command.extend(["--" + flag, str(tmp_path / flag)])
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["status"] == "PLAN_ONLY_NO_MODEL_CALLS"
    assert list(tmp_path.iterdir()) == []


def arguments():
    return argparse.Namespace(
        phase="v15-timing-npu",
        engine="fixture-engine",
        tree=Path("tree"),
        scores=Path("scores"),
        reference_source=Path("reference"),
        judge_config=None,
        api_key_env="FIXTURE_ENV",  # pragma: allowlist secret - environment variable name fixture
        base_url=None,
        served_model=None,
        local_judge_output=None,
    )


def test_v15_original_limits_and_summary_settings_are_not_weakened():
    native = ENTRY.native_arguments(arguments())
    assert native.limit is None and native.only == [] and native.retry_failed is False
    assert native.audio_loader == "soundfile"
    assert native.bootstrap == 2000 and native.seed == 20260925
    assert native.concurrency == 1 and native.timeout_s == 120.0


@pytest.mark.parametrize(
    "url",
    [
        None,
        "http://user:fixture@localhost/v1",  # pragma: allowlist secret - invalid URL fixture
        "http://localhost/v1?token=fixture",
    ],  # pragma: allowlist secret - rejection fixtures
)
def test_local_judge_requires_explicit_credential_free_endpoint(url):
    args = arguments()
    args.phase = "v10-interruption-localjudge"
    args.base_url = url
    args.served_model = "fixture/not-gpt"
    with pytest.raises(ValueError, match="credential-free"):
        ENTRY.validate(args)


def test_local_judge_cannot_overwrite_original_summary():
    args = arguments()
    args.phase = "v15-localjudge"
    args.base_url = "http://localhost/v1"
    args.judge_config = Path("judge-config.json")
    args.local_judge_output = args.scores
    with pytest.raises(ValueError, match="separate"):
        ENTRY.validate(args)


def test_nonjudge_calls_only_original_three_subsets(monkeypatch):
    args = arguments()
    args.phase = "v10-nonjudge"
    calls = []

    def evaluate(tree, paths, subsets, judge):
        calls.append((tree, paths, subsets, judge))
        return {"subsets": {name: {"result": {"fixture": 0}} for name in subsets}}

    modules = {
        "benchmarks": types.ModuleType("benchmarks"),
        "benchmarks.duplex": types.ModuleType("duplex"),
        "benchmarks.duplex.reference_source": types.SimpleNamespace(
            verify_reference=lambda *a: "verified"
        ),
        "benchmarks.duplex.reference_core": types.SimpleNamespace(
            V10_REFERENCE_FILES="original-files"
        ),
        "benchmarks.duplex.reference_v10": types.SimpleNamespace(
            evaluate=evaluate, JudgeLedger=None
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    result = ENTRY.score(args)
    assert set(result) == set(ENTRY.NONJUDGE_SUBSETS)
    assert calls == [(args.tree, "verified", list(ENTRY.NONJUDGE_SUBSETS), None)]


def test_original_system_exit_is_reported_as_failure(tmp_path, monkeypatch):
    command = ["score", "--phase", "v10-nonjudge", "--execute"]
    for flag in ["omni-root", "reference-source", "tree", "output"]:
        command.extend(["--" + flag, str(tmp_path / flag)])
    monkeypatch.setattr(sys, "argv", command)

    def original_refusal(args):
        raise SystemExit("already evaluated; reuse original results")

    monkeypatch.setattr(ENTRY, "score", original_refusal)
    with pytest.raises(SystemExit, match="already evaluated"):
        ENTRY.main()
    receipt = json.loads((tmp_path / "output/qualification-public.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["error_type"] == "SystemExit"


def test_vad_binding_passes_state_reset_and_requires_npu_output():
    spec = importlib.util.spec_from_file_location(
        "duplex_vad", ROOT / "scripts/full_duplex_npu_vad.py"
    )
    vad = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vad)
    binding = vad.NpuSilero.__new__(vad.NpuSilero)
    calls = []
    binding.model = argparse.Namespace(reset_states=lambda: calls.append("reset"))
    binding.reset_states()
    assert calls == ["reset"]
    binding.model = lambda wave, rate: argparse.Namespace(device="cpu")
    wave = argparse.Namespace(to=lambda device: calls.append(device))
    with pytest.raises(ValueError, match="remain on NPU"):
        binding(wave, 16000)
    assert calls[-1] == "npu:0"
