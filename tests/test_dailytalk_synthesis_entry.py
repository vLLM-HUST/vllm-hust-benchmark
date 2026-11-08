"""Public execution guards and saved-result routing without neural execution."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "dailytalk_synthesis", ROOT / "scripts/run_dailytalk_synthesis_npu.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_default_plan_loads_no_inputs():
    result = ENTRY.run(SimpleNamespace(action="plan"))
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu", "npu:-1"])
def test_only_explicit_npu_execute(device):
    with pytest.raises(ValueError, match="explicit NPU"):
        ENTRY.run(SimpleNamespace(action="execute", device=device))


@pytest.mark.parametrize(
    "confirmed,lease,budget",
    [(False, None, 6), (True, None, 6), (True, "lease", 0), (True, "lease", 7)],
)
def test_idle_window_lease_and_memory_guard_precede_input_loading(
    confirmed, lease, budget
):
    with pytest.raises(ValueError, match="idle resource"):
        ENTRY.run(
            SimpleNamespace(
                action="execute",
                device="npu:0",
                resource_window_confirmed=confirmed,
                lease_file=lease,
                allocator_gib=budget,
            )
        )


def test_verify_reuses_separate_saved_native_stage(monkeypatch, tmp_path):
    prepared = tmp_path / "public-prepared-stage"
    saved = tmp_path / "original-valid-stage"
    seen = []

    def preparation(args):
        seen.append(("prepare", args.action, args.stage))

    def audit(args):
        seen.append(("audit", args.action, args.stage))
        return {"audit_model_calls": 0, "source_checkbox_eligible": False}

    monkeypatch.setattr(ENTRY, "prepare", preparation)
    monkeypatch.setattr(ENTRY, "audit", audit)
    result = ENTRY.run(
        SimpleNamespace(
            action="verify", device="npu:0", stage=prepared, result_stage=saved
        )
    )
    assert seen == [("prepare", "verify", prepared), ("audit", "verify", saved)]
    assert result["audit_model_calls"] == 0


def test_completed_results_are_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(ENTRY, "prepare", lambda _: None)
    (tmp_path / "native-execution").mkdir()
    with pytest.raises(FileExistsError, match="retain previous"):
        ENTRY.run(
            SimpleNamespace(
                action="execute",
                device="npu:0",
                stage=tmp_path,
                result_stage=None,
                resource_window_confirmed=True,
                lease_file=tmp_path / "lease",
                allocator_gib=6,
            )
        )
    assert not (tmp_path / "lease").exists()


def test_real_server_public_entry_receipt_pins_current_code():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dailytalk-public-workspace-entry-20261101.json"
        ).read_text("utf-8")
    )
    for name, sha in (
        ("entry", "entry_sha256"),
        ("native_entry", "native_entry_sha256"),
    ):
        raw = (ROOT / receipt[name]).read_text("utf-8").encode()
        assert hashlib.sha256(raw).hexdigest() == receipt[sha]
    assert receipt["exit_codes"] == [0, 0, 0]
    assert receipt["native_entry_exit_codes"] == [0, 0]
    assert (
        receipt["original_utterances"]
        == receipt["saved_original_wav_files_full_pcm_decoded"]
        == 29
    )
    assert receipt["software_files"] == 52
    assert receipt["original_vocoder_symlink_verified"]
    assert not receipt["torch_imported"]
    assert not receipt["public_execute_path_reexecuted"]
    assert not receipt["source_checkbox_eligible"]
