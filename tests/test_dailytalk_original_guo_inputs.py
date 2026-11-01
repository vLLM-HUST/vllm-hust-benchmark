"""Preparation fixtures are not native TTS or original-data execution evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "dailytalk_guo", ROOT / "scripts/prepare_dailytalk_original_guo_inputs.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def assert_separate_source_delivery(source):
    if not source["checked"]:
        assert source["delivery_receipt"] is None
        return
    final = json.loads((ROOT / source["delivery_receipt"]).read_text("utf-8"))
    assert final["source_checkbox_eligible"]
    assert final["native_evidence"].endswith(
        "dailytalk-original-fixed29-synthesis-audit-20261101.json"
    )
    assert final["original_wav_files_full_pcm_decoded"] == 29
    assert final["actual_ci_public_research_full_record_roundtrips"] == 29
    assert final["installed_provider_payloads_verified"] == 79
    assert final["general_complete_dialogue_preparation_available"]
    assert final["qualified_ci_conclusions"] == "success"


def test_gru_cpu_equation_evidence_does_not_promote_native_source():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dailytalk-gru-cpu-equations-20261101.json"
        ).read_text(encoding="utf-8")
    )
    code = (ROOT / receipt["entry"]).read_text(encoding="utf-8")
    assert hashlib.sha256(code.encode()).hexdigest() == receipt["entry_sha256"]
    assert receipt["numeric_cases"] == 18
    assert receipt["maximum_absolute_error"] < receipt["atol"]
    assert not receipt["npu_initialized"]
    assert (
        receipt["original_checkpoints_loaded"]
        == receipt["original_tasks_executed"]
        == 0
    )
    assert receipt["source_checkbox_eligible"] is False
    source = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )["delivery_inventory"]["DAILYTALK"]
    assert_separate_source_delivery(source)


def test_npu_equation_control_is_bound_and_not_original_synthesis():
    source = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )["delivery_inventory"]["DAILYTALK"]
    candidate = source["fp32_gru_compatibility_candidate"]
    receipt = json.loads(
        (ROOT / candidate["npu_equation_receipt"]).read_text(encoding="utf-8")
    )
    assert candidate["synthetic_npu_equivalence_verified"] is True
    assert_separate_source_delivery(source)
    code = (ROOT / receipt["entry"]).read_text(encoding="utf-8")
    assert hashlib.sha256(code.encode()).hexdigest() == receipt["entry_sha256"]
    assert (
        receipt["numeric_cases"] == 18
        and receipt["maximum_absolute_error"] < receipt["atol"]
    )
    assert receipt["dtype"] == "float32" and receipt["hf32"] is False
    assert (
        receipt["controller_exit_code"] == 0
        and receipt["protected_services_same_birth"]
    )
    assert receipt["lease_released_and_control_processes_exited"]
    assert (
        receipt["original_checkpoint_loads"] == receipt["original_tasks_executed"] == 0
    )
    assert receipt["source_checkbox_eligible"] is False


def test_actual_fixed29_synthesis_requires_separate_saved_native_evidence():
    source = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )["delivery_inventory"]["DAILYTALK"]
    assert source["fp32_gru_compatibility_candidate"]["original_synthesis_verified"]
    native = json.loads(
        (ROOT / source["current_native_synthesis_attempt"]["receipt"]).read_text(
            encoding="utf-8"
        )
    )
    assert native["original_utterances"] == native["wav_files_full_pcm_decoded"] == 29
    assert (
        native["wav_files_nonzero"]
        == native["png_files_signature_and_dimensions_verified"]
        == 29
    )
    assert native["acoustic_forward_batches"] == native["vocoder_forward_batches"] == 4
    assert native["official_model_files_rehashed"] == 2
    assert native["original_compatibility_source_files_reverified"] == 52
    assert native["original_guo_feature_files_reverified"] == 29
    assert native["native_exit_code"] == 0 and native["exclusive_lease_released"]
    assert native["audit_model_calls"] == 0 and not native["quality_score_claimed"]
    assert not native["source_checkbox_eligible"]
    assert_separate_source_delivery(source)
    plotting = json.loads(
        (ROOT / source["plotting_overlay_receipt"]).read_text(encoding="utf-8")
    )
    assert plotting["control"]["original_render_function_passed"]
    assert plotting["selected_matplotlib"] == "3.9.4"
    assert plotting["wheel_sha256"] in (ROOT / plotting["requirements"]).read_text(
        encoding="utf-8"
    )


def test_plan_never_reads_data_or_creates_output(tmp_path):
    report = runner.prepare(
        SimpleNamespace(
            metadata_root=tmp_path / "missing",
            audio_root=tmp_path / "missing-audio",
            output=tmp_path / "absent",
            action="plan",
        )
    )
    assert report["history_window_turns"] == 10
    assert report["history_encoder_embedding_dimension"] == 512
    assert report["model_calls"] == report["embedding_arrays_created"] == 0
    assert not report["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())


def test_rows_preserve_original_bytes_and_order():
    raw = b"0_1_d7|1|{HH}|hello|neutral\r\n0_0_d8|0|{AY}|other|neutral\n"
    rows = runner.select_rows(raw, {"0_1_d7"})
    assert rows[0][0] == raw.splitlines(keepends=True)[0]


@pytest.mark.parametrize(
    "raw",
    [
        b"0_1_d7|1|x|a|n\n0_1_d7|1|x|a|n\n",
        b"0_1_d7|1|missing-fields\n",
        b"0_0_d8|0|x|b|n\n",
    ],
)
def test_reject_duplicate_malformed_or_missing_original_rows(raw):
    with pytest.raises(ValueError):
        runner.select_rows(raw, {"0_1_d7"})


def test_history_is_original_last_ten_not_all_or_zero_history():
    raw = b"".join(f"{i}_0_d7|0|x|turn{i}|n\n".encode() for i in range(12))
    rows = runner.select_rows(raw, {f"{i}_0_d7" for i in range(12)})
    plan = runner.history_plan(rows)
    assert plan[0]["history_ids"] == []
    assert plan[1]["history_ids"] == ["0_0_d7"]
    assert plan[-1]["history_ids"] == [f"{i}_0_d7" for i in range(1, 11)]
    assert plan[-1]["feature_path"] == "text_emb/0-text_emb-11_0_d7.npy"


def test_missing_history_turn_rejected():
    rows = runner.select_rows(b"1_0_d7|0|x|a|n\n", {"1_0_d7"})
    with pytest.raises(ValueError, match="contiguous"):
        runner.history_plan(rows)


def test_fixture_materialization_verify_and_tamper(tmp_path, monkeypatch):
    import prepare_dailytalk_original_audio as audio
    import prepare_dailytalk_original_metadata as metadata

    source = tmp_path / "metadata"
    processed = source / "preprocessed_data/DailyTalk"
    processed.mkdir(parents=True)
    original = b"0_1_d7|1|{HH}|hello|neutral\r\n"
    (processed / "val_phone.txt").write_bytes(original)
    for name in runner.MAPS:
        (processed / name).write_bytes(b'{"original": 1}\n')
    monkeypatch.setattr(audio, "prepare", lambda args: None)
    monkeypatch.setattr(audio, "selected_ids", lambda *args: {"0_1_d7"})
    monkeypatch.setattr(metadata, "prepare", lambda args: None)
    args = SimpleNamespace(
        metadata_root=source,
        audio_root=tmp_path / "audio",
        output=tmp_path / "output",
        action="prepare",
    )
    report = runner.prepare(args)
    assert report["original_utterances"] == 1 and report["model_calls"] == 0
    assert (args.output / "val_phone.fixed3.private.txt").read_bytes() == original
    contract = json.loads((args.output / "guo-input-contract.private.json").read_text())
    assert not contract["embeddings_generated"] and not contract["phonemes_recomputed"]
    with pytest.raises(ValueError, match="fresh"):
        runner.prepare(args)
    args.action = "verify"
    runner.prepare(args)
    (args.output / "val_phone.fixed3.private.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        runner.prepare(args)


def test_output_cannot_overlap_original_inputs(tmp_path):
    with pytest.raises(ValueError, match="separate external"):
        runner.prepare(
            SimpleNamespace(
                metadata_root=tmp_path / "metadata",
                audio_root=tmp_path / "audio",
                output=tmp_path,
                action="prepare",
            )
        )
