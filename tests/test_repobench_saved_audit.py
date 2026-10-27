"""Saved-output audit fixtures only; no native publisher/model execution."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "repobench_saved", ROOT / "scripts/verify_repobench_original_saved.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize(
    "tamper", [None, "raw", "pred", "gt", "unknown", "missing", "duplicate"]
)
def test_saved_original_outputs_and_gt_cannot_be_changed(tmp_path, tamper):
    task = {
        "language": "python",
        "setting": "in_file",
        "filtered_index": 4,
        "original": {"level": "2k", "next_line": "ground truth"},
    }
    attempt = {
        "task_index": 0,
        "language": "python",
        "setting": "in_file",
        "filtered_index": 4,
        "status": "GENERATED_ORIGINAL_NEXT_LINE",
        "token_ids": [3],
        "raw_output": "generated line",
        "pred": "generated line",
    }
    row = {"idx": 4, "level": "2k", "pred": "generated line", "gt": "ground truth"}
    if tamper == "raw":
        attempt["raw_output"] = "changed raw answer"
    elif tamper == "pred":
        attempt["pred"] = "cleaned to pass"
    elif tamper == "gt":
        row["gt"] = "changed reference"
    elif tamper == "unknown":
        attempt["status"] = "STARTED_UNKNOWN_OUTCOME"
    if tamper != "missing":
        (tmp_path / "attempt-00000.private.json").write_text(json.dumps(attempt))
    output = tmp_path / "python"
    output.mkdir()
    (output / "in_file.jsonl").write_text(
        json.dumps(row)
        + "\n"
        + (json.dumps(row) + "\n" if tamper == "duplicate" else "")
    )
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "generated line")
    author = SimpleNamespace(get_first_line_not_comment=lambda raw, language: raw)

    def run():
        completed, unknown = AUDIT.audit_saved_tasks(
            [task], tmp_path, author, tokenizer
        )
        AUDIT.audit_result_files(completed, tmp_path)
        return completed, unknown

    if tamper:
        with pytest.raises(ValueError):
            run()
    else:
        completed, unknown = run()
        assert len(completed) == 1 and not unknown


def test_partial_unknown_attempt_is_retained_not_completed(tmp_path):
    task = {"language": "python", "setting": "in_file", "filtered_index": 0}
    attempt = {**task, "task_index": 0, "status": "STARTED_UNKNOWN_OUTCOME"}
    (tmp_path / "attempt-00000.private.json").write_text(json.dumps(attempt))
    completed, unknown = AUDIT.audit_saved_tasks(
        [task], tmp_path, None, None, partial=True
    )
    assert completed == [] and unknown == [0]


@pytest.mark.parametrize(
    "mode",
    [
        "legacy_frozen",
        "legacy_not_frozen",
        "controlled_changed",
        "nonrepeatable",
        "controlled_exact",
    ],
)
def test_fixed_seed_scoring_never_overwrites_legacy_reports(
    tmp_path, monkeypatch, mode
):
    root = tmp_path / "run"
    root.mkdir()
    selection = [{"language": "python"}]
    receipt = {
        "source_id": "REPOBENCH",
        "code_revision": AUDIT.native.REVISION,
        "model_revision": AUDIT.native.MODEL_REVISION,
        "prompt_budget": 15800,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
        "status": "ORIGINAL_GENERATION_AND_AUTHOR_SCORING_COMPLETED_PENDING_SAVED_AUDIT",
        "verified_parameter_device_type": "npu",
        "per_stratum": 1,
        "selected_tasks": 1,
        "model_calls": 1,
    }
    if mode.startswith("controlled"):
        receipt["scoring_python_hash_seed"] = 0
    (root / "receipt-public.json").write_text(json.dumps(receipt))
    (root / "selection.private.json").write_text(json.dumps(selection))
    saved = "fixed report" if mode == "controlled_exact" else "legacy report"
    original = root / "python-official-scoring.txt"
    original.write_text(saved)
    before = original.stat().st_mtime_ns
    monkeypatch.setattr(AUDIT.native, "verify_original", lambda *a: None)
    monkeypatch.setattr(AUDIT.native, "verify_files", lambda *a: None)
    monkeypatch.setattr(AUDIT.native, "prepare", lambda *a: selection)
    monkeypatch.setattr(AUDIT.native, "original_module", lambda *a: SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: None)
        ),
    )
    monkeypatch.setattr(
        AUDIT, "audit_saved_tasks", lambda *a, **kw: ([("fixture", "fixture")], [])
    )
    monkeypatch.setattr(AUDIT, "audit_result_files", lambda *a, **kw: None)
    counter = []

    def score(*a):
        counter.append(1)
        return (
            "different repeat"
            if mode == "nonrepeatable" and len(counter) % 2 == 0
            else "fixed report"
        )

    monkeypatch.setattr(AUDIT.native, "score_original", score)
    freeze = (
        None
        if mode in ["legacy_not_frozen", "controlled_changed", "controlled_exact"]
        else tmp_path / "fixed-scores"
    )
    args = SimpleNamespace(
        source_root=tmp_path / "source",
        archive=tmp_path / "source.zip",
        data_root=tmp_path / "data",
        model_root=tmp_path / "model",
        run_root=root,
        partial=False,
        freeze_scoring_output=freeze,
    )
    if mode in ["legacy_not_frozen", "controlled_changed", "nonrepeatable"]:
        with pytest.raises(ValueError):
            AUDIT.execute(args)
    else:
        result = AUDIT.execute(args)
        assert (
            result["scoring_python_hash_seed"] == 0 and result["new_model_calls"] == 0
        )
        if freeze:
            assert (
                freeze / "python-official-scoring.txt"
            ).read_text() == "fixed report"
            assert result["legacy_original_report_matches_fixed_seed"] == {
                "python": False
            }
    assert original.read_text() == saved and original.stat().st_mtime_ns == before
