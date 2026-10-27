"""Software fixtures and frozen evidence contracts; no downloads or model calls."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "ruler_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_ruler_original_profile.py",
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def generated_rows(tmp_path):
    rows = [
        {
            "index": 19,
            "input": f"fixture-{i}",
            "answer_prefix": " Answer:",
            "outputs": ["gold"],
            "others": {"id": i},
            "length": 30,
        }
        for i in range(3)
    ]
    path = tmp_path / "generated.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), rows


def test_duplicate_original_indices_retained_in_physical_order(tmp_path):
    path, sha, rows = generated_rows(tmp_path)
    actual, offsets = runner.read_selection(path, sha, None)
    assert actual == rows
    assert offsets == [0, 1, 2]
    assert runner.read_selection(path, sha, [2, 0])[1] == [2, 0]


@pytest.mark.parametrize("offsets", [[], [0, 0], [-1], [3], [True]])
def test_invalid_selection_rejected(tmp_path, offsets):
    path, sha, _ = generated_rows(tmp_path)
    with pytest.raises(ValueError):
        runner.read_selection(path, sha, offsets)


def test_digest_failure_before_reading(tmp_path):
    path, _, _ = generated_rows(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.read_selection(path, "wrong", None)


def test_suffix_only_does_not_leak_prompt_reference():
    result = {
        "choices": [{"text": "", "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 8, "completion_tokens": 0},
    }
    assert runner.suffix_response(result, 8, 128) == {"text": [""]}
    result["choices"][0].update(text="actual suffix", finish_reason="length")
    result["usage"]["completion_tokens"] = 128
    assert runner.suffix_response(result, 8, 128) == {"text": ["actual suffix"]}
    with pytest.raises(ValueError):
        runner.suffix_response(result, 9, 128)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://localhost/v1?k=x",
        "http://user@localhost/v1",
        "http://localhost/",
    ],
)
def test_no_implicit_external_prompt_upload(url):
    with pytest.raises(ValueError):
        runner.endpoint(url)


def test_changed_budget_rejected():
    body = {
        "prompt": "fixture",
        "max_tokens": 128,
        "temperature": 0.0,
        "top_k": 32,
        "top_p": 1.0,
        "stop": [],
    }
    assert (
        runner.completion_request(body, "fixture", 128, "model")["prompt"] == "fixture"
    )
    body["max_tokens"] = 32
    with pytest.raises(ValueError):
        runner.completion_request(body, "fixture", 128, "model")


class FixtureClient:
    def __init__(self, **kwargs):
        self.budget = kwargs["tokens_to_generate"]

    def __call__(self, prompt):
        return self._send_request(
            {
                "prompt": prompt,
                "max_tokens": self.budget,
                "temperature": 0.0,
                "top_k": 32,
                "top_p": 1.0,
                "stop": [],
            }
        )


def setup_run(tmp_path, monkeypatch, *, capacity=200):
    path, sha, _ = generated_rows(tmp_path)
    args = argparse.Namespace(
        source=tmp_path,
        data=path,
        data_sha256=sha,
        task="fixture",
        rows=None,
        output=tmp_path / "private",
        api_base="http://127.0.0.1:1234/v1",
        model="fixture-model",
        context_window=capacity,
    )
    calls = []

    def post(url, body):
        calls.append((url, body))
        if url.endswith("/tokenize"):
            return {"count": 20}
        return {
            "choices": [{"text": "wrong", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 128},
        }

    def metric(preds, refs):
        assert preds == ["wrong"] * 3 and refs == [["gold"]] * 3
        return 0.0

    monkeypatch.setattr(runner, "post", post)
    monkeypatch.setattr(
        runner,
        "original_components",
        lambda *a: (FixtureClient, 128, lambda s, c: s, metric),
    )
    return args, calls


def test_full_flow_retains_wrong_outputs_and_duplicate_indices(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["completed_rows"] == 3 and receipt["original_metric_score"] == 0.0
    assert receipt["status"] == "ORIGINAL_PROFILE_COMPLETED_REQUIRES_AUDIT"
    assert not receipt["source_checkbox_eligible"]
    preds = [
        json.loads(line)
        for line in (args.output / "predictions.private.jsonl").read_text().splitlines()
    ]
    assert [p["index"] for p in preds] == [19] * 3
    assert [b["prompt"] for u, b in calls if u.endswith("/completions")] == [
        f"fixture-{i} Answer:" for i in range(3)
    ]
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert len(calls) == 6


def test_capacity_failure_does_not_truncate_or_attempt_inference(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, capacity=100)
    with pytest.raises(ValueError, match="no truncation"):
        runner.run(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "EXECUTION_ERROR" and receipt["completed_rows"] == 0
    assert len(calls) == 1 and calls[0][0].endswith("/tokenize")


def test_source_check_precedes_import(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.original_components(tmp_path, "niah_single_1")


def test_cpu_receipt_matches_entry_without_upgrading_native_completion():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (
            root / "docs/workload-data/validation/ruler-public-entry-cpu-20261025.json"
        ).read_text()
    )
    assert (
        receipt["public_entry_sha256"]
        == hashlib.sha256(
            (root / "scripts/run_ruler_original_profile.py").read_bytes()
        ).hexdigest()
    )
    assert receipt["model_calls"] == 0 and not receipt["source_checkbox_eligible"]
    replay = receipt["saved_native_transcript_replay"]
    assert replay["original_rows_checked"] == 75 and replay["new_model_calls"] == 0
    assert not replay["full500_native_profile_complete"]
    assert not replay["new_public_entry_live_native_run_claim"]


def test_original_full500_evidence_is_single_profile_and_not_a_fresh_rerun():
    root = Path(__file__).resolve().parents[1] / "docs/workload-data/validation"
    receipt = json.loads(
        (root / "ruler-niah4k-full500-native-audit-20261025.json").read_text()
    )
    assert (
        receipt["original_rows_selected"]
        == receipt["original_rows_completed_and_audited"]
        == receipt["original_generation_attempts"]
        == 500
    )
    assert (
        receipt["original_output_budget"] == 128
        and sum(receipt["finish_reasons"].values()) == 500
    )
    assert receipt["public_entry_saved_native_transcript_replay_rows"] == 500
    assert receipt["new_model_calls_during_audit"] == 0
    assert not receipt["new_live_public_entry_execution_claim"]
    assert (
        not receipt["source_checkbox_eligible"]
        and receipt["not_all13_tasks_or_other_lengths"]
    )


def test_five_live_public_profiles_preserve_full_long_inputs_and_original_budgets():
    root = Path(__file__).resolve().parents[1] / "docs/workload-data/validation"
    receipt = json.loads(
        (root / "ruler-public-first5-native-audit-20261025.json").read_text()
    )
    assert len(receipt["profiles"]) == receipt["completed_profile_count"] == 5
    assert receipt["completed_original_rows_total"] == 15
    for profile in receipt["profiles"]:
        assert (
            profile["original_row_offsets"] == [0, 1, 2]
            and profile["completed_original_rows"] == 3
        )
        assert profile["live_public_entry_execution_verified"]
        assert sum(profile["finish_reasons"].values()) == 3
        assert all(
            count + profile["original_output_budget"] <= 131072
            for count in profile["actual_input_tokens"]
        )
        assert profile["new_model_calls_during_audit"] == 0
    assert any(p["generator_max_seq_length"] == 131072 for p in receipt["profiles"])
    assert (
        not receipt["source_checkbox_eligible"]
        and not receipt["all13_tasks_or_complete_length_grid_claim"]
    )


def test_eight_profile_consumer_checks_compare_raw_data_without_claiming_native_completion():
    root = Path(__file__).resolve().parents[1] / "docs/workload-data/validation"
    receipt = json.loads(
        (root / "ruler-eight-profile-binding-chain-20261025.json").read_text()
    )
    assert len(receipt["profiles"]) == 8
    assert (
        receipt["bound_and_indexed_rows_total"]
        == receipt["raw_original_projection_verified_rows_total"]
        == 4000
    )
    assert receipt[
        "raw_original_inputs_references_indices_and_generation_metadata_equal"
    ]
    assert receipt["public_roundtrip_rows_total"] == 48
    assert (
        receipt["research_index_import_rows_total"]
        == receipt["research_direct_prepare_resolve_rows_total"]
        == 24
    )
    assert receipt["binding_cli_is_candidate_not_final_ci_claim"]
    assert (
        receipt["model_calls"] == 0 and not receipt["native_execution_or_scoring_claim"]
    )
    assert (
        not receipt["research_default_dependency_lock_updated"]
        and not receipt["source_checkbox_eligible"]
    )
