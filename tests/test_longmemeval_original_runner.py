"""CPU fixtures only; never a claim of native benchmark validation."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "longmem_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_longmemeval_original.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1",
        "http://user:secret@localhost/v1",  # pragma: allowlist secret - deliberately fake credential fixture
        "http://localhost/v1?key=value",
    ],
)
def test_external_or_credential_endpoint_is_rejected(url):
    with pytest.raises(ValueError):
        runner.endpoint(url)


def test_complete_history_does_not_silently_keep_latest_fifty():
    row = {"haystack_sessions": [[{"has_answer": True}] for _ in range(70)]}
    original = deepcopy(row)

    def publisher(
        entry,
        mode,
        topk,
        useronly,
        history_format,
        cot,
        tokenizer,
        backend,
        limit,
        merge,
        *,
        con,
    ):
        assert mode == "orig-session" and topk == 70
        assert not useronly and history_format == "json" and cot and not con
        assert (
            backend == "huggingface"
            and limit == 131072 - 800 - 1000
            and merge == "none"
        )
        entry["haystack_sessions"][0][0].pop("has_answer")
        return "publisher prompt"

    assert runner.complete_prompt(publisher, row, object(), 131072) == (
        "publisher prompt",
        800,
    )
    assert row == original


def test_truncation_is_blocked_before_tokenizer_or_model_calls(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("no transport allowed")

    monkeypatch.setattr(runner, "post", unexpected)
    tokenizer = runner.NoTruncationTokenizer("http://localhost:8000/v1", "fixture")
    with pytest.raises(ValueError, match="truncation prohibited"):
        tokenizer(
            "original history", max_length=10, truncation=True, return_tensors="pt"
        )


def test_fixed_offsets_cannot_replace_out_of_range_rows():
    assert runner.selection([{}, {}, {}], [0, 1, 2]) == [0, 1, 2]
    for indices in [[], [0, 0], [-1], [3], [True]]:
        with pytest.raises(ValueError):
            runner.selection([{}, {}, {}], indices)


def test_unknown_publisher_code_never_executes(tmp_path):
    path = tmp_path / "src/generation/run_generation.py"
    path.parent.mkdir(parents=True)
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="source identity"):
        runner.publisher_prompt(tmp_path)


def test_actual_generation_checkpoint_does_not_claim_judge_or_source_completion():
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/workload-data/validation/longmemeval-original-native-checkpoint-20261025.json"
    )
    receipt = json.loads(path.read_text("utf-8"))
    assert receipt["native_generations_completed_and_audited"] == 6
    assert {p["variant"] for p in receipt["native_profiles"]} == {"oracle", "s"}
    for profile in receipt["native_profiles"]:
        assert profile["completed_and_audited_rows"] == 3
        assert profile["selected_physical_offsets"] == [0, 1, 2]
        assert profile["generation_budget"] == 800
        assert profile["original_generator_sha256"] == runner.GENERATOR_SHA256
        assert profile["all_original_sessions_preserved"]
        assert not profile["official_judge_executed"]
        assert profile["new_generation_calls_during_audit"] == 0
    assert receipt["offline_judge_client_environment"]["pip_check_returncode"] == 0
    assert not receipt["source_checkbox_eligible"]
