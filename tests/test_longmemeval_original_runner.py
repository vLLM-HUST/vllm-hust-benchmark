"""CPU fixtures only; never a claim of native benchmark validation."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def rag_fixture():
    row = {
        "question_id": "fixture-only",
        "question": "original question",
        "answer": "reference-not-input",
        "haystack_sessions": [
            [{"role": "user", "content": "first"}],
            [{"role": "user", "content": "second"}],
        ],
        "haystack_session_ids": ["s0", "s1"],
        "haystack_dates": ["date0", "date1"],
    }
    record = deepcopy(row)
    record["retrieval_results"] = {
        "query": row["question"],
        "ranked_items": [
            {"corpus_id": "s1", "text": "second", "timestamp": "date1"},
            {"corpus_id": "s0", "text": "first", "timestamp": "date0"},
        ],
        "metrics": {},
    }
    return row, record


def fixture_indexer(session, granularity, identity, date):
    assert granularity == "session"
    return [session[0]["content"]], [identity], [date]


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


def test_rag_preserves_whole_original_corpus_and_publisher_ranking():
    row, record = rag_fixture()
    assert runner.retrieval_inputs([row], [0], [record], fixture_indexer) == {0: record}


def test_rag_retains_legitimate_repeated_original_session_ids_and_multiplicity():
    row, record = rag_fixture()
    row["haystack_session_ids"] = ["s0", "s0"]
    record["haystack_session_ids"] = ["s0", "s0"]
    record["retrieval_results"]["ranked_items"][0]["corpus_id"] = "s0"
    assert runner.retrieval_inputs([row], [0], [record], fixture_indexer) == {0: record}
    record["retrieval_results"]["ranked_items"][0] = deepcopy(
        record["retrieval_results"]["ranked_items"][1]
    )
    with pytest.raises(ValueError):
        runner.retrieval_inputs([row], [0], [record], fixture_indexer)


@pytest.mark.parametrize(
    "change",
    ["question", "history", "missing", "duplicate", "text", "timestamp", "query"],
)
def test_rag_rejects_altered_tasks_or_original_corpus(change):
    row, record = rag_fixture()
    ranked = record["retrieval_results"]["ranked_items"]
    if change == "question":
        record["question"] = "replacement"
    elif change == "history":
        record["haystack_sessions"] = []
    elif change == "missing":
        ranked.pop()
    elif change == "duplicate":
        ranked[1] = deepcopy(ranked[0])
    elif change == "text":
        ranked[0]["text"] = "invented"
    elif change == "timestamp":
        ranked[0]["timestamp"] = "invented"
    else:
        record["retrieval_results"]["query"] = "replacement"
    with pytest.raises(ValueError):
        runner.retrieval_inputs([row], [0], [record], fixture_indexer)


def test_rag_rejects_replacement_or_duplicate_cases():
    row, record = rag_fixture()
    for records in [[], [record, record], [dict(record, question_id="replacement")]]:
        with pytest.raises(ValueError):
            runner.retrieval_inputs([row], [0], records, fixture_indexer)


def test_rag_uses_original_flat_session_top10_con_contract():
    row, record = rag_fixture()

    def publisher(
        entry, mode, topk, useronly, form, cot, tokenizer, backend, limit, merge, *, con
    ):
        assert entry == record and mode == "flat-session" and topk == 10
        assert not useronly and form == "json" and cot and not con
        assert backend == "huggingface" and limit == 129272 and merge == "none"
        entry["haystack_sessions"].clear()
        return "publisher RAG prompt"

    assert runner.complete_prompt(publisher, record, object(), 131072, rag=True) == (
        "publisher RAG prompt",
        800,
    )
    assert record["haystack_sessions"] == row["haystack_sessions"]


def test_prepare_only_never_calls_generation(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace

    rows = [dict(rag_fixture()[0], question_id=str(i)) for i in range(500)]
    raw = json.dumps(rows).encode()
    (tmp_path / "original-fixture.json").write_bytes(raw)
    monkeypatch.setitem(
        runner.DATA_FILES,
        "fixture",
        ("original-fixture.json", hashlib.sha256(raw).hexdigest()),
    )
    monkeypatch.setattr(
        runner,
        "publisher_prompt",
        lambda source: lambda *a, **kw: "fixture original prompt",
    )
    calls = []

    def tokenize_only(url, body):
        assert url.endswith("/tokenize")
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        calls.append(body)
        return {"count": 10, "tokens": [1] * 10}

    monkeypatch.setattr(runner, "post", tokenize_only)
    result = runner.run(
        SimpleNamespace(
            api_base="http://localhost:19468/v1",
            source=tmp_path,
            data_root=tmp_path,
            variant="fixture",
            full=False,
            direct=False,
            output=tmp_path / "prepared",
            model="fixture-only",
            context_window=131072,
            prepare_only=True,
            disable_thinking=True,
        )
    )
    assert result["status"] == "ORIGINAL_PROMPTS_PREPARED_NO_GENERATION"
    assert result["completed_rows"] == result["model_calls"] == 0
    assert (
        len(calls)
        == len(list((tmp_path / "prepared").glob("*.prepared.private.json")))
        == 3
    )
    assert not (tmp_path / "prepared/hypotheses.private.jsonl").exists()


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


def test_original_bm25_source_plan_has_no_download_or_data(tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "longmemeval",
            "--author-root",
            str(tmp_path / "absent"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert (
        report["files"] == 13
        and report["bytes"] == 81752
        and report["model_calls"] == 0
    )
    assert not report["source_checkbox_eligible"] and not list(tmp_path.iterdir())


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
