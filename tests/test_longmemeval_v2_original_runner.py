"""Software fixtures only: no original tasks, native calls or judge scoring."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "lme_v2_native_runner", ROOT / "scripts/run_longmemeval_v2_original.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original(index, domain="enterprise", scorer="norm_phrase_set_match"):
    return {
        "id": f"fixture-{index}",
        "domain": domain,
        "question": "fixture question",
        "answer": "fixture reference",
        "image": None,
        "eval_function": scorer,
    }


def test_fixed_original_selection_never_filters_judge_questions(runner):
    rows = [
        original(0),
        original(1, "web"),
        original(2),
        original(3, "web", "llm_abstention_checker"),
        original(4, "web", "llm_gotchas_checker"),
        original(5, "web"),
    ]
    snapshot = deepcopy(rows)
    selected, judges = runner.select_original(rows, "web")
    assert [index for index, _ in selected] == [1, 3, 4]
    assert judges == ["llm_abstention_checker", "llm_gotchas_checker"]
    assert rows == snapshot


def test_unknown_scoring_function_and_small_samples_are_rejected(runner):
    with pytest.raises(ValueError, match="at least three"):
        runner.select_original([original(i) for i in range(3)], "enterprise", count=1)
    with pytest.raises(ValueError, match="unknown publisher scorer"):
        runner.select_original(
            [original(i, scorer="not_a_scorer") for i in range(3)], "enterprise"
        )


def test_extra_original_range_never_repeats_prior_first_three(runner):
    selected, judges = runner.select_original(
        [original(i) for i in range(7)], "enterprise", start=3
    )
    assert [index for index, _ in selected] == [3, 4, 5] and not judges
    with pytest.raises(ValueError, match="full cannot use offset"):
        runner.select_original(
            [original(i) for i in range(7)], "enterprise", full=True, start=3
        )


def test_local_execution_does_not_inherit_cloud_key_or_export_agent_traces(
    runner, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "software-fixture-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://fixture.invalid")
    monkeypatch.setenv("PYTHONPATH", "/fixture/inherited-cann")
    environment = runner.execution_environment(tmp_path)
    assert "OPENAI_API_KEY" not in environment and "OPENAI_BASE_URL" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["OPENAI_AGENTS_DISABLE_TRACING"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert (
        runner.execution_environment(tmp_path, judge_env="OPENAI_API_KEY")[
            "OPENAI_API_KEY"
        ]
        == "software-fixture-key"
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/v1",
        "http://remote.invalid/v1",
        "http://user:password@localhost/v1",  # pragma: allowlist secret - invalid basic-auth fixture
        "http://localhost/v1?key=fixture",
    ],
)
def test_only_credential_free_local_model_endpoints(runner, endpoint):
    with pytest.raises(ValueError, match="loopback"):
        runner.local_endpoint(endpoint)


def test_judge_block_occurs_before_network_or_model_execution(
    runner, tmp_path, monkeypatch
):
    author, data, output = (tmp_path / name for name in ["author", "data", "out"])
    author.mkdir()
    data.mkdir()
    rows = [original(i, scorer="llm_abstention_checker") for i in range(3)]
    (data / "questions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "verify_source", lambda *_: author)
    monkeypatch.setattr(
        runner,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("no network call allowed"),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("no model call allowed"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--author-root",
            str(author),
            "--data-root",
            str(data),
            "--output",
            str(output),
            "--domain",
            "enterprise",
            "--execute",
        ],
    )
    with pytest.raises(ValueError, match="external judge"):
        runner.main()
    assert not output.exists()


def saved_fixture(tmp_path):
    native = tmp_path / "native"
    (native / "runtime_inputs").mkdir(parents=True)
    rows = [original(i) for i in range(3)]
    selected = list(enumerate(rows))
    haystack = {
        row["id"]: ["fixture-trajectory-a", "fixture-trajectory-b"] for row in rows
    }
    runtime = [
        {key: value for key, value in row.items() if key != "image"} for row in rows
    ]
    (native / "runtime_inputs/questions.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    (native / "runtime_inputs/haystack.json").write_text(
        json.dumps(haystack), encoding="utf-8"
    )
    records, prompts = [], []
    for row in rows:
        # Author log view uses paths; its API view contains inline image bytes.
        log = [
            {
                "role": "user",
                "content": [{"type": "image_path", "image_path": "fixture.png"}],
            }
        ]
        api = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,Zml4dHVyZQ=="},
                    }
                ],
            }
        ]
        records.append(
            {
                "question_id": row["id"],
                "question_text": row["question"],
                "answer_gold": row["answer"],
                "eval_function": row["eval_function"],
                "haystack_ids": haystack[row["id"]],
                "response_raw": "wrong fixture answer",
                "response_parsed_boxed": "wrong fixture answer",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "score_bool": False,
                "score": 0.0,
                "prompt_messages": log,
            }
        )
        prompts.append(
            {"question_id": row["id"], "messages": api, "prompt_messages": log}
        )
    (native / "prompt_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in prompts), encoding="utf-8"
    )
    (native / "aggregated_metrics.json").write_text(
        json.dumps({"overall": {"count_all_questions": 3}}), encoding="utf-8"
    )
    return native, selected, haystack, records


def store_records(native, records):
    (native / "per_question.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )


def fixture_metrics():
    return SimpleNamespace(
        extract_boxed_answer=lambda value: value,
        is_unknown=lambda value: value == "UNKNOWN",
        score_to_bool=bool,
        eval_from_spec=lambda _spec, response, answer: response == answer,
    )


def test_saved_original_judge_result_is_not_recalled_or_charged(runner, tmp_path):
    native, selected, haystack, records = saved_fixture(tmp_path)
    runtime = json.loads((native / "runtime_inputs/questions.json").read_text("utf-8"))
    for (_, original), record, item in zip(selected, records, runtime):
        original["eval_function"] = record["eval_function"] = item["eval_function"] = (
            "llm_gotchas_checker"
        )
    (native / "runtime_inputs/questions.json").write_text(
        json.dumps(runtime), encoding="utf-8"
    )
    store_records(native, records)
    metrics = fixture_metrics()
    metrics.eval_from_spec = lambda *_args, **_kwargs: pytest.fail(
        "saved stochastic judgments must not be re-called"
    )
    with pytest.raises(ValueError, match="not authorized"):
        runner.verify_results(native, selected, haystack, metrics, tmp_path)
    assert (
        runner.verify_results(
            native,
            selected,
            haystack,
            metrics,
            tmp_path,
            accept_saved_judge_scores=True,
        )
        == 3
    )


def test_paid_judge_opt_in_without_key_never_starts_network(
    runner, tmp_path, monkeypatch
):
    author, data, output = (tmp_path / name for name in ["author", "data", "out"])
    author.mkdir()
    data.mkdir()
    (data / "questions.jsonl").write_text(
        "\n".join(
            json.dumps(original(i, scorer="llm_abstention_checker")) for i in range(3)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "verify_source", lambda *_: author)
    monkeypatch.setattr(
        runner,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("no network without key"),
    )
    monkeypatch.delenv("FIXTURE_MISSING_JUDGE_KEY", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--author-root",
            str(author),
            "--data-root",
            str(data),
            "--reader-assets",
            str(tmp_path / "model"),
            "--output",
            str(output),
            "--domain",
            "enterprise",
            "--execute",
            "--allow-paid-judge",
            "--judge-api-key-env",
            "FIXTURE_MISSING_JUDGE_KEY",
        ],
    )
    with pytest.raises(ValueError, match="credential environment"):
        runner.main()
    assert not output.exists()


def test_wrong_answers_retained_and_author_log_api_views_not_conflated(
    runner, tmp_path
):
    native, selected, haystack, records = saved_fixture(tmp_path)
    store_records(native, records)
    assert (
        runner.verify_results(native, selected, haystack, fixture_metrics(), tmp_path)
        == 3
    )


@pytest.mark.parametrize(
    "failure",
    ["empty400", "reference", "haystack_order", "score", "budget", "coverage"],
)
def test_author_exit0_is_not_enough_for_native_completion(runner, tmp_path, failure):
    native, selected, haystack, records = saved_fixture(tmp_path)
    if failure == "empty400":
        records[0]["response_raw"] = ""
        records[0]["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    elif failure == "reference":
        records[0]["answer_gold"] = "substituted"
    elif failure == "haystack_order":
        records[0]["haystack_ids"] = list(reversed(records[0]["haystack_ids"]))
    elif failure == "score":
        records[0]["score_bool"] = True
    elif failure == "budget":
        records[0]["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 20001,
            "total_tokens": 20011,
        }
    else:
        records.pop()
    store_records(native, records)
    with pytest.raises(ValueError):
        runner.verify_results(native, selected, haystack, fixture_metrics(), tmp_path)


def test_processor_cache_binds_fixed_revision_without_reading_weights(runner, tmp_path):
    assets = tmp_path / "model"
    assets.mkdir()
    raw = b"metadata fixture"
    (assets / "config.json").write_bytes(raw)
    pins = {
        "model": {
            "repo_id": "fixture/model",
            "revision": "fixed-fixture",
            "files": [
                {
                    "path": "config.json",
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                {
                    "path": "missing.safetensors",
                    "bytes": 100,
                    "sha256": "not-used-for-processor",
                },
            ],
        }
    }
    cache = tmp_path / "cache"
    try:
        assert runner.prepare_processor_cache(assets, cache, pins) == 1
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip(
                "Windows lacks symlink privilege; Linux CI/server verify this cache operation"
            )
        raise
    hub = cache / "hub/models--fixture--model"
    assert (hub / "refs/main").read_text("utf-8") == "fixed-fixture"
    assert (hub / "snapshots/fixed-fixture").resolve() == assets.resolve()
