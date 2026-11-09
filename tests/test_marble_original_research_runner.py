"""Offline native-runner fixtures, never original tasks or model runs."""

import hashlib
import json
from pathlib import Path
import sys
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bind_marble_research_local as binder  # noqa: E402
import run_marble_original_research as entry  # noqa: E402


@pytest.mark.parametrize("returned_rows,exit_code", [([2], 0), ([], 1), ([0], 1)])
def test_cli_recovery_selects_only_original_row_two(
    tmp_path, monkeypatch, returned_rows, exit_code
):
    prepared = {"fixed_original_rows": [0, 1, 2], "source_checkbox_eligible": False}
    monkeypatch.setattr(entry, "prepare", lambda *_: prepared)

    def execute(output, report):
        assert output == tmp_path
        assert report["fixed_original_rows"] == [2]
        assert report["full_fixed_original_rows"] == [0, 1, 2]
        assert report["partial_recovery_only"] is True
        assert report["source_checkbox_eligible"] is False
        return dict(
            report,
            completed=[
                {"original_row": row, "cli_exit_code": 0, "result_record_present": True}
                for row in returned_rows
            ],
            model_http={"requests": 1, "pending": 0, "non200_responses": 0},
        )

    monkeypatch.setattr(entry, "execute", execute)
    argv = [
        arg
        for name in ("author-root", "prepared-root", "bound-root", "output")
        for arg in ("--" + name, str(tmp_path))
    ]
    assert entry.main([*argv, "--execute", "--original-row", "2"]) == exit_code


def fixture_prefix(tmp_path, count=3):
    source = tmp_path / "original-prefix"
    source.mkdir()
    rows = []
    for index in range(count):
        config = {
            "task": {"content": f"synthetic fixture {index}", "reference": "retain"},
            "agents": [{"agent_id": "fixture"}],
            "relationships": [],
            "coordinate_mode": "graph",
            "environment": {"type": "Research", "max_iterations": 5},
            "memory": {"type": "BaseMemory"},
            "llm": "gpt-3.5-turbo",
            "metrics": {"evaluate_llm": "gpt-4o"},
            "output": {"file_path": "fixture"},
            "unknown_extension": ["keep"],
        }
        raw = json.dumps(config).encode()
        filename = f"original-row-{index:05d}.yaml"
        (source / filename).write_bytes(raw)
        rows.append(
            {
                "configuration": filename,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_row_index": index,
                "source_file": "multiagentbench/research/research_main.jsonl",
            }
        )
    (source / "preparation-public.json").write_text(
        json.dumps(
            {
                "source_id": "MULTIAGENTBENCH",
                "status": "OFFICIAL_RESEARCH_RUNTIME_PROFILE_PREPARED_NOT_EXECUTED",
                "runtime_profile": "official-research",
                "selected_configurations": count,
                "configurations": rows,
            }
        )
    )
    bound = tmp_path / "bound"
    binder.prepare(
        source,
        bound,
        model="openai/local-actor",
        judge="openai/local-judge",
        execute=True,
    )
    return source, bound


def test_default_plan_does_not_read_originals(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entry, "prepare", lambda *a: pytest.fail("source read"))
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    assert entry.main(args) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
    assert not list(tmp_path.iterdir())


def test_isolated_cli_plan_imports_only_software(tmp_path):
    script = Path(entry.__file__)
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    result = subprocess.run(
        [sys.executable, "-I", str(script), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY"
    assert not list(tmp_path.iterdir())


def test_invalid_live_endpoint_rejected_before_preparing_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "prepare", lambda *a: pytest.fail("asset preparation"))
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    with pytest.raises(ValueError, match="loopback"):
        entry.main(args + ["--execute", "--model-api-base", "http://example.com:1/v1"])
    assert not list(tmp_path.iterdir())


def test_response_reuse_requires_explicit_live_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "prepare", lambda *a: pytest.fail("asset preparation"))
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    with pytest.raises(SystemExit) as error:
        entry.main(args + ["--reuse-model-results", str(tmp_path / "saved")])
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_complete_original_prefix_and_unknown_fields_kept(tmp_path):
    source, bound = fixture_prefix(tmp_path)
    configs, actor, judge = entry.original_prefix(source, bound)
    assert (
        len(configs) == 3
        and actor == "openai/local-actor"
        and judge == "openai/local-judge"
    )
    assert configs[0]["unknown_extension"] == ["keep"]


def test_changed_bound_task_rejected_even_if_receipt_rehashed(tmp_path):
    source, bound = fixture_prefix(tmp_path)
    path = bound / "original-row-00000.yaml"
    config = json.loads(path.read_text())
    config["task"]["content"] = "changed task"
    path.write_text(json.dumps(config))
    p = bound / "binding-public.json"
    report = json.loads(p.read_text())
    report["configurations"][0]["bound_configuration_sha256"] = entry.digest(path)
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="task changed"):
        entry.original_prefix(source, bound)


def test_dropped_original_case_rejected(tmp_path):
    source, bound = fixture_prefix(tmp_path)
    p = bound / "binding-public.json"
    report = json.loads(p.read_text())
    report["configurations"].pop()
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original Research prefix"):
        entry.original_prefix(source, bound)


def test_full_original_research_hundred_cases_not_capped_to_validation_three(tmp_path):
    source, bound = fixture_prefix(tmp_path, count=100)
    configs, actor, judge = entry.original_prefix(source, bound)
    assert len(configs) == 100
    assert configs[-1]["task"]["content"] == "synthetic fixture 99"
    assert actor == "openai/local-actor" and judge == "openai/local-judge"


def test_prepare_stages_only_approved_compat_copy_no_model(tmp_path, monkeypatch):
    source, bound = fixture_prefix(tmp_path)
    author = tmp_path / "author"
    (author / "marble/evaluator").mkdir(parents=True)
    original = author / "marble/evaluator/evaluator.py"
    original.write_bytes(b"synthetic original software")
    profile = tmp_path / "source-profile.json"
    profile.write_text(
        json.dumps({"files": [{"path": "marble/evaluator/evaluator.py"}]})
    )
    monkeypatch.setattr(entry, "SOURCE_PROFILE", profile)
    monkeypatch.setattr(entry, "verify_source", lambda *a: None)
    monkeypatch.setattr(
        entry,
        "compatibility_copy",
        lambda raw: (b"synthetic approved compat", [324, 325, 326]),
    )
    output = tmp_path / "native-output"
    try:
        result = entry.prepare(author, source, bound, output)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("host cannot create runtime resource symlink")
        raise
    assert result["completed"] == [] and result["model_requests"] == 0
    assert original.read_bytes() == b"synthetic original software"
    assert (
        output / "author/marble/evaluator/evaluator.py"
    ).read_bytes() == b"synthetic approved compat"
    for index in range(3):
        config = json.loads(
            (output / f"original-row-{index:05d}/runtime.private.yaml").read_text()
        )
        assert config["task"]["content"] == f"synthetic fixture {index}"
        assert config["environment"]["max_iterations"] == 5


def test_task_failure_cannot_be_reported_as_cli_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(entry, "prepare", lambda *a: {})
    monkeypatch.setattr(
        entry,
        "execute",
        lambda *a: {"completed": [{"cli_exit_code": 1, "result_file_exists": False}]},
    )
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    assert entry.main(args + ["--execute"]) == 1


def http_fixture(root, folder, status):
    queue = root / folder
    queue.mkdir(parents=True)
    (queue / ("a" * 32 + ".request.json")).write_text("{}")
    (queue / ("a" * 32 + ".response.json")).write_text(
        json.dumps({"http_status": status, "response": {}})
    )
    return queue


def test_native_http400_is_retained_not_mistaken_for_model_success(tmp_path):
    http_fixture(tmp_path, "original-row-00000/live-model-queue", 200)
    http_fixture(tmp_path, "original-row-00001/live-model-queue", 400)
    assert entry.saved_model_http_summary(tmp_path) == {
        "requests": 2,
        "responses": 2,
        "pending": 0,
        "http_status_counts": {"200": 1, "400": 1},
        "non200_responses": 1,
        "quality_scored_by_this_summary": False,
    }


def test_missing_response_remains_pending_not_a_request_to_replay(tmp_path):
    queue = tmp_path / "model-queue"
    queue.mkdir()
    (queue / ("a" * 32 + ".request.json")).write_text("{}")
    result = entry.saved_model_http_summary(tmp_path)
    assert result["requests"] == result["pending"] == 1
    assert result["responses"] == result["non200_responses"] == 0


@pytest.mark.parametrize("status", [None, True, "200", 600])
def test_invalid_saved_http_status_is_not_silent_success(tmp_path, status):
    http_fixture(tmp_path, "model-queue", status)
    with pytest.raises(ValueError, match="explicit HTTP status"):
        entry.saved_model_http_summary(tmp_path)


def test_orphan_response_is_rejected(tmp_path):
    queue = http_fixture(tmp_path, "model-queue", 200)
    (queue / ("a" * 32 + ".request.json")).unlink()
    with pytest.raises(ValueError, match="without matching"):
        entry.saved_model_http_summary(tmp_path)


@pytest.mark.parametrize(
    "requests,pending,failed,rows,code",
    [
        (1, 0, 1, [0, 1, 2], 1),
        (1, 1, 0, [0, 1, 2], 1),
        (0, 0, 0, [0, 1, 2], 1),
        (1, 0, 0, [0, 1, 2], 0),
        (1, 0, 0, [0], 1),
        (1, 0, 0, [0, 1, 1], 1),
    ],
)
def test_cli_exit_zero_and_result_file_do_not_hide_http_failures(
    tmp_path,
    monkeypatch,
    capsys,
    requests,
    pending,
    failed,
    rows,
    code,
):
    monkeypatch.setattr(entry, "prepare", lambda *a: {})
    monkeypatch.setattr(
        entry,
        "execute",
        lambda *a: {
            "completed": [
                {"original_row": row, "cli_exit_code": 0, "result_record_present": True}
                for row in rows
            ],
            "model_http": {
                "requests": requests,
                "pending": pending,
                "non200_responses": failed,
            },
            "source_checkbox_eligible": False,
        },
    )
    args = sum(
        (
            ["--" + key, str(tmp_path / key)]
            for key in ["author-root", "prepared-root", "bound-root", "output"]
        ),
        [],
    )
    assert entry.main(args + ["--execute"]) == code
    assert json.loads(capsys.readouterr().out)["source_checkbox_eligible"] is False


@pytest.mark.parametrize(
    "data,expected",
    [
        ("", False),
        ("{", False),
        ("{}", False),
        ('{"iterations":[]}', False),
        ('{"iterations":[{}]}', False),
        ('{"iterations":[{"iteration":0}]}', True),
    ],
)
def test_result_record_requires_recorded_iterations(tmp_path, data, expected):
    result = tmp_path / "original-result.jsonl"
    result.write_text(data, encoding="utf-8")
    assert entry.has_result_record(result) is expected
