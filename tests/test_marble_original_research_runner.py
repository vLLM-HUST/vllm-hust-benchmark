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


def fixture_prefix(tmp_path):
    source = tmp_path / "original-prefix"
    source.mkdir()
    rows = []
    for index in range(3):
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
                "selected_configurations": 3,
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
    with pytest.raises(ValueError, match="first3"):
        entry.original_prefix(source, bound)


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
