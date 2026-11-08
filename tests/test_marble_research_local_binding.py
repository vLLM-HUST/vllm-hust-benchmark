"""Pure deployment-binding fixtures; no native task or model execution."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "marble_local_binding",
    Path(__file__).resolve().parents[1] / "scripts/bind_marble_research_local.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def configuration():
    return {
        "task": {"content": "synthetic original fixture", "hidden_reference": ["keep"]},
        "environment": {"type": "Research", "max_iterations": 5, "extension": True},
        "coordinate_mode": "graph",
        "memory": {"type": "BaseMemory"},
        "agents": [{"agent_id": "a", "extension": {"keep": 1}}],
        "relationships": [["a", "a", "fixture"]],
        "engine_planner": {"opaque": True},
        "llm": "gpt-3.5-turbo",
        "metrics": {"evaluate_llm": "gpt-4o", "relevance": True},
        "output": {"file_path": "original/path", "extension": "keep"},
        "unknown_extension": {"nested": [1, 2]},
    }


def test_binding_changes_only_three_fields(tmp_path):
    original = configuration()
    snapshot = deepcopy(original)
    bound = entry.bind_config(
        original,
        model="openai/local-actor",
        judge="openai/local-judge",
        result_path=tmp_path / "result.jsonl",
    )
    assert original == snapshot
    for key in original.keys() - {"llm", "metrics", "output"}:
        assert bound[key] == original[key]
    assert bound["metrics"]["relevance"] and bound["output"]["extension"] == "keep"


@pytest.mark.parametrize(
    "field,value",
    [
        ("coordinate_mode", "centralized"),
        ("llm", "different"),
        ("environment", {"type": "Research", "max_iterations": 3}),
        ("agents", [{"agent_id": "a", "llm": "external-model"}]),
    ],
)
def test_non_original_runtime_rejected(tmp_path, field, value):
    config = configuration()
    config[field] = value
    with pytest.raises(ValueError):
        entry.bind_config(
            config,
            model="openai/local",
            judge="openai/local",
            result_path=tmp_path / "result",
        )


def test_plan_does_not_read_source_or_create_output(tmp_path):
    result = entry.prepare(
        tmp_path / "absent",
        tmp_path / "out",
        model="openai/local",
        judge="openai/local",
    )
    assert result["status"] == "PLAN_ONLY" and result["model_calls"] == 0
    assert not list(tmp_path.iterdir())


def prepare_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "original-row-00000.yaml"
    raw = json.dumps(configuration()).encode()
    path.write_bytes(raw)
    receipt = {
        "source_id": "MULTIAGENTBENCH",
        "status": "OFFICIAL_RESEARCH_RUNTIME_PROFILE_PREPARED_NOT_EXECUTED",
        "runtime_profile": "official-research",
        "selected_configurations": 1,
        "configurations": [
            {
                "source_file": "multiagentbench/research/research_main.jsonl",
                "source_row_index": 0,
                "configuration": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }
    (source / "preparation-public.json").write_text(json.dumps(receipt))
    return source, path


def test_exact_prefix_bound_without_models(tmp_path):
    source, _ = prepare_fixture(tmp_path)
    result = entry.prepare(
        source,
        tmp_path / "output",
        model="openai/local",
        judge="openai/local-judge",
        execute=True,
    )
    assert result["selected_configurations"] == 1
    bound = json.loads((tmp_path / "output/original-row-00000.yaml").read_text())
    assert bound["task"] == configuration()["task"]
    assert result["original_iterations"] == 5 and not result["native_execution"]


def test_modified_original_rejected_before_output_creation(tmp_path):
    source, path = prepare_fixture(tmp_path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.prepare(
            source,
            tmp_path / "output",
            model="openai/local",
            judge="openai/local",
            execute=True,
        )
    assert not (tmp_path / "output").exists()


def test_shuffled_prefix_rejected(tmp_path):
    source, _ = prepare_fixture(tmp_path)
    p = source / "preparation-public.json"
    receipt = json.loads(p.read_text())
    receipt["configurations"][0]["source_row_index"] = 1
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prefix"):
        entry.prepare(
            source,
            tmp_path / "output",
            model="openai/local",
            judge="openai/local",
            execute=True,
        )
