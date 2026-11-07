"""Invented offline recipe controls, not original PaperBench execution."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pb_prerequisites", ROOT / "scripts/check_paperbench_native_prerequisites.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


CANONICAL = """### Canonical command

```bash
uv run python -m paperbench.nano.entrypoint \
  paperbench.paper_split=all \
  paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
  paperbench.solver.completer_config=paperbench.solvers.basicagent.completer:OpenAIResponsesTurnCompleterConfig \
  paperbench.solver.completer_config.tools="[{'type':'web_search'}]" \
  paperbench.solver.time_limit=86400 \
  paperbench.reproduction.timeout=86400 \
  runner.max_retries=0
```
"""


def fixture(tmp_path):
    texts = [
        "Miniconda3-py313_25.5.1-0-Linux-x86_64.sh\nconda tos accept\n",
        "FROM invented-control-only\n",
        "--platform=linux/amd64\n--platform=linux/amd64\n",
        "time_limit: int | None = chz.field(default=5 * 60\n"
        "OpenAIResponsesTurnCompleterConfig(\n"
        'tools=[WebSearchToolParam(type="web_search_preview")]\n',
        "paperbench.reproduction.timeout=60\n\n" + CANONICAL,
        'default="pb-env:latest"\n',
        'docker_image="pb-reproducer:latest"\n'
        'docker_image="pb-env:latest"\n'
        "grade_locally: bool = True\n",
    ]
    entries = []
    for name, text in zip(runner.RECIPES, texts, strict=True):
        raw = text.encode()
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        entries.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
            }
        )
    pins = tmp_path / "fixture-pins.json"
    pins.write_text(
        json.dumps(
            {
                "source_id": "PAPERBENCH",
                "revision": "invented-control",
                "files": entries,
            }
        )
    )
    return pins


@pytest.mark.parametrize(
    "architecture,match", [("aarch64", False), ("x86_64", True), ("AMD64", True)]
)
def test_architecture_never_promotes_preflight_to_execution(
    tmp_path, architecture, match
):
    pins = fixture(tmp_path)
    result = runner.inspect_original(
        tmp_path, pins_path=pins, host_architecture=architecture
    )
    assert result["native_host_architecture_matches"] is match
    assert result["verified_original_files"] == 7
    assert result["original_image_roles"] == {
        "agent": "pb-env:latest",
        "reproduction": "pb-reproducer:latest",
        "remote_judge": "pb-env:latest",
        "judge_defaults_to_local": True,
    }
    assert result["original_recipe_auto_accepts_anaconda_terms"]
    assert not result["vendor_terms_accepted_by_this_check"]
    assert not result["cross_arch_execution_qualified"]
    assert not result["docker_built_or_run"]
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]
    assert (
        result["publisher_reproduction_entry"]
        == "project/paperbench/paperbench/reproduce.py"
    )
    assert result["publisher_scoring_entry"] == "project/paperbench/paperbench/grade.py"
    assert not result["native_agent_reproduction_or_scoring_executed"]
    assert result["original_basicagent_default_model_api"] == "OpenAI Responses"
    assert result["original_basicagent_default_builtin_tools"] == ["web_search_preview"]
    assert not result["model_endpoint_and_tool_execution_qualified"]
    assert not result["chat_completions_health_check_is_sufficient"]
    assert result["original_canonical_protocol"] == {
        "paper_split": "all",
        "agent_time_limit_seconds": 86400,
        "reproduction_timeout_seconds": 86400,
        "responses_builtin_tools": ["web_search"],
        "runner_max_retries": 0,
    }


@pytest.mark.parametrize(
    "argument,replacement",
    [
        ("paperbench.solver.time_limit=86400", "paperbench.solver.time_limit=300"),
        ("paperbench.reproduction.timeout=86400", "paperbench.reproduction.timeout=60"),
        ("web_search", "no_tools"),
        ("runner.max_retries=0", "runner.max_retries=0 runner.max_retries=1"),
    ],
)
def test_canonical_command_cannot_be_replaced_by_dev_profile(argument, replacement):
    with pytest.raises(ValueError, match="canonical"):
        runner.canonical_protocol(CANONICAL.replace(argument, replacement))


def test_modified_source_is_rejected_before_recipe_inspection(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / runner.RECIPES[0]).write_text("changed")
    with pytest.raises(ValueError):
        runner.inspect_original(tmp_path, pins_path=pins)


def test_missing_required_recipe_is_rejected(tmp_path):
    pins = fixture(tmp_path)
    value = json.loads(pins.read_text())
    value["files"].pop()
    pins.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="pins required"):
        runner.inspect_original(tmp_path, pins_path=pins)


@pytest.mark.parametrize(
    "missing", ["OpenAIResponsesTurnCompleterConfig(", "web_search_preview"]
)
def test_changed_model_contract_requires_review(tmp_path, missing):
    pins = fixture(tmp_path)
    target = tmp_path / runner.RECIPES[3]
    raw = target.read_bytes().replace(missing.encode(), b"invented-control")
    target.write_bytes(raw)
    value = json.loads(pins.read_text())
    item = next(x for x in value["files"] if x["path"] == runner.RECIPES[3])
    item.update(
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        git_blob=hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
    )
    pins.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="model/tool contract"):
        runner.inspect_original(tmp_path, pins_path=pins)
