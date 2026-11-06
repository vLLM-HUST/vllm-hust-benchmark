"""Source/selection fixtures only; no VM, model, setup or grader execution."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_osworld_original as runner


@pytest.mark.parametrize("emit_score", [True, False])
def test_execution_delegates_original_task_and_restores_runtime(
    args, monkeypatch, emit_score
):
    """Execution fixture only: no VM, HTTP, model or scientific evidence."""
    calls = []
    for name in [
        "desktop_env",
        "desktop_env.controllers",
        "desktop_env.evaluators",
        "desktop_env.evaluators.getters",
        "mm_agents",
    ]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    modules = {}
    for name in [
        "desktop_env.desktop_env",
        "desktop_env.controllers.python",
        "desktop_env.controllers.setup",
        "desktop_env.evaluators.getters.chrome",
        "mm_agents.qwen3vl_agent",
        "lib_run_single",
    ]:
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    environment = modules["desktop_env.desktop_env"]
    factory = object()
    environment.create_vm_manager_and_provider = factory
    env = SimpleNamespace(close=lambda: calls.append("closed"))
    environment.DesktopEnv = lambda **kwargs: env
    for name in ["desktop_env.controllers.python", "desktop_env.controllers.setup"]:
        modules[name].requests = object()
    playwright = object()
    modules["desktop_env.evaluators.getters.chrome"].sync_playwright = playwright

    class Agent:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def reset(self, _logger=None):
            pass

    modules["mm_agents.qwen3vl_agent"].Qwen3VLAgent = Agent

    def original_run(
        agent,
        actual_env,
        example,
        steps,
        instruction,
        original_args,
        destination,
        scores,
    ):
        assert actual_env is env and steps == 15 and original_args.max_tokens == 32768
        assert instruction == example["instruction"] == "original fixture"
        assert example["evaluator"]["expected"] == ["original-reference"]
        calls.append(example["id"])
        if emit_score:
            scores.append(0.0)

    modules["lib_run_single"].run_single_example = original_run
    monkeypatch.setattr(runner, "check_client", lambda *_: None)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self):
            return json.dumps({"data": [{"id": args.model}]}).encode()

    monkeypatch.setattr(runner, "urlopen", lambda *_, **__: Response())
    monkeypatch.setenv("OPENAI_BASE_URL", "retained-original-endpoint")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-original-key")
    prepared = runner.plan(args)
    if emit_score:
        result = runner.execute(args, prepared)
        assert result["native_tasks"] == 3 and not result["source_checkbox_eligible"]
    else:
        with pytest.raises(ValueError, match="original finite task score"):
            runner.execute(args, prepared)
    import os

    assert os.environ["OPENAI_BASE_URL"] == "retained-original-endpoint"
    assert os.environ["OPENAI_API_KEY"] == "fixture-original-key"
    assert calls[-1] == "closed"
    assert environment.create_vm_manager_and_provider is factory
    assert (
        modules["desktop_env.evaluators.getters.chrome"].sync_playwright is playwright
    )
    saved = json.loads((args.output / "execution-public.json").read_text())
    assert saved["source_checkbox_eligible"] is False


@pytest.fixture
def args(tmp_path, monkeypatch):
    source = tmp_path / "author"
    base = source / "evaluation_examples"
    cases = base / "examples/chrome"
    cases.mkdir(parents=True)
    (base / "test_all.json").write_text(
        json.dumps({"chrome": [f"case-{i}" for i in range(4)]})
    )
    files = []
    for index in range(4):
        path = cases / f"case-{index}.json"
        raw = json.dumps(
            {
                "id": f"case-{index}",
                "instruction": "original fixture",
                "config": [{"type": "fixture-original"}],
                "evaluator": {"expected": ["original-reference"]},
            }
        ).encode()
        path.write_bytes(raw)
        files.append(
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps({"revision": "fixture", "files": files}))
    monkeypatch.setattr(runner, "PROFILE", pins)
    monkeypatch.setattr(runner, "verify_source", lambda root, profile: source)
    return SimpleNamespace(
        author_root=source,
        output=tmp_path / "fresh-output",
        model="fixture-actual-model",
        api_base="http://localhost:19476/v1",
        api_key_env=None,
        provider="docker",
        vm_path="operator-disposable-native-vm",
        owned_qemu_profile=None,
        slow_arm_tcg=False,
        case_index=None,
        limit=3,
        all_tasks=False,
    )


def test_offline_plan_preserves_fixed_prefix_and_original_budgets(args):
    original = (
        args.author_root / "evaluation_examples/examples/chrome/case-0.json"
    ).read_bytes()
    result = runner.plan(args)
    assert [task["id"] for task in result["tasks"]] == ["case-0", "case-1", "case-2"]
    assert result["max_steps"] == 15 and result["max_tokens"] == 32768
    assert result["native_tasks"] == 0 and not result["source_checkbox_eligible"]
    assert not args.output.exists()
    assert (
        args.author_root / "evaluation_examples/examples/chrome/case-0.json"
    ).read_bytes() == original


def test_explicit_all_and_single_fixed_case(args):
    args.all_tasks = True
    assert len(runner.plan(args)["tasks"]) == 4
    args.all_tasks = False
    args.case_index = 2
    assert runner.plan(args)["tasks"][0]["id"] == "case-2"
    args.case_index = 3
    with pytest.raises(ValueError, match="first three"):
        runner.plan(args)


@pytest.mark.parametrize(
    "base",
    [
        "https://name:password@example.org/v1",
        "http://localhost/v1?x=1",
        "http://localhost/not-v1",
    ],
)
def test_ambiguous_endpoint_rejected(args, base):
    args.api_base = base
    with pytest.raises(ValueError, match="endpoint"):
        runner.plan(args)


def test_existing_output_and_implicit_vm_allocation_refused(args):
    args.output.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        runner.plan(args)
    args.output.rmdir()
    args.vm_path = None
    with pytest.raises(ValueError, match="explicit disposable"):
        runner.plan(args)


def test_owned_qemu_scope_does_not_claim_untested_full_catalog(args, tmp_path):
    profile = tmp_path / "owned.json"
    profile.write_text(
        json.dumps(
            {
                "scope": "operator-owned-disposable-qemu",
                "monitor_socket": "/control/monitor.sock",
                "snapshot_name": "qualified_baseline",
                "ports": {"server": 15000, "chromium": 19222, "vnc": 0, "vlc": 0},
            }
        )
    )
    args.provider, args.owned_qemu_profile, args.slow_arm_tcg = (
        "owned-qemu",
        profile,
        True,
    )
    assert runner.plan(args)["owned_profile"]["snapshot_name"] == "qualified_baseline"
    args.all_tasks = True
    with pytest.raises(ValueError, match="fixed first three"):
        runner.plan(args)


def test_original_task_tampering_rejected(args):
    path = args.author_root / "evaluation_examples/examples/chrome/case-0.json"
    path.write_text('{"id":"case-0","instruction":"changed"}')
    with pytest.raises(ValueError, match="identity"):
        runner.plan(args)
