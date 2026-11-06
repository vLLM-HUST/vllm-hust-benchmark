"""Software fixtures only; no website, model or original benchmark execution."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "webarena_runner_test", ROOT / "scripts/run_webarena_agentlab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_does_not_require_framework_or_model():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_webarena_agentlab.py"),
            "--task-id",
            "0",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "status": "PLAN_ONLY",
        "source_id": "WEBARENA",
        "task_id": 0,
        "source_checkbox_eligible": False,
    }


def test_required_site_and_unavailable_sites_are_distinct(runner, tmp_path):
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"SHOPPING_ADMIN": "http://127.0.0.1:19680/admin"}))
    configured, missing = runner.site_configuration(sites, ["shopping_admin"])
    assert configured["SHOPPING_ADMIN"] == "http://127.0.0.1:19680/admin"
    assert len(missing) == 6
    assert all(
        configured[key].startswith("http://127.0.0.1:1/unconfigured/")
        for key in missing
    )
    assert len(set(configured.values())) == 7
    with pytest.raises(ValueError, match="required"):
        runner.site_configuration(sites, ["shopping"])


@pytest.mark.parametrize(
    "value", [[], {"UNKNOWN": "http://127.0.0.1:1"}, {"SHOPPING_ADMIN": 1}]
)
def test_invalid_site_map_rejected(runner, tmp_path, value):
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.site_configuration(sites, ["shopping_admin"])


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1:1000/?secret=x",
        "http://user:password@127.0.0.1:1000",
        "http://127.0.0.1:1000/#fragment",
    ],
)
def test_site_endpoint_trust_boundary(runner, tmp_path, url):
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"SHOPPING_ADMIN": url}))
    with pytest.raises(argparse.ArgumentTypeError, match="loopback"):
        runner.site_configuration(sites, ["shopping_admin"])


def test_complete_original_task_preserved(runner, monkeypatch, tmp_path):
    root = tmp_path / "author"
    (root / "config_files").mkdir(parents=True)
    task = {
        "task_id": 0,
        "sites": ["shopping_admin"],
        "eval": {"reference_answers": {"exact_match": "fixture"}},
        "image": None,
        "require_login": True,
    }
    (root / "config_files/test.raw.json").write_text(json.dumps([task]))
    calls = []
    monkeypatch.setattr(runner, "verify_source", lambda *args: calls.append(args))
    assert runner.original_task(root, 0) == task
    assert calls == [(root, runner.PINS)]
    for invalid in (-1, 1, True):
        with pytest.raises(ValueError, match="task ID"):
            runner.original_task(root, invalid)


def test_unverified_original_source_is_rejected(runner, tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.original_task(tmp_path, 0)


def args_for(tmp_path):
    root = tmp_path / "author"
    root.mkdir()
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"SHOPPING_ADMIN": "http://127.0.0.1:19680/admin"}))
    return SimpleNamespace(
        author_root=root,
        sites=sites,
        output=tmp_path / "results",
        model="Qwen/Qwen3.5-9B",
        endpoint="http://127.0.0.1:19476/v1",
        queue=None,
        max_steps=30,
        seed=42,
        max_output_tokens=None,
        task_id=0,
    )


def test_modified_packaged_task_rejected_before_runtime_import(
    runner, monkeypatch, tmp_path
):
    args = args_for(tmp_path)
    task = {
        "task_id": 0,
        "sites": ["shopping_admin"],
        "eval": {"reference_answers": "original fixture"},
    }
    monkeypatch.setattr(runner, "original_task", lambda *_: task)
    monkeypatch.setattr(runner.metadata, "version", runner.VERSIONS.__getitem__)
    packaged = tmp_path / "packaged.json"
    packaged.write_text(
        json.dumps([{**task, "eval": {"reference_answers": "changed fixture"}}])
    )
    monkeypatch.setattr(
        runner.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: packaged),
    )
    with pytest.raises(ValueError, match="packaged task differs"):
        runner.execute(args)
    assert not args.output.exists()


def test_existing_output_rejected(runner, monkeypatch, tmp_path):
    args = args_for(tmp_path)
    args.output.mkdir()
    monkeypatch.setattr(
        runner, "original_task", lambda *_: {"sites": ["shopping_admin"]}
    )
    with pytest.raises(ValueError, match="fresh output"):
        runner.execute(args)


def test_endpoint_and_queue_are_exclusive(runner, tmp_path):
    args = args_for(tmp_path)
    args.queue = tmp_path / "queue"
    with pytest.raises(ValueError, match="choose one"):
        runner.execute(args)


def test_private_queue_cannot_overlap_original_source(runner, monkeypatch, tmp_path):
    args = args_for(tmp_path)
    args.endpoint = None
    args.queue = args.author_root
    monkeypatch.setattr(
        runner, "original_task", lambda *_: {"sites": ["shopping_admin"]}
    )
    with pytest.raises(ValueError, match="separate from outputs and source"):
        runner.execute(args)


@pytest.mark.parametrize("steps,tokens", [(0, None), (30, 0), (-1, 100)])
def test_invalid_budgets_rejected(runner, tmp_path, steps, tokens):
    args = args_for(tmp_path)
    args.max_steps, args.max_output_tokens = steps, tokens
    with pytest.raises(ValueError, match="positive budgets"):
        runner.execute(args)
