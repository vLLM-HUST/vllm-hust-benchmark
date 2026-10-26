"""Fake official-interface integration only; no dataset, network or NPU usage."""

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_jsonschemabench_official")
source_preparer = importlib.import_module("prepare_jsonschemabench_author_source")


def test_publisher_archive_source_checks_bytes_and_rejects_extra_python(
    tmp_path, monkeypatch
):
    root = tmp_path / "author"
    root.mkdir()
    raw = b"# inert software fixture\n"
    path = root / "example.py"
    path.write_bytes(raw)
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "example.py",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "git_blob": hashlib.sha1(
                            b"blob " + str(len(raw)).encode() + b"\0" + raw
                        ).hexdigest(),
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(source_preparer, "PINS", pins)
    assert runner.validate_source(root, "publisher-files") == root.resolve()
    extra = root / "unexpected.py"
    extra.write_text("# inert extra")
    with pytest.raises(ValueError, match="unexpected"):
        runner.validate_source(root, "publisher-files")
    extra.unlink()
    path.write_text("# changed fixture")
    with pytest.raises(ValueError):
        runner.validate_source(root, "publisher-files")


def test_all11_native_delivery_closure_preserves_scope_and_research_evidence():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/jsonschemabench-official-workflow-readiness-20261024.json"
        ).read_text("utf-8")
    )["native_delivery_closure_20261026"]
    pins = json.loads(runner.PINS.read_text("utf-8"))
    protocols = receipt["protocols"]
    assert {row["task"] for row in protocols} == set(pins["config_names"])
    assert len(protocols) == 11
    assert sum(row["selected_original_test_prefix"] for row in protocols) == 68
    assert (
        next(row for row in protocols if row["task"] == "WashingtonPost")[
            "selected_original_test_prefix"
        ]
        == 38
    )
    assert all(
        row["selected_original_test_prefix"] == 3
        for row in protocols
        if row["task"] != "WashingtonPost"
    )
    assert receipt["source_checkbox_eligible"]
    assert receipt["original_messages_schemas_and_saved_outputs_equal"]
    assert (
        receipt["full_original_data_provider_audit"][
            "public_and_fixed_research_readbacks"
        ]
        == 99
    )
    assert not receipt["full_original_data_provider_audit"][
        "research_default_dependency_changed"
    ]
    assert not receipt["full_all_schema_inference_claim"]
    assert not receipt["performance_claim"]
    assert not receipt["model_quality_pass_claim"]


def test_task_selection_does_not_double_count_default():
    assert runner.select_tasks(["A", "B", "default"]) == ["A", "B"]
    assert runner.select_tasks(["A", "default"], ["default"]) == ["default"]
    for tasks in (["A", "default"], ["A", "A"], ["unknown"], []):
        with pytest.raises(ValueError):
            runner.select_tasks(["A", "B", "default"], tasks)


def test_fixed_publisher_source_matches_runner_revision_and_all_import_roots():
    pins = json.loads(source_preparer.PINS.read_text(encoding="utf-8"))
    assert pins["revision"] == runner.AUTHOR_REVISION
    assert len(pins["files"]) == 44
    assert sum(item["bytes"] for item in pins["files"]) == 1843390
    paths = {item["path"] for item in pins["files"]}
    assert {
        "core/__init__.py",
        "core/bench.py",
        "core/evaluator.py",
        "engines/__init__.py",
        "requirements.txt",
        "docs/custom_engine.md",
    } <= paths


@pytest.mark.parametrize(
    "value",
    [
        "https://localhost",
        "http://remote.invalid",
        "http://localhost/v1",
        "http://user:pass@localhost",  # pragma: allowlist secret - rejected inert fixture credentials
        "http://localhost/?key=fixture",
    ],
)
def test_nonlocal_or_credential_routes_rejected(value):
    with pytest.raises(ValueError):
        runner.origin(value)


def test_clean_source_identity_required(tmp_path, monkeypatch):
    dirty = False

    def git(command, **kwargs):
        if "--show-toplevel" in command:
            return str(tmp_path)
        if "HEAD" in command:
            return runner.AUTHOR_REVISION
        return " M core/dataset.py" if dirty else ""

    monkeypatch.setattr(runner.subprocess, "check_output", git)
    assert runner.validate_source(tmp_path) == tmp_path.resolve()
    dirty = True
    with pytest.raises(ValueError, match="clean fixed"):
        runner.validate_source(tmp_path)


@dataclass
class FakeOutput:
    generation: str = "fixture answer"


def test_checkpoint_keeps_failed_attempt_without_replacement(tmp_path):
    class Engine:
        def generate(self, *args):
            raise ValueError("fixture failure")

    receipt = {"attempted_samples": 0, "returned_samples": 0}
    saved = []
    engine = runner.checkpoint_engine(
        Engine(), tmp_path, receipt, lambda: saved.append(dict(receipt))
    )
    with pytest.raises(ValueError):
        engine.generate(
            "A", [{"role": "user", "content": "fixture"}], {"type": "object"}
        )
    record = json.loads((tmp_path / "attempt-000000.private.json").read_text())
    assert record["status"] == "GENERATION_FAILED_NOT_REPLACED"
    assert receipt == {
        "attempted_samples": 1,
        "returned_samples": 0,
        "new_generation_attempts": 1,
    }
    assert len(saved) == 3


def test_saved_output_reuse_has_no_model_fallback(tmp_path):
    class Engine:
        def generate(self, *args):
            raise AssertionError("must not regenerate a valid saved output")

    prior = [
        {
            "path": "inert-prior",
            "sha256": "inert-digest",
            "record": {
                "task": "A",
                "original_messages": [],
                "original_schema": {"type": "object"},
                "output": {"generation": "saved fixture"},
            },
        }
    ]
    receipt = {"attempted_samples": 0, "returned_samples": 0}
    engine = runner.checkpoint_engine(
        Engine(),
        tmp_path,
        receipt,
        lambda: None,
        prior=prior,
        restore=lambda value: FakeOutput(**value),
    )
    result = engine.generate("A", [], {"type": "object"})
    assert result.generation == "saved fixture"
    assert receipt["reused_samples"] == receipt["returned_samples"] == 1
    assert receipt.get("new_generation_attempts", 0) == 0
    with pytest.raises(ValueError, match="differ"):
        other = tmp_path / "other"
        other.mkdir()
        runner.checkpoint_engine(
            Engine(),
            other,
            {"attempted_samples": 0},
            lambda: None,
            prior=prior,
            restore=lambda value: FakeOutput(**value),
        ).generate("B", [], {})


def test_revalidate_only_empty_compiler_rejection_and_never_valid_generation(tmp_path):
    @dataclass
    class Output:
        generation: str
        metadata: dict
        generated_tokens: list

    calls = []

    class Engine:
        def generate(self, *args):
            calls.append(args)
            return Output("new", {"compile_status": {"code": 0}}, ["new"])

    old = {
        "generation": "",
        "metadata": {"compile_status": {"code": 1}},
        "generated_tokens": [],
    }
    prior = [
        {
            "path": "old",
            "sha256": "digest",
            "record": {
                "task": "A",
                "original_messages": [],
                "original_schema": {},
                "output": old,
            },
        }
    ]
    receipt = {"attempted_samples": 0, "returned_samples": 0}
    engine = runner.checkpoint_engine(
        Engine(),
        tmp_path,
        receipt,
        lambda: None,
        prior=prior,
        restore=lambda value: Output(**value),
        revalidate_rejections=[0],
    )
    assert engine.generate("A", [], {}).generation == "new"
    assert len(calls) == 1 and receipt["revalidated_prior_compile_rejections"] == 1
    assert receipt.get("reused_samples", 0) == 0
    assert old["generation"] == ""
    old["generation"] = "already generated"
    other = tmp_path / "other"
    other.mkdir()
    engine = runner.checkpoint_engine(
        Engine(),
        other,
        {"attempted_samples": 0},
        lambda: None,
        prior=prior,
        restore=lambda value: Output(**value),
        revalidate_rejections=[0],
    )
    with pytest.raises(ValueError, match="only empty"):
        engine.generate("A", [], {})
    assert len(calls) == 1


def test_reuse_failed_prefix_requires_terminal_profile_and_complete_inputs(tmp_path):
    expected = {"source_id": "fixture", "origin": "http://127.0.0.1:19466"}
    receipt = {
        **expected,
        "status": "FAILED_NATIVE_FLOW_NOT_COMPLETE",
        "attempted_samples": 2,
        "returned_samples": 1,
    }
    (tmp_path / "receipt-public.json").write_text(json.dumps(receipt))
    saved = {
        "task": "A",
        "original_messages": [],
        "original_schema": {},
        "status": "GENERATION_RETURNED_NOT_YET_OFFICIALLY_EVALUATED",
        "output": {"task": "A", "messages": [], "schema": {}},
    }
    (tmp_path / "attempt-000000.private.json").write_text(json.dumps(saved))
    tail = tmp_path / "attempt-000001.private.json"
    tail.write_text(json.dumps({"status": "GENERATION_FAILED_NOT_REPLACED"}))
    with pytest.raises(ValueError, match="independently confirm"):
        runner.reusable_prefix(tmp_path, expected)
    prefix, digest = runner.reusable_prefix(
        tmp_path, expected, failed_tail_terminal=True
    )
    assert len(prefix) == 1 and len(digest) == 64
    with pytest.raises(ValueError, match="profile differs"):
        runner.reusable_prefix(
            tmp_path, {**expected, "model": "different"}, failed_tail_terminal=True
        )
    tail.write_text(json.dumps({"status": "STARTED"}))
    with pytest.raises(ValueError, match="unresolved"):
        runner.reusable_prefix(tmp_path, expected, failed_tail_terminal=True)


def test_official_bench_bridge_preserves_location_and_defaults(tmp_path, monkeypatch):
    source, snapshot, out = (
        tmp_path / name for name in ["author", "snapshot", "output"]
    )
    source.mkdir()
    snapshot.mkdir()

    def verified_source(p, layout="git"):
        assert layout == "git"
        return Path(p).resolve()

    monkeypatch.setattr(runner, "validate_source", verified_source)
    monkeypatch.setattr(runner, "validate_snapshot", lambda p, pins: Path(p).resolve())
    dataset = NS(
        DATASET_HUGGINGFACE_PATH="original",
        Dataset=lambda config: [1, 2, 3],
        DatasetConfig=lambda task: task,
    )
    observed = []

    class Engine:
        def generate(self, task, messages, schema):
            return FakeOutput()

    def bench(**kwargs):
        observed.append(kwargs)
        assert "messages_formatter" not in kwargs
        assert kwargs["save_outputs"] is True
        assert dataset.DATASET_HUGGINGFACE_PATH == str(snapshot.resolve())
        return [
            [
                kwargs["engine"].generate(
                    task,
                    [{"role": "user", "content": "original fixture"}],
                    {"type": "object"},
                )
                for _ in range(kwargs["limit"])
            ]
            for task in kwargs["tasks"]
        ]

    modules = {
        "core.dataset": dataset,
        "core.bench": NS(bench=bench),
        "core.engine": NS(),
        "core.types": NS(),
    }
    real_import = runner.importlib.import_module
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: modules[name] if name in modules else real_import(name),
    )
    monkeypatch.delitem(sys.modules, "core", raising=False)
    sdk = ModuleType("openai")
    sdk.OpenAI = lambda **kwargs: NS(close=lambda: None)
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.setattr(runner, "create_engine", lambda **kwargs: Engine())
    args = NS(
        official_source=source,
        snapshot_root=snapshot,
        output=out,
        tasks=["Github_easy"],
        full=False,
        limit=3,
        model="fixture-local",
        max_tokens=128,
        context_capacity=4096,
        origin="http://localhost:19466",
    )
    runner.run(args)
    receipt = json.loads((out / "receipt-public.json").read_text())
    assert len(observed) == 1
    assert receipt["status"] == "ORIGINAL_BENCH_AND_EVALUATOR_RETURNED"
    assert receipt["attempted_samples"] == receipt["returned_samples"] == 3
    assert receipt["schema_rewriting"] is False
    assert receipt["source_checkbox_eligible"] is False
    assert len(list(out.glob("attempt-*.private.json"))) == 3
    assert dataset.DATASET_HUGGINGFACE_PATH == "original"
