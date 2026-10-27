"""Inert isolation/identity controls; no model or Python candidate execution."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_mbpp_isolated_scoring")
worker = importlib.import_module("mbpp_isolated_worker")
generator = importlib.import_module("run_mbpp_original_generation")
preparer = importlib.import_module("prepare_mbpp_scoring_inputs")


def task(i):
    return {
        "task_id": i,
        "text": "inert fixture",
        "code": "reference fixture",
        "test_setup_code": "original setup fixture",
        "test_list": ["test fixture one", "test fixture two", "test fixture three"],
        "challenge_test_list": ["separate challenge fixture"],
    }


def test_assembly_preserves_original_setup_three_tests_and_raw_code():
    tasks = [
        {
            "task_id": i,
            "original_task": task(i),
            "completion": "raw invalid code",
            "finish_reason": "stop",
        }
        for i in (11, 12, 13)
    ]
    tasks[0]["completion"] = ""
    references, candidates, gold = worker.assemble(tasks)
    assert references == ["test fixture one\ntest fixture two\ntest fixture three"] * 3
    assert candidates[0] == ["original setup fixture\n"]
    assert candidates[1] == ["original setup fixture\nraw invalid code"]
    assert gold[0] == ["original setup fixture\nreference fixture"]
    assert "separate challenge fixture" not in "".join(references)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong-completion"])
def test_result_coverage_never_omits_failures(failure):
    results = {
        i: [(0, {"task_id": i, "completion_id": 0, "passed": i != 1})] for i in range(3)
    }
    assert worker.checked_results(results, 3) == [True, False, True]
    if failure == "missing":
        results.pop(1)
    elif failure == "duplicate":
        results[0].append(results[0][0])
    else:
        results[0][0][1]["completion_id"] = 1
    with pytest.raises(ValueError):
        worker.checked_results(results, 3)


def test_launcher_never_falls_back_to_host_on_unsupported_platform(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no host fallback"):
        runner.launch(SimpleNamespace())


@pytest.mark.parametrize(
    "root", ["/data/private/python310", "/root/python310", "/proc/runtime"]
)
def test_runtime_prefix_never_creates_forbidden_namespace_root(monkeypatch, root):
    monkeypatch.setattr(runner, "validate_runtime", lambda value: Path(value))
    with pytest.raises(ValueError, match="forbidden namespace root"):
        runner.namespace_runtime(root)


def test_accessible_dedicated_runtime_prefix_is_allowed(monkeypatch):
    monkeypatch.setattr(runner, "validate_runtime", lambda value: Path(value))
    assert runner.namespace_runtime("/opt/mbpp-python310") == Path(
        "/opt/mbpp-python310"
    )


def test_worker_refuses_before_scorer_import_when_isolation_guard_fails(monkeypatch):
    def reject(helper):
        raise RuntimeError("fixture isolation not established")

    monkeypatch.setattr(worker, "isolation_guard", reject)
    with pytest.raises(RuntimeError, match="isolation not established"):
        worker.evaluate()


def test_author_source_is_checked_without_importing_it(tmp_path):
    root = tmp_path / "author"
    folder = root / "bigcode_eval/tasks/custom_metrics"
    folder.mkdir(parents=True)
    pins = {"files": {}}
    for name in ("__init__.py", "code_eval.py", "execute.py"):
        raw = b"raise RuntimeError('must not import fixture')\n"
        (folder / name).write_bytes(raw)
        pins["files"][f"bigcode_eval/tasks/custom_metrics/{name}"] = hashlib.sha256(
            raw
        ).hexdigest()
    assert len(runner.author_modules(root, pins)) == 3
    (folder / "execute.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned evaluator"):
        runner.author_modules(root, pins)


@pytest.fixture
def prepared_chain(tmp_path):
    rows = [task(i) for i in range(11, 511)]
    demos = [task(i) for i in (2, 3, 4)]
    source = {"revision": "inert fixture"}

    def transport(url, body):
        if url.endswith("/tokenize"):
            return {"count": 10}
        return {
            "choices": [{"text": "raw invalid fixture", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    generation, prepared = tmp_path / "generation", tmp_path / "prepared"
    generator.generate(
        list(enumerate(rows[:3])),
        demos,
        output=generation,
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture-model",
        max_tokens=32,
        context=128,
        source_identity=source,
        transport=transport,
    )
    preparer.prepare(
        generation, rows=rows, examples=demos, source_identity=source, output=prepared
    )
    return prepared, generation, rows, demos, source


@pytest.mark.parametrize("change", [None, "task", "completion", "receipt"])
def test_prepared_chain_rechecked_against_original_saved_generation(
    prepared_chain, change
):
    prepared, generation, rows, demos, source = prepared_chain
    task_file = prepared / "tasks-private.json"
    receipt_file = prepared / "receipt-private.json"
    tasks = json.loads(task_file.read_text("utf-8"))
    receipt = json.loads(receipt_file.read_text("utf-8"))
    if change == "task":
        tasks[0]["original_task"]["test_list"] = ["replacement test"]
    elif change == "completion":
        tasks[0]["completion"] = "posthoc cleaned code"
    elif change == "receipt":
        receipt["generation_receipt_sha256"] = "changed"
    if change is not None:
        task_file.write_text(json.dumps(tasks), encoding="utf-8")
        receipt["tasks_sha256"] = runner.digest(task_file)
        receipt_file.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(ValueError):
            runner.original_inputs(
                prepared, generation, rows=rows, examples=demos, source_identity=source
            )
    else:
        checked, _, _ = runner.original_inputs(
            prepared, generation, rows=rows, examples=demos, source_identity=source
        )
        assert checked == tasks


@pytest.mark.parametrize("failure", [None, "worker-exit", "incomplete-results"])
def test_launcher_staging_with_mocked_namespace_process(
    tmp_path, prepared_chain, monkeypatch, failure
):
    """Only command/staging wiring: the fake process never evaluates any code."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    _, generation, rows, demos, _ = prepared_chain
    asset_root = tmp_path / "assets"
    files = []
    for split, payload in (("test", rows), ("prompt", [task(i) for i in range(1, 11)])):
        relative = f"full/{split}-00000-of-00001.parquet"
        path = asset_root / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(payload), path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": runner.digest(path),
            }
        )
    asset_pins = tmp_path / "asset-pins.json"
    asset_pins.write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": "fixture-only",
                    "revision": "fixture",
                    "files": files,
                }
            }
        ),
        encoding="utf-8",
    )
    identity = {
        "repo_id": "fixture-only",
        "revision": "fixture",
        "files": files,
        "asset_manifest_sha256": runner.digest(asset_pins),
    }
    gen_receipt = generation / "receipt-private.json"
    value = json.loads(gen_receipt.read_text("utf-8"))
    value["source_identity"] = identity
    gen_receipt.write_text(json.dumps(value), encoding="utf-8")
    prepared = tmp_path / "rebound"
    preparer.prepare(
        generation, rows=rows, examples=demos, source_identity=identity, output=prepared
    )
    pins = json.loads(runner.PINS.read_text("utf-8"))
    author = tmp_path / "author"
    for relative in pins["files"]:
        path = author / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "raise RuntimeError('inert fixture: never import')\n", encoding="utf-8"
        )
        pins["files"][relative] = runner.digest(path)
    scoring_pins = tmp_path / "scoring-pins.json"
    scoring_pins.write_text(json.dumps(pins), encoding="utf-8")
    packages = tmp_path / "packages"
    for name, version in pins["client_versions"].items():
        metadata = packages / f"{name}-{version}.dist-info/METADATA"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    for name in ("numpy", "numpy.libs", "tqdm"):
        (packages / name).mkdir()
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "lib").mkdir()
    (runtime / "bin/python3").write_text("inert interpreter fixture", encoding="utf-8")
    work = tmp_path / "work"
    monkeypatch.setattr(runner, "ASSET_PINS", asset_pins)
    monkeypatch.setattr(runner, "PINS", scoring_pins)
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(runner.os, "chown", lambda *args: None, raising=False)
    monkeypatch.setattr(runner.shutil, "which", lambda name: "/fixture/" + name)
    calls = []

    def fake_process(command, **kwargs):
        calls.append(command)
        assert "unshare" in command and "--kill-child" in command
        assert (
            "--user" in command
            and "--mount" in command
            and "--net" in command
            and "--pid" in command
        )
        assert "HF_ALLOW_CODE_EVAL" not in kwargs["env"]
        if failure == "worker-exit":
            return SimpleNamespace(returncode=1)
        report = {
            "source_id": "MBPP",
            "status": "SELECTED_ORIGINAL_TESTS_CODEEVAL_SCORED",
            "tasks_sha256": runner.digest(prepared / "tasks-private.json"),
            "task_ids": [11, 12, 13],
            "evaluator_revision": pins["evaluator_revision"],
            "original_reference_checks_passed": True,
            "isolation_checks_passed_before_import_and_execution": True,
            "generated_code_executed": True,
            "passed": [] if failure == "incomplete-results" else [True, False, True],
            "metric": {"pass@1": 2 / 3},
        }
        (work / "assets/output/receipt-public.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_process)
    args = SimpleNamespace(
        assets=asset_root,
        generation=generation,
        prepared_dir=prepared,
        author_root=author,
        package_root=packages,
        python_runtime=runtime,
        work_root=work,
        sandbox_uid=91001,
        deadline=600,
    )
    if failure:
        with pytest.raises(RuntimeError):
            runner.launch(args)
        assert (work / "failure-public.json").is_file()
    else:
        result = runner.launch(args)
        assert not result["source_checkbox_eligible"]
        assert result["passed"] == [True, False, True]
        assert (work / "assets/inputs/humaneval_isolated_worker.py").is_file()
    assert len(calls) == 1
