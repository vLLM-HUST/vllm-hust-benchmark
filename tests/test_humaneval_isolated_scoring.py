"""CPU fixtures for preparation and fail-closed controls, not native scoring."""

import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "isolated_humaneval", ROOT / "scripts/run_humaneval_isolated_scoring.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path):
    author, prepared = tmp_path / "author", tmp_path / "prepared"
    (author / "human_eval").mkdir(parents=True)
    (author / "data").mkdir()
    prepared.mkdir()
    module = author / "human_eval/data.py"
    module.write_text("# inert software fixture\n")
    problems = [
        {
            "task_id": f"HumanEval/{n}",
            "prompt": "original",
            "test": "original oracle",
            "entry_point": "f",
            "canonical_solution": "original reference",
        }
        for n in range(4)
    ]
    data = author / "data/HumanEval.jsonl.gz"
    with gzip.open(data, "wt") as stream:
        for row in problems:
            stream.write(json.dumps(row) + "\n")
    samples = prepared / "samples-private.jsonl"
    samples.write_text(
        "".join(
            json.dumps(
                {"task_id": row["task_id"], "completion": "invalid candidate retained"}
            )
            + "\n"
            for row in problems[:3]
        )
    )
    pins = {
        "files": {"human_eval/data.py": runner.digest(module)},
        "revision": "fixture",
        "data_sha256": runner.digest(data),
    }
    receipt = {
        "source_id": "HUMANEVAL",
        "status": "OFFICIAL_SCORER_INPUT_PREPARED_NOT_EXECUTED",
        "source_revision": "fixture",
        "completion_postprocessing": "NONE_EXACT_ORIGINAL_TEXT",
        "samples_sha256": runner.digest(samples),
        "historical_samples": [{"task_id": row["task_id"]} for row in problems[:3]],
    }
    (prepared / "receipt-private.json").write_text(json.dumps(receipt))
    return author, prepared, pins, problems


def test_preserves_all_original_fields_and_candidate_text_without_execution(tmp_path):
    author, prepared, pins, problems = fixture(tmp_path)
    selected, samples = runner.original_inputs(author, prepared, pins)
    assert selected == problems[:3]
    assert "original reference" not in samples.read_text()
    assert "invalid candidate retained" in samples.read_text()


@pytest.mark.parametrize("target", ["source", "completion", "receipt"])
def test_rejects_modified_author_or_prepared_content(tmp_path, target):
    author, prepared, pins, _ = fixture(tmp_path)
    path = {
        "source": author / "human_eval/data.py",
        "completion": prepared / "samples-private.jsonl",
        "receipt": prepared / "receipt-private.json",
    }[target]
    path.write_text(path.read_text() + "tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        runner.original_inputs(author, prepared, pins)


def test_extra_author_module_is_not_silently_imported(tmp_path):
    author, prepared, pins, _ = fixture(tmp_path)
    (author / "human_eval/unreviewed.py").write_text("# inert\n")
    with pytest.raises(ValueError, match="inventory"):
        runner.original_inputs(author, prepared, pins)


@pytest.mark.parametrize("prefix", ["/", "/usr", "/usr/local", "/data", "/root"])
def test_broad_runtime_mount_rejected(prefix):
    with pytest.raises(ValueError, match="dedicated"):
        runner.validate_runtime(prefix)


def test_non_linux_never_falls_back_to_host(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no host fallback"):
        runner.launch(SimpleNamespace(work_root=tmp_path / "not-created"))
    assert not (tmp_path / "not-created").exists()


def test_worker_has_no_remote_transport_or_device_mount_and_checks_before_import():
    worker = (ROOT / "scripts/humaneval_isolated_worker.py").read_text()
    body = worker[worker.index("def evaluate():") :]
    assert body.index('"network reachable"') < body.index("from human_eval.evaluation")
    assert body.index('"capabilities nonempty"') < body.index(
        "from human_eval.evaluation"
    )
    launcher = "".join(
        (ROOT / "scripts/run_humaneval_isolated_scoring.py").read_text().split()
    )
    assert '"--propagation","unchanged"' in launcher
    assert (
        '"--kill-child"'
        in (ROOT / "scripts/run_humaneval_isolated_scoring.py").read_text()
    )
    assert 'bind("/dev"' not in worker
