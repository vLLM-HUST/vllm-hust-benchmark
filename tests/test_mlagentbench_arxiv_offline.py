"""Offline transport/source fixtures, not a real scientific task run."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ml_graph_offline", ROOT / "scripts/prepare_mlagentbench_arxiv_offline.py"
)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


def test_plan_does_not_load_torch_or_assets():
    assert MODULE.prepare(SimpleNamespace(execute=False))["model_calls"] == 0


def test_relative_original_split_read_stays_inside_original_working_directory():
    tree = ast.parse(
        (ROOT / "scripts/prepare_mlagentbench_arxiv_offline.py").read_text()
    )
    body = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )
    guarded = next(node for node in body.body if isinstance(node, ast.Try))
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_idx_split"
        for stmt in guarded.body
        for node in ast.walk(stmt)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_idx_split"
        for stmt in guarded.finalbody
        for node in ast.walk(stmt)
    )


def test_original_transport_copies_only_to_expected_owned_cache(tmp_path):
    archive = tmp_path / "original.zip"
    archive.write_bytes(b"tiny fixture")
    networks = tmp_path / "networks"
    networks.mkdir()
    confirm, download = MODULE.offline_transport(
        archive, networks, "http://official.example/arxiv.zip"
    )
    assert confirm("http://official.example/arxiv.zip") is True
    copied = Path(download("http://official.example/arxiv.zip", str(networks)))
    assert copied.read_bytes() == archive.read_bytes()
    with pytest.raises(FileExistsError):
        download("http://official.example/arxiv.zip", str(networks))
    with pytest.raises(ValueError):
        confirm("http://other.example/arxiv.zip")
    with pytest.raises(ValueError):
        download("http://official.example/arxiv.zip", str(tmp_path))


def test_pinned_original_task_software_not_silently_modified(tmp_path, monkeypatch):
    task = tmp_path / "task"
    (task / "scripts").mkdir(parents=True)
    original = task / "scripts/prepare.py"
    original.write_bytes(b"# fixture only\n")
    pins = tmp_path / "pins.json"
    pins.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": MODULE.PREFIX + "scripts/prepare.py",
                        "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(MODULE, "PINS", pins)
    assert MODULE.verify_task(task) == 1
    original.write_bytes(b"# changed fixture\n")
    with pytest.raises(ValueError, match="differs"):
        MODULE.verify_task(task)
