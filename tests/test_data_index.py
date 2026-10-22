"""Navigation is deterministic, path-only, and never promotes evidence validity."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "data_index", Path(__file__).resolve().parents[1] / "scripts/data_index.py"
)
index = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(index)


def test_classification_and_exclusions():
    assert (
        index.classify("archive/mod-git-20261022/example/snapshot/x.json")["collection"]
        == "archive/mod-git-20261022/example"
    )
    assert (
        index.classify("reports/vllm-hust-midterm-20261019/m01/x.json")["collection"]
        == "reports/vllm-hust-midterm-20261019/m01"
    )
    assert (
        index.classify("tests/fixtures/x.json")["category"] == "synthetic-test-fixtures"
    )
    assert index.classify("submissions/run/x.json")["write_policy"] == "append-new-run"
    assert index.classify(index.OUTPUT) is None
    assert index.classify("README.md") is None
    for path in ("/archive/x", "archive/../secret"):
        with pytest.raises(ValueError):
            index.classify(path)


def test_deterministic_counts_and_entrypoint_truncation():
    paths = [f"reports/example/{i}/manifest.json" for i in range(15)]
    paths += ["reports/example/README.md", "reports/example/结果.log"]
    result = index.build(paths)
    assert result == index.build(list(reversed(paths)) + paths)
    group = result["collections"][0]
    assert result["file_count"] == 17
    assert group["entrypoint_count"] == 16
    assert group["entrypoints_truncated"] is True
    assert len(group["entrypoints"]) == 12
    assert group["entrypoints"][0] == "reports/example/README.md"
    assert group["formats"] == {".json": 15, ".log": 1, ".md": 1}


@pytest.fixture
def repository(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "reports/run").mkdir(parents=True)
    # Invalid JSON remains opaque: navigation must not parse evidence.
    (tmp_path / "reports/run/M01.json").write_bytes(b"not valid JSON\x00")
    (tmp_path / "reports/run/untracked.log").write_text("not indexed")
    subprocess.run(["git", "add", "reports/run/M01.json"], cwd=tmp_path, check=True)
    monkeypatch.setattr(index, "ROOT", tmp_path)
    return tmp_path


def test_write_check_and_staged_paths(repository, capsys):
    original = (repository / "reports/run/M01.json").read_bytes()
    assert index.main(["--check"]) == 1
    assert index.main(["--write"]) == 0
    assert index.main(["--check"]) == 0
    assert index.tracked_paths(repository) == ["reports/run/M01.json"]
    subprocess.run(["git", "add", index.OUTPUT], cwd=repository, check=True)
    assert index.main(["--check"]) == 0  # No recursive self-counting.
    subprocess.run(
        ["git", "add", "reports/run/untracked.log"], cwd=repository, check=True
    )
    assert index.main(["--check"]) == 1
    assert (repository / "reports/run/M01.json").read_bytes() == original


def test_query_and_limits(repository, capsys):
    assert index.main(["--query", "M01 RUN", "--files", "--limit", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["matched"] == 1
    assert result["results"][0]["path"] == "reports/run/M01.json"
    assert result["truncated"] is False
    index.main(["--query", "untracked"])
    assert json.loads(capsys.readouterr().out)["matched"] == 0
    for args in (["--limit", "0"], ["--write", "--files"], ["--check", "--query", "x"]):
        with pytest.raises(SystemExit):
            index.main(args)


def test_guide_local_links_exist():
    import re

    root = Path(__file__).resolve().parents[1]
    for target in re.findall(r"\]\(([^)]+)\)", (root / "DATA_INDEX.md").read_text()):
        if "://" not in target:
            assert (root / target.split("#", 1)[0]).exists(), target
