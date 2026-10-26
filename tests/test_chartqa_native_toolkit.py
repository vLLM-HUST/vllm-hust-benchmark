import importlib.util
import io
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "chartqa_toolkit",
    Path(__file__).parents[1] / "scripts/prepare_chartqa_native_toolkit.py",
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def checkout(tmp_path, monkeypatch, *, revision=None, dirty=""):
    (tmp_path / ".git").mkdir()
    responses = {
        ("rev-parse", "HEAD"): revision or tool.REVISION,
        ("status", "--porcelain", "--untracked-files=all"): dirty,
        ("fsck", "--no-reflogs"): "",
        (
            "ls-files",
        ): "run.py\nvlmeval/dataset/image_vqa.py\nvlmeval/dataset/utils/vqa_eval.py",
        ("rev-parse", "HEAD^{tree}"): "a" * 40,
    }
    monkeypatch.setattr(tool, "git", lambda root, *args: responses[args])


def test_verified_source_is_not_execution(tmp_path, monkeypatch):
    checkout(tmp_path, monkeypatch)
    result = tool.verify(tmp_path)
    assert result["tracked_files"] == 3
    assert result["model_calls"] == 0
    assert result["native_benchmark_verified"] is False


def test_reject_changed_revision(tmp_path, monkeypatch):
    checkout(tmp_path, monkeypatch, revision="b" * 40)
    with pytest.raises(ValueError, match="revision"):
        tool.verify(tmp_path)


def test_reject_changed_or_added_files(tmp_path, monkeypatch):
    checkout(tmp_path, monkeypatch, dirty="?? generated.py")
    with pytest.raises(ValueError, match="untracked"):
        tool.verify(tmp_path)


def test_reject_ancestor_repository(tmp_path):
    with pytest.raises(ValueError, match="dedicated"):
        tool.verify(tmp_path)


@pytest.mark.parametrize("name", ["../escape", ".git/config", "/absolute", "a\\b"])
def test_archive_rejects_unsafe_member(tmp_path, monkeypatch, name):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        item = tarfile.TarInfo(f"VLMEvalKit-{tool.REVISION}/{name}")
        item.size = 1
        archive.addfile(item, io.BytesIO(b"x"))
    payload.seek(0)
    monkeypatch.setattr(tool.urllib.request, "urlopen", lambda *a, **kw: payload)
    output = tmp_path / "source"
    output.mkdir()
    with pytest.raises(ValueError):
        tool.acquire_archive(output)
    assert not (output / ".git").exists()
