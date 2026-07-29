from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_snapshot_consistency.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_snapshot_consistency", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE_DATA = [{"entry_id": "single-1", "engine": "vllm-hust"}]
MULTI_DATA: list[dict] = []


def write_local_snapshots(tmp_path: Path) -> Path:
    snapshot_dir = tmp_path / "leaderboard-data" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "leaderboard_single.json").write_text(
        json.dumps(SINGLE_DATA), encoding="utf-8"
    )
    (snapshot_dir / "leaderboard_multi.json").write_text(
        json.dumps(MULTI_DATA), encoding="utf-8"
    )
    return snapshot_dir


def write_website_snapshots(website_repo: Path) -> Path:
    data_dir = website_repo / "public" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "leaderboard_single.json").write_text(
        json.dumps(SINGLE_DATA), encoding="utf-8"
    )
    (data_dir / "leaderboard_multi.json").write_text(
        json.dumps(MULTI_DATA), encoding="utf-8"
    )
    return data_dir


def install_fake_hf(monkeypatch, download_fn) -> None:
    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = download_fn
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)


def test_three_repos_in_sync(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    write_local_snapshots(tmp_path)
    website_repo = tmp_path / "website"
    write_website_snapshots(website_repo)

    def fake_download(repo_id, filename, repo_type, token):
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / filename
        data = SINGLE_DATA if filename == "leaderboard_single.json" else MULTI_DATA
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    install_fake_hf(monkeypatch, fake_download)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_snapshot_consistency.py",
            "--benchmark-repo",
            str(tmp_path),
            "--hf-repo-id",
            "test/repo",
            "--website-repo",
            str(website_repo),
        ],
    )
    assert module.main() == 0
    captured = capsys.readouterr()
    assert "all repos in sync" in captured.out
    assert captured.err == ""


def test_hf_mismatch_returns_error(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    write_local_snapshots(tmp_path)

    def fake_download(repo_id, filename, repo_type, token):
        cache_dir = tmp_path / "hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / filename
        # Return different content for every file.
        path.write_text(json.dumps([{"entry_id": "different"}]), encoding="utf-8")
        return str(path)

    install_fake_hf(monkeypatch, fake_download)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_snapshot_consistency.py",
            "--benchmark-repo",
            str(tmp_path),
            "--hf-repo-id",
            "test/repo",
            "--skip-website",
        ],
    )
    assert module.main() == 1
    captured = capsys.readouterr()
    assert "checksum mismatch" in captured.err
    assert "all repos in sync" not in captured.out


def test_website_repo_not_found_warns(tmp_path: Path, monkeypatch, capsys) -> None:
    module = load_module()
    write_local_snapshots(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_snapshot_consistency.py",
            "--benchmark-repo",
            str(tmp_path),
            "--website-repo",
            str(tmp_path / "nonexistent-website"),
            "--hf-repo-id",
            "none",
        ],
    )
    assert module.main() == 0
    captured = capsys.readouterr()
    assert "all repos in sync" in captured.out
    assert "website data directory not found" in captured.err


def test_hf_download_failure_warns_and_skips(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = load_module()
    write_local_snapshots(tmp_path)

    def failing_download(repo_id, filename, repo_type, token):
        raise RuntimeError("network unavailable")

    install_fake_hf(monkeypatch, failing_download)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_snapshot_consistency.py",
            "--benchmark-repo",
            str(tmp_path),
            "--hf-repo-id",
            "test/repo",
            "--skip-website",
        ],
    )
    assert module.main() == 0
    captured = capsys.readouterr()
    assert "all repos in sync" in captured.out
    assert "failed to download" in captured.err
    assert "network unavailable" in captured.err
