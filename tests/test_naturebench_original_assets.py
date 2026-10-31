"""Tiny offline transport/identity fixtures, not original scientific task execution."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "naturebench_assets", ROOT / "scripts/prepare_naturebench_original_assets.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_original_cpu_assets_are_not_native_scientific_delivery():
    directory = ROOT / "docs/workload-data"
    item = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["NATUREBENCH"]
    phase = json.loads((ROOT / item["preparation_receipt"]).read_text("utf-8"))
    assets = phase["cpu_task_asset_preparation"]
    assert (
        assets["original_cpu_tasks"] == 3 and assets["original_files_verified"] == 576
    )
    assert assets["lfs_sha256_files"] + assets["git_blob_files"] == 576
    assert (
        assets["existing_valid_original_files_reused"]
        + assets["two_missing_full_original_objects_verified"]
        == 576
    )
    assert assets["original_bytes_verified"] == 2754532128
    assert assets["original_offline_skip_download_preparation_exit_code"] == 0
    assert (
        not phase["source_checkbox_eligible"] and not assets["source_checkbox_eligible"]
    )
    assert (
        assets["model_calls"] == 0
        and not assets["scientific_task_or_generated_code_execution"]
    )
    assert not phase["clean_preparation_client"][
        "scientific_task_environment_qualified"
    ]
    assert not phase["raw_payloads_in_git_or_local_workstation"]
    if item["checked"]:
        assert item["delivery_receipt"] != item["preparation_receipt"]


def test_complete_original_cpu_file_identity_profile():
    profile = runner.load_profile()
    assert len(profile["files"]) == 576
    assert sum(item["bytes"] for item in profile["files"]) == 2754532128
    assert sum("sha256" in item for item in profile["files"]) == 500
    assert not any("/data_archives/" in item["path"] for item in profile["files"])


def test_pagination_transport_retains_exact_original_path_and_encoded_cursor():
    revision = "a" * 40
    path = f"/api/datasets/FrontisAI/NatureBench/tree/{revision}"
    query = "?recursive=true&cursor=fixture%3D%3D&limit=1000"
    url = "https://huggingface.co" + path + query
    assert runner.pagination_url(url, "https://hf-mirror.com", revision) == (
        "https://hf-mirror.com" + path + query
    )
    assert runner.pagination_url(url, "https://huggingface.co", revision) == url
    for other in (
        "https://example.invalid" + path + query,
        "https://huggingface.co/api/datasets/Other/Repo/tree/" + revision,
        "https://huggingface.co/datasets/FrontisAI/NatureBench/resolve/" + revision,
    ):
        assert runner.pagination_url(other, "https://hf-mirror.com", revision) == other
    with pytest.raises(ValueError, match="revision"):
        runner.pagination_url(url, "https://hf-mirror.com", "b" * 40)
    with pytest.raises(ValueError, match="revision"):
        runner.pagination_url(
            url.replace("https:", "http:"), "https://hf-mirror.com", revision
        )
    with pytest.raises(ValueError, match="endpoint"):
        runner.pagination_url(url, "https://unapproved.invalid", revision)


def test_supported_http_backend_extension_has_no_network_in_fixture(monkeypatch):
    import requests

    calls = []

    def send(self, request, **kwargs):
        calls.append((request.url, kwargs))
        return "fixture"

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    revision = "a" * 40
    url = f"https://huggingface.co/api/datasets/FrontisAI/NatureBench/tree/{revision}?cursor=x"
    session = runner.backend_factory("https://hf-mirror.com", revision)
    request = requests.Request("GET", url).prepare()
    assert session.get_adapter(url).send(request, timeout=None) == "fixture"
    assert calls[0][0] == url.replace("https://huggingface.co", "https://hf-mirror.com")
    assert calls[0][1]["timeout"] == (15, 60)


def tiny_profile(tmp_path):
    files = []
    for number, lfs in enumerate((True, False)):
        name = f"tasks/fixture/problem/file{number}.py"
        raw = b"raise RuntimeError('fixture must never be executed')\n"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = (
            hashlib.sha256(raw).hexdigest()
            if lfs
            else hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
        )
        files.append(
            {"path": name, "bytes": len(raw), "sha256" if lfs else "git_blob": digest}
        )
    return {"files": files}


def test_original_lfs_and_git_blob_bytes_verified_without_executing_code(tmp_path):
    profile = tiny_profile(tmp_path)
    assert runner.verify_files(tmp_path, profile) == 2
    path = tmp_path / profile["files"][1]["path"]
    path.write_bytes(path.read_bytes().replace(b"RuntimeError", b"RuntimeErroX"))
    with pytest.raises(ValueError, match="digest"):
        runner.verify_files(tmp_path, profile)


def test_extra_task_files_rejected(tmp_path):
    profile = tiny_profile(tmp_path)
    (tmp_path / "tasks/fixture/extra.txt").write_text("fixture", "utf-8")
    with pytest.raises(ValueError, match="additional"):
        runner.verify_files(tmp_path, profile)


def test_complete_http_object_range_and_digest_not_content_sampling(
    tmp_path, monkeypatch
):
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    raw = b"complete original fixture object\n"
    item = {
        "path": "tasks/fixture/problem/original.csv",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    calls = []

    def get(request, timeout):
        calls.append((request.full_url, request.get_header("Range"), timeout))
        return io.BytesIO(raw)

    monkeypatch.setattr(runner, "urlopen", get)
    runner.fetch_publisher_file(
        data,
        cache,
        item,
        {"repo_id": "FrontisAI/NatureBench", "revision": "a" * 40},
        "https://hf-mirror.com",
    )
    assert (data / item["path"]).read_bytes() == raw
    assert calls[0][0].endswith("original.csv?download=true")
    assert calls[0][1] == f"bytes=0-{len(raw) - 1}"
    assert runner.verify_files(data, {"files": [item]}) == 1


def test_bad_original_object_digest_never_published_as_original_file(
    tmp_path, monkeypatch
):
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    cache.mkdir()
    item = {
        "path": "tasks/fixture/problem/original.csv",
        "bytes": 3,
        "sha256": hashlib.sha256(b"old").hexdigest(),
    }
    monkeypatch.setattr(runner, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="digest"):
        runner.fetch_publisher_file(
            data,
            cache,
            item,
            {"repo_id": "FrontisAI/NatureBench", "revision": "a" * 40},
            "https://hf-mirror.com",
        )
    assert not (data / item["path"]).exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_resume_reuses_only_verified_original_files(tmp_path, monkeypatch, corrupt):
    author, data, cache = (tmp_path / name for name in ("author", "data", "cache"))
    for path in (author, data, cache):
        path.mkdir()
    profile = tiny_profile(data)
    (data / profile["files"][1]["path"]).unlink()
    if corrupt:
        path = data / profile["files"][0]["path"]
        path.write_bytes(path.read_bytes().replace(b"RuntimeError", b"RuntimeErroX"))
    monkeypatch.setattr(runner, "verify_source", lambda *args: author)
    monkeypatch.setattr(runner.metadata, "version", lambda name: "0.36.0")
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=101 * 1024**3)
    )
    calls = []
    monkeypatch.setattr(
        runner, "_download_original", lambda *args: calls.append(args) or 576
    )
    if corrupt:
        with pytest.raises(ValueError, match="digest"):
            runner.download(
                author, data, cache, profile, "https://hf-mirror.com", resume=True
            )
        assert not calls
    else:
        assert (
            runner.download(
                author, data, cache, profile, "https://hf-mirror.com", resume=True
            )
            == 576
        )
        assert (
            len(calls) == 1 and runner.verify_files(data, profile, complete=False) == 1
        )


def test_publisher_resume_downloads_only_missing_original_objects(
    tmp_path, monkeypatch
):
    author, data, cache = (tmp_path / name for name in ("author", "data", "cache"))
    for path in (author, data, cache):
        path.mkdir()
    profile = tiny_profile(data)
    missing = data / profile["files"][1]["path"]
    raw = missing.read_bytes()
    missing.unlink()
    existing = data / profile["files"][0]["path"]
    before = existing.stat().st_mtime_ns
    monkeypatch.setattr(runner, "verify_source", lambda *args: author)
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=101 * 1024**3)
    )
    calls = []

    def fetch(data_root, cache_root, item, source, endpoint):
        calls.append(item["path"])
        (data_root / item["path"]).write_bytes(raw)

    monkeypatch.setattr(runner, "fetch_publisher_file", fetch)
    assert (
        runner.download(
            author,
            data,
            cache,
            profile,
            "https://hf-mirror.com",
            resume=True,
            transport="publisher-files",
        )
        == 2
    )
    assert calls == [profile["files"][1]["path"]]
    assert existing.stat().st_mtime_ns == before


def test_default_plan_is_offline_and_creates_nothing(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_naturebench_original_assets.py"),
            "--data-root",
            str(tmp_path / "absent"),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "OFFLINE_ORIGINAL_CPU_ASSET_PLAN"
    assert receipt["files"] == 576 and receipt["model_calls"] == 0
    assert not receipt["source_checkbox_eligible"]
    assert not (tmp_path / "absent").exists()


def test_download_uses_unchanged_author_cli_and_restores_client_configuration(
    tmp_path, monkeypatch
):
    author = tmp_path / "author"
    author.mkdir()
    data, cache = tmp_path / "data", tmp_path / "cache"
    monkeypatch.setattr(runner, "verify_source", lambda *args: author)
    monkeypatch.setattr(runner.metadata, "version", lambda name: "0.36.0")
    monkeypatch.setattr(
        runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=101 * 1024**3)
    )
    configurations = []
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            configure_http_backend=lambda factory=None: configurations.append(factory)
        ),
    )
    monkeypatch.setenv("HF_ENDPOINT", "fixture-old-endpoint")
    monkeypatch.setenv("HF_TOKEN", "fixture-do-not-send")
    calls = []

    def run(path, run_name):
        calls.append(list(sys.argv))
        assert runner.os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert "HF_TOKEN" not in runner.os.environ
        assert Path.cwd() == author and run_name == "__main__"

    monkeypatch.setattr(runner.runpy, "run_path", run)
    monkeypatch.setattr(runner, "verify_files", lambda *args: 576)
    argv, directory = sys.argv, Path.cwd()
    profile = {"repo_id": "FrontisAI/NatureBench", "revision": "a" * 40}
    assert runner.download(author, data, cache, profile, "https://hf-mirror.com") == 576
    assert calls[0][-1] == "--download-only"
    assert calls[0][calls[0].index("--tasks") + 1] == "cpu"
    assert "--skip-judge" not in calls[0] and "--agent" not in calls[0]
    assert callable(configurations[0]) and configurations[-1] is None
    assert runner.os.environ["HF_ENDPOINT"] == "fixture-old-endpoint"
    assert runner.os.environ["HF_TOKEN"] == "fixture-do-not-send"
    assert sys.argv is argv and Path.cwd() == directory
