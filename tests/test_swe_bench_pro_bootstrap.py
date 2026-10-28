import importlib.util
import hashlib
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "swe_bootstrap", ROOT / "scripts/bootstrap_swe_bench_pro.py"
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_model_probe_checks_served_identity_without_model_inference(monkeypatch):
    seen = []

    def request(value, timeout):
        seen.append(value)
        return io.BytesIO(
            json.dumps({"data": [{"id": "example", "max_model_len": 262144}]}).encode()
        )

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", request)
    result = bootstrap.model_check("https://model.example/v1", "openai/example", None)
    assert result == {"model_name": "openai/example", "max_model_len": 262144}
    assert seen[0].full_url == "https://model.example/v1/models"
    with pytest.raises(ValueError, match="absent"):
        bootstrap.model_check("https://model.example/v1", "openai/other", None)


@pytest.mark.parametrize(
    "url",
    [
        "https://secret@model.example/v1",
        "https://model.example/v1?key=secret",
        "file:///v1",
        "https://model.example",
    ],
)
def test_refuses_credentials_or_ambiguous_model_url_before_network(url, monkeypatch):
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda *a, **k: pytest.fail("network must not run"),
    )
    with pytest.raises(ValueError):
        bootstrap.model_check(url, "openai/example", None)


def test_bad_download_is_not_promoted_and_cannot_overwrite_existing_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(b"wrong release"),
    )
    destination = tmp_path / "crane.tar.gz"
    with pytest.raises(ValueError, match="SHA256"):
        bootstrap.download(
            "https://release.example/arm64.tar.gz", destination, "a" * 64
        )
    assert not destination.exists()
    assert destination.with_suffix(".gz.partial").exists()
    destination.write_bytes(b"other user file")
    with pytest.raises(ValueError, match="already exists"):
        bootstrap.download(
            "https://release.example/arm64.tar.gz", destination, "a" * 64
        )
    assert destination.read_bytes() == b"other user file"


@pytest.mark.parametrize("payload", [b"truncated", b"wrong source bytes!"])
def test_source_download_verifies_git_identity_before_promoting(
    tmp_path, monkeypatch, payload
):
    expected = b"frozen source bytes"
    digest = hashlib.sha1(b"blob 19\0" + expected).hexdigest()
    monkeypatch.setattr(
        bootstrap.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload)
    )
    path = tmp_path / "source.json"
    with pytest.raises(ValueError, match="Git blob SHA"):
        bootstrap.download(
            "https://source.example/file",
            path,
            git_blob={"size": len(expected), "sha": digest},
        )
    assert not path.exists()
    assert path.with_suffix(".json.partial").read_bytes() == payload


def test_missing_auth_fails_without_request(monkeypatch):
    monkeypatch.delenv("BOOTSTRAP_TEST_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BOOTSTRAP_TEST_API_KEY"):
        bootstrap.model_check(
            "https://model.example/v1", "openai/example", "BOOTSTRAP_TEST_API_KEY"
        )


def test_refuses_foreign_work_directory_before_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        bootstrap.sys,
        "argv",
        [
            "bootstrap",
            "--work",
            str(tmp_path),
            "--docker",
            "/example/docker",
            "--model-api",
            "http://model.example/v1",
            "--model-name",
            "openai/example",
            "--resume",
        ],
    )
    marker = tmp_path / "keep.txt"
    marker.write_text("unrelated")
    with pytest.raises(SystemExit):
        bootstrap.main()
    assert marker.read_text() == "unrelated"
    assert not (tmp_path / "bootstrap-state.json").exists()


@pytest.mark.parametrize("truncate_once", [False, True])
def test_standalone_installer_pins_source_and_refuses_changed_files(
    tmp_path, monkeypatch, truncate_once
):
    monkeypatch.setattr(bootstrap, "LOCK", tmp_path / "absent-lock.json")
    seen = []
    revision = "a" * 40
    content = b"frozen source bytes"
    blob_sha = hashlib.sha1(b"blob 19\0" + content).hexdigest()
    directories = {
        "": ["README.md", "pyproject.toml", "src/", "docs/", "scripts/"],
        "src": ["vllm_hust_benchmark/"],
        "src/vllm_hust_benchmark": [
            "__init__.py",
            "_version.py",
            "swe_bench_pro.py",
            "swe_bench_pro_inputs.py",
            "swe_bench_pro_timing.py",
            "data/",
        ],
        "src/vllm_hust_benchmark/data": [
            "swe_bench_pro_v1_lock.json",
            "swe_bench_pro_bootstrap_lock.json",
        ],
        "docs": ["examples/"],
        "docs/examples": ["swe-bench-pro/"],
        "docs/examples/swe-bench-pro": ["client-requirements.txt", "single-agent.json"],
        "scripts": ["prepare_swe_bench_pro_environment.py", "run_swe_bench_pro.py"],
    }
    tree_ids = {key: hashlib.sha1(key.encode()).hexdigest() for key in directories}
    tree_ids[""] = revision
    trees = {}
    for directory, entries in directories.items():
        tree = []
        for name in entries:
            if name.endswith("/"):
                child = (directory + "/" + name.rstrip("/")).lstrip("/")
                tree.append(
                    {"path": name.rstrip("/"), "type": "tree", "sha": tree_ids[child]}
                )
            else:
                tree.append(
                    {
                        "path": name,
                        "type": "blob",
                        "sha": blob_sha,
                        "size": len(content),
                    }
                )
        trees[tree_ids[directory]] = {"tree": tree, "truncated": False}

    def fetch(request, timeout):
        seen.append(request.full_url)
        if "/commits/" in request.full_url:
            return io.BytesIO(json.dumps({"sha": revision}).encode())
        if "/git/trees/" in request.full_url:
            return io.BytesIO(
                json.dumps(trees[request.full_url.rsplit("/", 1)[-1]]).encode()
            )
        assert request.full_url.endswith("?ref=" + revision)
        if truncate_once and "/contents/README.md?" in request.full_url:
            if seen.count(request.full_url) == 1:
                return io.BytesIO(b"truncated")
        return io.BytesIO(b"frozen source bytes")

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", fetch)
    target = tmp_path / "fresh-source"
    args = ["--source-revision", "main", "--source-directory", str(target)]
    assert bootstrap.source_repository(args) == target
    assert len([url for url in seen if "/contents/" in url]) == 13 + int(truncate_once)
    assert (
        json.loads((target / "source-installation.json").read_text())["commit"]
        == revision
    )
    seen.clear()
    bootstrap.source_repository(args)
    assert not any("/contents/" in url for url in seen)
    (target / "src/vllm_hust_benchmark/swe_bench_pro.py").write_text("changed")
    with pytest.raises(ValueError, match="installed source changed"):
        bootstrap.source_repository(args)
    # A hash recorded from a bad download is not proof of upstream identity.
    installed = target / "src/vllm_hust_benchmark/swe_bench_pro.py"
    receipt_path = target / "source-installation.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["files"]["src/vllm_hust_benchmark/swe_bench_pro.py"] = bootstrap.sha256(
        installed
    )
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Git blob SHA"):
        bootstrap.source_repository(args)


def test_wrong_registry_image_fails_before_client_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        bootstrap, "model_check", lambda *a: {"model_name": "openai/example"}
    )
    dataset = tmp_path / "authorized.parquet"
    dataset.write_bytes(b"test V1 input")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"dataset": {"parquet_sha256": bootstrap.sha256(dataset)}})
    )
    monkeypatch.setattr(bootstrap, "LOCK", lock)

    def check(command, **kwargs):
        if "info" in command:
            return json.dumps(
                {"OSType": "linux", "Architecture": "aarch64", "Images": 0}
            )
        assert "digest" in command
        return "sha256:" + "b" * 64

    monkeypatch.setattr(bootstrap.subprocess, "check_output", check)
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *a, **k: pytest.fail("client installation must not start"),
    )
    work = tmp_path / "fresh-work"
    monkeypatch.setattr(
        bootstrap.sys,
        "argv",
        [
            "bootstrap",
            "--work",
            str(work),
            "--docker",
            "/example/docker",
            "--crane",
            "/example/crane",
            "--dataset",
            str(dataset),
            "--model-api",
            "http://model.example/v1",
            "--model-name",
            "openai/example",
        ],
    )
    with pytest.raises(ValueError, match="frozen upstream digest"):
        bootstrap.main()
    assert not (work / "client").exists()
    assert not (work / "sources").exists()
    assert json.loads((work / "bootstrap-state.json").read_text())["status"] == "FAILED"
