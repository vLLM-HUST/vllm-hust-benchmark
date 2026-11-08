"""Inert transport fixtures; not original Agent, reproduction or grading evidence."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "paper_assets", ROOT / "scripts/prepare_paperbench_original_assets.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def identity(raw, *, lfs=False):
    sha = hashlib.sha256(raw).hexdigest()
    value = raw
    item = {
        "path": "project/paperbench/data/papers/fixture/paper.md",
        "bytes": len(raw),
        "sha256": sha,
    }
    if lfs:
        value = f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {len(raw)}\n".encode()
        item["git_lfs"] = {"pointer_bytes": len(value), "oid_sha256": sha}
    item["git_blob"] = hashlib.sha1(
        b"blob " + str(len(value)).encode() + b"\0" + value
    ).hexdigest()
    return item


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@pytest.mark.parametrize("lfs", [False, True])
def test_original_payload_publish_and_reuse(tmp_path, monkeypatch, lfs):
    raw = b"inert software test bytes"
    item = identity(raw, lfs=lfs)
    path = tmp_path / "published"
    calls = []

    def open_fixture(request, timeout):
        calls.append(request.full_url)
        assert timeout == 90
        return Response(raw)

    monkeypatch.setattr(runner, "urlopen", open_fixture)
    runner.fetch_original_payload(path, item, "https://fixture.invalid/original")
    assert path.read_bytes() == raw and len(calls) == 1
    assert not path.with_name("published.partial").exists()
    runner.prepare(
        tmp_path, {"files": [item], "repository": "fixture", "revision": "fixed"}
    )
    assert len(calls) == 2  # Different pinned relative target, not another source.
    runner.prepare(
        tmp_path, {"files": [item], "repository": "fixture", "revision": "fixed"}
    )
    assert len(calls) == 2


@pytest.mark.parametrize("payload", [b"short", b"unexpected extra payload"])
def test_failed_payload_is_not_published(tmp_path, monkeypatch, payload):
    item = identity(b"original bytes")
    monkeypatch.setattr(runner, "urlopen", lambda *a, **k: Response(payload))
    path = tmp_path / "asset"
    with pytest.raises(ValueError):
        runner.fetch_original_payload(path, item, "https://fixture.invalid/original")
    assert not path.exists() and path.with_name("asset.partial").exists()


@pytest.mark.parametrize("existing", ["asset", "asset.partial"])
def test_existing_final_or_partial_never_overwritten(tmp_path, monkeypatch, existing):
    path = tmp_path / "asset"
    (tmp_path / existing).write_bytes(b"retained")
    monkeypatch.setattr(
        runner, "urlopen", lambda *a, **k: pytest.fail("must not request")
    )
    with pytest.raises(ValueError, match="never overwrite"):
        runner.fetch_original_payload(
            path, identity(b"original"), "https://fixture.invalid/original"
        )
    assert (tmp_path / existing).read_bytes() == b"retained"


def test_same_size_corruption_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "urlopen", lambda *a, **k: Response(b"wrong"))
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.fetch_original_payload(
            tmp_path / "asset",
            identity(b"right", lfs=True),
            "https://fixture.invalid/original",
        )
    assert not (tmp_path / "asset").exists()


@pytest.mark.parametrize(
    "bad",
    [
        "../escape",
        "project/paperbench/data/papers/../escape",
        "C:\\escape",
        "/absolute",
        "project/paperbench/data/papers//duplicate",
    ],
)
def test_pinned_paths_fail_before_io(tmp_path, bad):
    pins = json.loads(runner.PINS.read_text("utf-8"))
    pins["files"][0]["path"] = bad
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(pins), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_pins(path)


def test_lfs_oid_change_rejected_in_plan(tmp_path):
    pins = copy.deepcopy(runner.load_pins())
    item = next(x for x in pins["files"] if "git_lfs" in x)
    item["git_lfs"]["oid_sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(pins), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_pins(path)


def test_offline_plan_no_output_or_native_completion(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_paperbench_original_assets.py"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 506 and report["bytes"] == 213434104
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not report["source_checkbox_eligible"] and not output.exists()


def test_budget_rejected_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--output",
            str(tmp_path / "none"),
            "--action",
            "prepare",
            "--max-bytes",
            "1",
        ],
    )
    monkeypatch.setattr(
        runner, "prepare", lambda *a, **k: pytest.fail("budget must fail first")
    )
    with pytest.raises(ValueError, match="byte budget"):
        runner.main()


def test_github_api_applies_only_to_regular_git_files(tmp_path, monkeypatch):
    lfs = identity(b"fixture lfs", lfs=True)
    regular = identity(b"fixture regular")
    regular["path"] = regular["path"].replace("paper.md", "rubric.json")
    fetched = []
    api = []
    monkeypatch.setattr(
        runner, "fetch_original_payload", lambda path, item, url: fetched.append(url)
    )
    monkeypatch.setattr(runner, "acquire", lambda path, item: api.append(item["url"]))
    runner.prepare(
        tmp_path,
        {
            "repository": "openai/frontier-evals",
            "revision": runner.REVISION,
            "files": [lfs, regular],
        },
        github_api=True,
    )
    assert fetched == [
        f"https://media.githubusercontent.com/media/openai/frontier-evals/{runner.REVISION}/{lfs['path']}"
    ]
    assert api == [
        f"https://api.github.com/repos/openai/frontier-evals/git/blobs/{regular['git_blob']}"
    ]
