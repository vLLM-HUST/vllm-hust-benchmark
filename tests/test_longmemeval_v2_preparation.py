"""Data preparation fixtures, never native benchmark or screenshot evidence."""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "lme_v2_preparation", ROOT / "scripts/prepare_longmemeval_v2_original.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_distinct_original_protocol_and_complete_author_source(runner):
    source = json.loads(runner.SOURCE_PINS.read_text("utf-8"))
    data = json.loads(runner.DATA_PINS.read_text("utf-8"))
    assert source["source_id"] == data["source_id"] == "LONGMEMEVAL_V2"
    assert len(source["files"]) == 53
    assert len(data["files"]) == 41
    assert sum(item["bytes"] for item in data["files"]) == 7120369667
    assert {
        "data/download_data.py",
        "data/prepare_data.py",
        "data/validate_data.py",
        "evaluation/run_eval.py",
    } <= {f["path"] for f in source["files"]}
    assert {
        "questions.jsonl",
        "trajectories.jsonl",
        "haystacks/lme_v2_small.json",
        "haystacks/lme_v2_medium.json",
    } <= {f["path"] for f in data["files"]}
    assert all(
        len(f["sha256"]) == 64 and len(f["git_blob"]) == 40 for f in source["files"]
    )


def test_official_reader_model_is_complete_not_text_only_weight_subset():
    pins = json.loads(
        (ROOT / "configs/workload-native/longmemeval-v2-reader-pins.json").read_text(
            "utf-8"
        )
    )
    assert pins["source_id"] == "LONGMEMEVAL_V2_READER"
    assert pins["model"]["repo_id"] == "Qwen/Qwen3.5-9B"
    files = pins["model"]["files"]
    assert len(files) == 16
    assert sum(f["bytes"] for f in files) == 19329393661
    assert len([f for f in files if f["path"].endswith(".safetensors")]) == 4
    assert {
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "config.json",
    } <= {f["path"] for f in files}
    assert all(len(f["sha256"]) == 64 for f in files)


def test_plan_does_not_create_directories_or_claim_native_completion(runner, tmp_path):
    args = argparse.Namespace(
        action="plan",
        author_root=tmp_path / "source",
        data_root=tmp_path / "data",
        output=tmp_path / "out",
        max_bytes=None,
    )
    result = runner.run(args)
    assert not list(tmp_path.iterdir())
    assert result["commands"] == []
    assert result["model_calls"] == 0
    assert (
        not result["native_benchmark_verified"]
        and not result["source_checkbox_eligible"]
    )


def test_original_preparation_receipt_is_not_native_or_consumer_completion():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/longmemeval-v2-original-readiness-20261026.json"
        ).read_text("utf-8")
    )
    assert receipt["original_data"]["files"] == 41
    assert receipt["unchanged_author_commands"]["check_screenshots_both_tiers"] is True
    assert receipt["unchanged_author_commands"]["validate_small_exit"] == 0
    assert receipt["unchanged_author_commands"]["validate_medium_exit"] == 0
    assert receipt["runtime"]["native_methods_executed"] is False
    assert receipt["runtime"]["public_provider_and_research_handoff_verified"] is False
    assert receipt["source_checkbox_eligible"] is False


@pytest.mark.parametrize("kind", ["sha256", "git_blob"])
def test_original_file_identity_and_same_size_corruption(runner, tmp_path, kind):
    raw = b"original fixture only"
    digest = (
        hashlib.sha256(raw).hexdigest()
        if kind == "sha256"
        else hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    )
    (tmp_path / "original").write_bytes(raw)
    pins = {"files": [{"path": "original", "bytes": len(raw), kind: digest}]}
    runner.verify_assets(tmp_path, pins)
    (tmp_path / "original").write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError, match="identity differs"):
        runner.verify_assets(tmp_path, pins)


@pytest.mark.parametrize(
    "name,kind", [("../escape", "file"), ("/absolute", "file"), ("link", "symlink")]
)
def test_original_archive_never_extracts_unsafe_members(runner, tmp_path, name, kind):
    folder = tmp_path / "trajectory_screenshots"
    folder.mkdir()
    with tarfile.open(folder / "web_screenshots.tar.gz", "w:gz") as archive:
        item = tarfile.TarInfo(name)
        if kind == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = "outside"
            archive.addfile(item)
        else:
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe original archive"):
        runner.archive_capacity(tmp_path)
    assert not (tmp_path / "escape").exists()


def test_prepare_runs_both_original_validators_without_skip_flags(
    runner, tmp_path, monkeypatch
):
    source, data, output = (tmp_path / name for name in ("author", "data", "out"))
    source.mkdir()
    data.mkdir()
    monkeypatch.setattr(runner, "verify_source", lambda *_: source)
    monkeypatch.setattr(runner, "verify_assets", lambda *_: None)
    monkeypatch.setattr(runner, "archive_capacity", lambda *_: 3)
    calls = []

    def fake(command, **kwargs):
        assert kwargs["cwd"] == source
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake)
    result = runner.run(
        argparse.Namespace(
            action="prepare",
            author_root=source,
            data_root=data,
            output=output,
            max_bytes=None,
        )
    )
    assert len(calls) == 3
    assert calls[0][2:] == ["--data-root", str(data), "--mode", "symlink"]
    assert [command[-2:] for command in calls[1:]] == [
        ["--tier", "small"],
        ["--tier", "medium"],
    ]
    assert result["both_original_tiers_validated"] is True
    assert result["native_benchmark_verified"] is False


def test_download_refuses_inadequate_budget_before_subprocess(
    runner, tmp_path, monkeypatch
):
    source = tmp_path / "author"
    source.mkdir()
    monkeypatch.setattr(runner, "verify_source", lambda *_: source)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("no download allowed"),
    )
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="explicit original-byte budget"):
        runner.run(
            argparse.Namespace(
                action="download",
                author_root=source,
                data_root=tmp_path / "data",
                output=output,
                max_bytes=1,
            )
        )
    receipt = json.loads((output / "receipt-public.json").read_text("utf-8"))
    assert receipt["status"] == "FAILED_NOT_COMPLETE" and receipt["commands"] == []


@pytest.mark.parametrize("download_query", [False, True])
def test_model_transport_download_query_is_opt_in_and_preserves_identity(
    tmp_path, monkeypatch, download_query
):
    spec = importlib.util.spec_from_file_location(
        "lme_v2_asset_transport", ROOT / "scripts/prepare_xsum_original_assets.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    pins = tmp_path / "pins.json"
    raw = b"software fixture only"
    pins.write_text(
        json.dumps(
            {
                "source_id": "SOFTWARE_FIXTURE",
                "model": {
                    "repo_id": "fixture/model",
                    "revision": "fixed-fixture",
                    "files": [
                        {
                            "path": "weights.fixture",
                            "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest(),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "external"
    monkeypatch.setattr("sys.argv", ["preparer", "--output", str(output)])
    seen = []

    def fetch(path, asset, url):
        seen.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        helper.verify(path, asset)

    monkeypatch.setattr(helper, "fetch", fetch)
    helper.main(
        pins_path=pins,
        source_id="SOFTWARE_FIXTURE",
        parts=("model",),
        hf_download_query=download_query,
    )
    assert seen == [
        "https://huggingface.co/fixture/model/resolve/fixed-fixture/weights.fixture"
        + ("?download=true" if download_query else "")
    ]
