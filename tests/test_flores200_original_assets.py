"""Small synthetic software fixtures only; no real dataset or native task run."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

spec = importlib.util.spec_from_file_location(
    "flores_assets",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_flores200_original_assets.py",
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def archive(path, entries):
    with tarfile.open(path, "w:gz") as bundle:
        for name, raw in entries:
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            bundle.addfile(item, io.BytesIO(raw))


def fixture_entries():
    return [
        (f"./flores200_dataset/{split}/{lang}.{split}", "  测试文本  \n".encode())
        for split in ("dev", "devtest")
        for lang in ("eng_Latn", "zho_Hans")
    ]


def fixture_dataset():
    return {"split_rows": {"dev": 1, "devtest": 1}, "language_code_count": 2}


def test_pins_keep_original_archive_and_model_language_gaps_explicit():
    pins = json.loads(assets.PINS.read_text("utf-8"))
    assert pins["source_id"] == "FLORES200"
    assert pins["dataset"]["url"].startswith("https://dl.fbaipublicfiles.com/")
    assert pins["dataset"]["language_code_count"] == 204
    assert pins["dataset"]["split_rows"] == {"dev": 997, "devtest": 1012}
    assert "not_separately_published" in pins["dataset"]["checksum_assurance"]
    assert not pins["dataset"]["hidden_test_available"]
    assert len(pins["model"]["files"]) == 8
    assert pins["model"]["noncommercial_research_only"]
    assert pins["model"]["original_codes_without_language_token"] == [
        "arb_Latn",
        "min_Arab",
        "sat_Olck",
    ]
    selected = assets.jobs(pins, ["data", "model", "data"], "https://hf-mirror.com")
    assert len(selected) == 9
    assert selected[0][2] == pins["dataset"]["url"]
    assert all(
        "hf-mirror.com/facebook/nllb-200-distilled-600M/resolve/" in url
        for _, _, url in selected[1:]
    )


def test_archive_inspection_preserves_parallel_sets_without_extraction(tmp_path):
    path = tmp_path / "fixture.tar.gz"
    archive(path, fixture_entries())
    before = path.read_bytes()
    result = assets.inspect_archive(path, fixture_dataset())
    assert result["regular_members"] == 4 and not result["extracted"]
    assert result["splits"] == {
        "dev": {"files": 2, "rows_each": 1},
        "devtest": {"files": 2, "rows_each": 1},
    }
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "bad", ["../outside", "/absolute", "flores200_dataset/../../escape"]
)
def test_archive_refuses_unsafe_members(tmp_path, bad):
    path = tmp_path / "unsafe.tar.gz"
    archive(path, [(bad, b"fixture\n")])
    with pytest.raises(ValueError, match="unsafe"):
        assets.inspect_archive(path, fixture_dataset())


def test_archive_refuses_duplicate_missing_or_unaligned_original_members(tmp_path):
    path = tmp_path / "fixture.tar.gz"
    entries = fixture_entries()
    archive(path, entries + [entries[0]])
    with pytest.raises(ValueError, match="duplicate"):
        assets.inspect_archive(path, fixture_dataset())
    archive(path, entries[:-1])
    with pytest.raises(ValueError, match="coverage incomplete"):
        assets.inspect_archive(path, fixture_dataset())
    archive(path, [(name, raw + b"extra\n") for name, raw in entries])
    with pytest.raises(ValueError, match="line count"):
        assets.inspect_archive(path, fixture_dataset())


def test_sha_payload_and_git_blob_have_distinct_identity_semantics(tmp_path):
    path = tmp_path / "metadata.json"
    raw = b'{"fixture": true}\n'
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    assets.verify(path, {"bytes": len(raw), "git_blob": blob})
    assets.verify(
        path, {"bytes": len(raw), "sha256": sha, "git_blob": "LFS-pointer-not-payload"}
    )
    with pytest.raises(ValueError, match="payload identity"):
        assets.verify(path, {"bytes": len(raw), "git_blob": "wrong"})
    with pytest.raises(ValueError, match="contained"):
        assets.contained(tmp_path, "data", "../outside")


def test_verify_only_cli_never_downloads_writes_or_claims_native_completion(
    tmp_path, monkeypatch, capsys
):
    root = tmp_path / "existing"
    (root / "data").mkdir(parents=True)
    path = root / "data/fixture.tar.gz"
    archive(path, fixture_entries())
    pins = {
        "source_id": "FLORES200",
        "dataset": {
            **fixture_dataset(),
            "path": path.name,
            "url": "https://example.invalid/never-download",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    }
    pin_file = tmp_path / "fixture-pins.json"
    pin_file.write_text(json.dumps(pins), encoding="utf-8")
    monkeypatch.setattr(assets, "PINS", pin_file)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare", "--output", str(root), "--parts", "data", "--verify-only"],
    )
    monkeypatch.setattr(
        assets, "fetch", lambda *args: pytest.fail("verification must not download")
    )
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assets.main()
    result = json.loads(capsys.readouterr().out)
    assert result["model_calls"] == 0 and not result["source_checkbox_eligible"]
    assert not (root / "preparation.private.json").exists()
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def prepare_fixture(tmp_path, monkeypatch):
    original = tmp_path / "fixture-source.tar.gz"
    archive(original, fixture_entries())
    raw = original.read_bytes()
    pins = {
        "source_id": "FLORES200",
        "dataset": {
            **fixture_dataset(),
            "path": "original.tar.gz",
            "url": "https://example.invalid/fixture",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    }
    pin_file = tmp_path / "fixture-pins.json"
    pin_file.write_text(json.dumps(pins), encoding="utf-8")
    monkeypatch.setattr(assets, "PINS", pin_file)
    output = tmp_path / "fresh-output"
    monkeypatch.setattr(
        sys, "argv", ["prepare", "--output", str(output), "--parts", "data"]
    )
    return output, raw, pin_file, pins


def test_fresh_cli_and_matching_resume_never_overwrite_or_claim_execution(
    tmp_path, monkeypatch
):
    output, raw, _, _ = prepare_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    assets.main()
    receipt = json.loads((output / "preparation.private.json").read_text("utf-8"))
    assert receipt["status"] == "PINNED_ASSETS_PREPARED_NOT_NATIVE_EXECUTION"
    assert receipt["model_calls"] == 0 and not receipt["source_checkbox_eligible"]
    assert (output / "data/original.tar.gz").read_bytes() == raw
    assert not (output / "data/original.tar.gz.partial").exists()
    with pytest.raises(ValueError, match="fresh output"):
        assets.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
    monkeypatch.setattr(
        assets,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("completed asset must be reused"),
    )
    assets.main()
    assert (output / "data/original.tar.gz").read_bytes() == raw
    assert json.loads((output / "preparation.private.json").read_text("utf-8"))[
        "files"
    ][0]["reused"]


def test_resume_requires_same_profile_and_corruption_never_triggers_replacement(
    tmp_path, monkeypatch
):
    output, raw, pin_file, pins = prepare_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    assets.main()
    monkeypatch.setattr(sys, "argv", sys.argv + ["--resume"])
    pin_file.write_text(
        json.dumps({**pins, "different_profile": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="pin identity mismatch"):
        assets.main()
    pin_file.write_text(json.dumps(pins), encoding="utf-8")
    bad = b"corrupted software fixture"
    (output / "data/original.tar.gz").write_bytes(bad)
    monkeypatch.setattr(
        assets,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("do not replace corrupt existing assets"),
    )
    with pytest.raises(ValueError, match="size mismatch"):
        assets.main()
    assert (output / "data/original.tar.gz").read_bytes() == bad
    assert (
        json.loads((output / "preparation.private.json").read_text("utf-8"))["status"]
        == "FAILED_NOT_COMPLETE"
    )


def test_containment_refuses_symlinked_part_escape(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    try:
        (root / "model").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink privilege unavailable on this software test host")
    with pytest.raises(ValueError, match="contained"):
        assets.contained(root, "model", "file.json")


def test_fetch_refuses_oversized_response_and_existing_destination(
    tmp_path, monkeypatch
):
    path = tmp_path / "metadata.json"
    raw = b"fixture"
    spec = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(
        assets, "urlopen", lambda *args, **kwargs: io.BytesIO(raw + b"extra")
    )
    with pytest.raises(ValueError, match="exceeds pinned"):
        assets.fetch(path, spec, "https://example.invalid/fixture")
    assert not path.exists()
    monkeypatch.setattr(assets, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    path.write_bytes(b"existing asset")
    with pytest.raises(FileExistsError):
        assets.fetch(path, spec, "https://example.invalid/fixture")
    assert path.read_bytes() == b"existing asset"
