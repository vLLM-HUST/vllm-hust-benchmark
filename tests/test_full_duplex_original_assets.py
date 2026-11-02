"""Inert archive transport/identity fixtures, not native speech inference."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "duplex_assets", ROOT / "scripts/prepare_full_duplex_original_assets.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture_zip(name="original/metadata.json", content=b'{"original":true}'):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr(name.replace("\\", "-"), content)
    raw = out.getvalue()
    if "\\" in name:
        # Keep a literal hostile stored name even on Windows ZipInfo normalization.
        raw = raw.replace(name.replace("\\", "-").encode(), name.encode())
    return raw, {
        "path": "v1.5/fixture.zip",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "url": "https://example.invalid/fixture",
    }


def test_real_pins_and_offline_plan_do_not_create_output(tmp_path):
    pins = entry.load_pins()
    assert len(pins) == 9 and sum(x["bytes"] for x in pins) == 705372348
    assert all(
        x["url"].startswith("https://drive.usercontent.google.com/") for x in pins
    )
    target = tmp_path / "not-created"
    p = subprocess.run(
        [sys.executable, entry.__file__, "--output", str(target)],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and not target.exists()


def test_classification_routes_public_preparation_without_claiming_generation():
    catalog = json.loads(
        (
            ROOT
            / "workload-preparation/data/catalogs/workload_delivery_classification_v1.json"
        ).read_text("utf-8")
    )
    routes = catalog["source_entrypoints"]["FULL_DUPLEX_BENCH"]
    assert {x["role"] for x in routes} == {"prepare_assets"}
    assert entry.__file__.endswith("prepare_full_duplex_original_assets.py")
    assert any(
        x["path"] == "scripts/prepare_full_duplex_original_assets.py" for x in routes
    )


def test_original_archive_prepare_verify_and_reuse(tmp_path):
    raw, item = fixture_zip()
    entry.prepare(tmp_path, [item], opener=lambda *a, **k: io.BytesIO(raw))
    path = tmp_path / item["path"]
    assert path.read_bytes() == raw
    assert entry.verify_archive(path, item)["original_json_decoded"] == 1
    entry.prepare(
        tmp_path,
        [item],
        opener=lambda *a, **k: pytest.fail("verified archive must not redownload"),
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size"):
        entry.prepare(tmp_path, [item])


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "a\\b.json"])
def test_unsafe_zip_never_published(tmp_path, name):
    raw, item = fixture_zip(name)
    with pytest.raises(ValueError, match="unsafe"):
        entry.prepare(tmp_path, [item], opener=lambda *a, **k: io.BytesIO(raw))
    assert not (tmp_path / item["path"]).exists()
    assert (tmp_path / (item["path"] + ".partial")).exists()


def test_oversized_response_and_partial_no_overwrite(tmp_path):
    raw, item = fixture_zip()
    with pytest.raises(ValueError, match="byte budget"):
        entry.prepare(tmp_path, [item], opener=lambda *a, **k: io.BytesIO(raw + b"x"))
    with pytest.raises(ValueError, match="retained partial"):
        entry.prepare(tmp_path, [item], opener=lambda *a, **k: io.BytesIO(raw))


def test_bad_json_or_digest_rejected(tmp_path):
    raw, item = fixture_zip(content=b"not JSON")
    with pytest.raises(ValueError):
        entry.prepare(tmp_path, [item], opener=lambda *a, **k: io.BytesIO(raw))
    other = tmp_path / "other"
    with pytest.raises(ValueError, match="SHA256"):
        entry.prepare(
            other,
            [dict(item, sha256="0" * 64)],
            opener=lambda *a, **k: io.BytesIO(raw),
        )


def test_contained_path_and_checkout_guards(tmp_path):
    with pytest.raises(ValueError, match="contained"):
        entry.archive_path(tmp_path, "../escape.zip")
    p = subprocess.run(
        [sys.executable, entry.__file__, "--output", str(ROOT / "work/forbidden")],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0 and "outside checkout" in p.stderr
