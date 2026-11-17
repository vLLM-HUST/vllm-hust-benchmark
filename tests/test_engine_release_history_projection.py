"""Publish the exact historical catalog while retaining unrelated withdrawal rules."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "history_projection", ROOT / "scripts/build_public_site.py"
)
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


def prepare(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "assets").mkdir()
    for name in [
        "index.html",
        "plugins.html",
        "dataset-introductions.html",
        "leaderboard-runs.html",
        "assets/site.js",
    ]:
        (source / name).write_text("")
    raw = (
        ROOT / "website-publication/data/engine-release-history-v1.json"
    ).read_bytes()
    (source / projection.PUBLIC_ENGINE_HISTORY_PATH).write_bytes(raw)
    (source / "data/unrelated.json").write_text(
        json.dumps(
            {"records": [{"id": "unrelated", "mods": ["BetterScale"], "value": 42}]}
        )
    )
    entry = dict(
        website_path=projection.PUBLIC_ENGINE_HISTORY_PATH,
        sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw),
    )
    (source / "data/benchmark-source-manifest.json").write_text(
        json.dumps({"files": [entry]})
    )
    return source, raw


def test_release_history_survives_actual_public_build(tmp_path):
    source, raw = prepare(tmp_path)
    output = tmp_path / "public"
    projection.build(source, output, "test-policy")
    projection.validate(output)
    assert (output / projection.PUBLIC_ENGINE_HISTORY_PATH).read_bytes() == raw
    value = json.loads(raw)
    assert {entry["id"] for entry in value["entries"]} == {"r4", "r5", "r6", "r7", "r8"}
    assert {entry["id"] for entry in value["attempts"]} == {"r1", "r2", "r3"}
    assert json.loads((output / "data/unrelated.json").read_text()) == {"records": []}


def test_resigning_manifest_cannot_publish_changed_history(tmp_path):
    source, raw = prepare(tmp_path)
    changed = json.loads(raw)
    changed["entries"][0]["measurements"][0]["arms"][1]["output_tokens_per_second"] *= 2
    rewritten = (json.dumps(changed) + "\n").encode()
    path = source / projection.PUBLIC_ENGINE_HISTORY_PATH
    path.write_bytes(rewritten)
    manifest = source / "data/benchmark-source-manifest.json"
    value = json.loads(manifest.read_text())
    value["files"][0].update(
        sha256=hashlib.sha256(rewritten).hexdigest(), bytes=len(rewritten)
    )
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="release history"):
        projection.build(source, tmp_path / "public", "test-policy")
