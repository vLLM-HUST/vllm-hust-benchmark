from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_visionarena_frozen_input.py"
MANIFEST = (
    ROOT
    / "docs"
    / "dataset-manifests"
    / "visionarena-chat-1394b4f-seed0-1000.manifest.json"
)


def _module():
    spec = importlib.util.spec_from_file_location("visionarena_frozen", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_zero_selection_is_canonical_and_stable() -> None:
    pytest.importorskip("numpy")
    module = _module()
    shard_order, selected = module.derive_selection()

    assert shard_order[:3] == [40, 2, 21]
    assert len(selected) == 1000
    assert len(set(selected)) == 1000
    assert max(selected) == 1973
    assert module._compact_json_sha256(shard_order) == (
        "c2121f82122d83bc005730dab935defc9175af24013d10b868d97b64917306b2"
    )
    assert module._compact_json_sha256(selected) == (
        "6e1b1d169e836663328c6f9a38145ed093192f93f36656b26327f2244eb583bc"
    )


def test_manifest_matches_derived_contract() -> None:
    module = _module()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "visionarena-frozen-input/v1"
    assert manifest["dataset"]["revision"] == module.REVISION
    assert manifest["reference_loader"]["version"] == "3.3.0"
    assert manifest["reference_loader"]["streaming"] is True
    assert manifest["selection"]["source_index_sha256"] == (
        module.EXPECTED_SOURCE_INDEX_SHA256
    )
    assert manifest["selection"]["content_sha256"] == (module.EXPECTED_CONTENT_SHA256)
    assert manifest["source_shard"]["sha256"] == module.EXPECTED_SHARD_SHA256


def test_prepare_creates_idempotent_one_shard_view(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "frozen"

    first = module.prepare(output, source)
    second = module.prepare(output, source)

    assert first == second
    assert first.is_symlink()
    assert first.resolve() == source.resolve()
    assert list((output / "data").iterdir()) == [first]


def test_reference_row_rng_matches_numpy_default_rng() -> None:
    np = pytest.importorskip("numpy")
    module = _module()
    _, selected = module.derive_selection()
    slots = np.random.default_rng(0).integers(0, 1000, size=1000)

    assert selected[:5] == [850, 636, 511, 269, 307]
    assert slots[:5].tolist() == [850, 636, 511, 269, 307]
