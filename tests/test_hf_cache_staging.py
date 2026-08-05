from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vllm_hust_benchmark import hf_cache_staging as staging


MODEL_REPOSITORY = "owner/model"
MODEL_REVISION = "a" * 40
DATASET_REPOSITORY = "owner/dataset"
DATASET_REVISION = "b" * 40


def _write_export(
    root: Path, files: dict[str, bytes], revision: str, *, checksums: bool = False
) -> None:
    records = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        metadata = root / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(f"{revision}\netag\n0\n", encoding="utf-8")
        records.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
    if checksums:
        (root / "SHA256SUMS").write_text("\n".join(records) + "\n", encoding="utf-8")


def _model_files() -> dict[str, bytes]:
    return {
        "config.json": b"{}\n",
        "model-00001-of-00001.safetensors": b"weights",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer": "model-00001-of-00001.safetensors"}}
        ).encode(),
        "tokenizer.json": b"{}\n",
    }


def test_staged_exact_revisions_resolve_offline(tmp_path: Path) -> None:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    model_source = tmp_path / "flat-model"
    dataset_source = tmp_path / "flat-dataset"
    _write_export(model_source, _model_files(), MODEL_REVISION, checksums=True)
    _write_export(dataset_source, {"train.json": b"[]\n"}, DATASET_REVISION)
    hub = tmp_path / "scratch" / "hub"
    model = staging.inspect_flat_export(
        model_source,
        repository=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        repo_type="model",
    )
    dataset = staging.inspect_flat_export(
        dataset_source,
        repository=DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        repo_type="dataset",
    )
    staged_model = staging.stage_flat_export(model, hub)
    staged_dataset = staging.stage_flat_export(dataset, hub)
    assert (
        huggingface_hub.snapshot_download(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            cache_dir=hub,
            local_files_only=True,
        )
        == staged_model["snapshot_path"]
    )
    assert (
        huggingface_hub.snapshot_download(
            DATASET_REPOSITORY,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=hub,
            local_files_only=True,
        )
        == staged_dataset["snapshot_path"]
    )
    assert (model_source / "config.json").read_bytes() == b"{}\n"


def test_wrong_revision_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "model"
    _write_export(source, _model_files(), "c" * 40)
    with pytest.raises(staging.StagingFailure, match="revision mismatch"):
        staging.inspect_flat_export(
            source,
            repository=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            repo_type="model",
        )


def test_missing_model_shard_fails_closed(tmp_path: Path) -> None:
    files = _model_files()
    del files["model-00001-of-00001.safetensors"]
    source = tmp_path / "model"
    _write_export(source, files, MODEL_REVISION)
    with pytest.raises(staging.StagingFailure, match="missing files"):
        staging.inspect_flat_export(
            source,
            repository=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            repo_type="model",
        )


def test_missing_dataset_file_with_stale_metadata_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_export(source, {"train.json": b"[]", "valid.json": b"[]"}, DATASET_REVISION)
    (source / "valid.json").unlink()
    with pytest.raises(staging.StagingFailure, match="metadata file set"):
        staging.inspect_flat_export(
            source,
            repository=DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )


def test_partial_file_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_export(source, {"train.json": b"[]"}, DATASET_REVISION)
    (source / ".cache" / "download.incomplete").write_bytes(b"partial")
    with pytest.raises(staging.StagingFailure, match="partial/lock"):
        staging.inspect_flat_export(
            source,
            repository=DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )


def test_source_symlink_escape_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_export(source, {"train.json": b"[]"}, DATASET_REVISION)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"[]")
    (source / "escape.json").symlink_to(outside)
    with pytest.raises(staging.StagingFailure, match="escapes flat export"):
        staging.inspect_flat_export(
            source,
            repository=DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )


def test_registry_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    spec = repo / "docs" / "target.json"
    spec.parent.mkdir(parents=True)
    spec_payload = {
        "model": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "data_identity": {
            "kind": "huggingface-dataset",
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
    }
    spec.write_text(json.dumps(spec_payload), encoding="utf-8")
    registry_path = repo / "src/vllm_hust_benchmark/data/official_targets.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "registry_version": "test",
                "targets": [
                    {
                        "target_id": "target",
                        "source_spec": {
                            "path": "docs/target.json",
                            "sha256": staging._sha256(spec),
                        },
                        "model": {"id": MODEL_REPOSITORY, "revision": "c" * 40},
                        "workload": {"data_identity": spec_payload["data_identity"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(staging.StagingFailure, match="differs from the registry"):
        staging._registry_target(repo, spec)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("huggingface-dataset", "huggingface-dataset"),
        ("huggingface-file", "huggingface-file"),
        ("release-asset", "release-asset"),
    ],
)
def test_target_data_kind_comes_from_the_hashed_registry_contract(
    tmp_path: Path, kind: str, expected: str
) -> None:
    repo = tmp_path / "repo"
    spec = repo / "docs" / "target.json"
    spec.parent.mkdir(parents=True)
    data_identity = {"kind": kind, "repository": DATASET_REPOSITORY}
    spec_payload = {
        "model": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "data_identity": data_identity,
    }
    spec.write_text(json.dumps(spec_payload), encoding="utf-8")
    registry_path = repo / "src/vllm_hust_benchmark/data/official_targets.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "source_spec": {
                            "path": "docs/target.json",
                            "sha256": staging._sha256(spec),
                        },
                        "model": {
                            "id": MODEL_REPOSITORY,
                            "revision": MODEL_REVISION,
                        },
                        "workload": {"data_identity": data_identity},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert staging.target_data_identity_kind(repo, spec) == expected


def test_official_runner_stages_before_model_and_contract_resolution() -> None:
    script = (
        Path(__file__).parents[1] / "scripts/run-official-ascend-goal-baseline.sh"
    ).read_text(encoding="utf-8")
    staging_call = script.index("  stage_flat_hf_inputs\n")
    model_resolution = script.index("if cached_model_path=$(resolve_runtime_model)")
    contract = script.index(
        "IMMUTABLE_INPUT_METADATA=$(verify_immutable_input_contract)"
    )
    assert staging_call < model_resolution < contract
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in script
    assert '[[ "$data_identity_kind" == "huggingface-dataset"' in script
    assert 'args+=(--dataset-source "$OFFICIAL_FLAT_DATASET_SOURCE")' in script
