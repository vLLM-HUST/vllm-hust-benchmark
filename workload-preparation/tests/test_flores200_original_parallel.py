"""Tiny invented parallel-archive fixtures; no original dataset/model execution."""

import hashlib
import io
import json
import sqlite3
import tarfile

import pytest

from vllm_hust_workload_data import flores200
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.adapters import iter_workload


def fixture(root, *, metadata=True, prefix="./"):
    root.mkdir(parents=True)
    path = root / "flores200_dataset.tar.gz"
    texts = {
        "eng_Latn": [" original source ", "", "third source", "fourth source"],
        "fra_Latn": [" reference only ", "", "third reference", "fourth reference"],
    }
    with tarfile.open(path, "w:gz") as archive:
        for split in ("dev", "devtest"):
            for code, lines in texts.items():
                data = ("\n".join(lines) + "\n").encode()
                info = tarfile.TarInfo(
                    f"{prefix}flores200_dataset/{split}/{code}.{split}"
                )
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if metadata:
                data = (
                    "URL\tdomain\ttopic\thas_image\thas_hyperlink\n"
                    + "\n".join(
                        f"https://example.invalid/{split}/{i}\tfixture-domain\tfixture-topic\tFalse\tTrue"
                        for i in range(4)
                    )
                    + "\n"
                ).encode()
                info = tarfile.TarInfo(
                    f"{prefix}flores200_dataset/metadata_{split}.tsv"
                )
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    spec = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_rows": {"dev": 4, "devtest": 4},
        "language_code_count": 2,
    }
    return path, spec, texts


def catalog(tmp_path, spec):
    root = tmp_path / "catalog"
    directory = root / "data/catalogs"
    directory.mkdir(parents=True)
    (directory / "source_manifests_v1.json").write_text(
        json.dumps(
            {
                "sources": {
                    "FLORES200": {
                        "source_id": "FLORES200",
                        "revision": "fixture-pinned-original-archive",
                        "source_url": "https://example.invalid/original-fixture",
                        "unit": "original_parallel_row",
                        "provenance_kind": "fixture_only",
                        "archive_layout": {
                            k: spec[k] for k in ("split_rows", "language_code_count")
                        },
                        "files": [
                            {
                                "path": "flores200_dataset.tar.gz",
                                "role": "data",
                                "rows": 8,
                                **{k: spec[k] for k in ("bytes", "sha256")},
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def test_complete_parallel_rows_metadata_and_original_blank_lines(tmp_path):
    path, spec, texts = fixture(tmp_path / "raw")
    rows = list(flores200.parallel_rows(path, dataset=spec))
    assert len(rows) == 8
    assert [(r["split"], r["row_index"]) for r in rows] == [
        (s, i) for s in ("dev", "devtest") for i in range(4)
    ]
    assert rows[0]["parallel_texts"] == {
        code: lines[0] for code, lines in texts.items()
    }
    assert rows[1]["parallel_texts"] == {"eng_Latn": "", "fra_Latn": ""}
    assert rows[4]["publisher_metadata"]["URL"] == "https://example.invalid/devtest/0"
    assert rows[4]["publisher_metadata"]["has_image"] == "False"


def test_archive_or_metadata_incompleteness_never_silently_filters(tmp_path):
    path, spec, _ = fixture(tmp_path / "missing", metadata=False)
    with pytest.raises(ValueError, match="metadata alignment"):
        list(flores200.parallel_rows(path, dataset=spec))
    path, spec, _ = fixture(tmp_path / "identity")
    bad = {**spec, "bytes": spec["bytes"] + 1}
    with pytest.raises(ValueError, match="identity"):
        list(flores200.parallel_rows(path, dataset=bad))
    bad = {**spec, "language_code_count": 3}
    with pytest.raises(ValueError, match="language coverage"):
        list(flores200.parallel_rows(path, dataset=bad))


def test_explicit_direction_preserves_source_reference_metadata_and_profile_identity(
    tmp_path, monkeypatch
):
    path, spec, _ = fixture(tmp_path / "raw")
    root = catalog(tmp_path, spec)
    monkeypatch.setattr(flores200, "dataset_spec", lambda **kwargs: spec)
    ids = []
    for number, (src, tgt) in enumerate(
        [("eng_Latn", "fra_Latn"), ("fra_Latn", "eng_Latn")]
    ):
        mapping = flores200.translation_mapping(src, tgt, catalog_root=root)
        assert mapping["features"] == {} and mapping["identity"]["task_id"] is None
        manifest = bind_source(
            "FLORES200",
            data_root=path.parent,
            files=[path.name],
            revision=mapping["source"]["revision"],
            mapping=mapping,
            catalog_root=root,
        )
        records = list(
            iter_workload(
                "FLORES200", data_root=path.parent, mapping=mapping, manifest=manifest
            )
        )
        assert len(records) == 8
        first = records[0]
        assert first.task["input"]["source_language"] == src
        assert first.task["input"]["target_language"] == tgt
        assert first.task["input"]["source_text"] == (
            " original source " if src == "eng_Latn" else " reference only "
        )
        assert "publisher_metadata" not in first.task["input"]
        assert "reference_text" not in first.task["input"]
        assert first.outcome_record()["reference_text"] == (
            " reference only " if tgt == "fra_Latn" else " original source "
        )
        assert first.outcome_record()["publisher_metadata"]["topic"] == "fixture-topic"
        assert first.task["identity_kind"] == "file_sha256_row_index"
        with pytest.raises(ValueError, match="not a ready model request"):
            first.request_body()
        profile = tmp_path / f"profile-{number}"
        profile.mkdir()
        mapping_file = profile / "mapping.json"
        mapping_file.write_text(json.dumps(mapping), encoding="utf-8")
        manifest_file = profile / "manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        report = run(
            {
                "sources": [
                    {
                        "source_id": "FLORES200",
                        "data_root": str(path.parent),
                        "mapping": str(mapping_file),
                        "manifest": str(manifest_file),
                    }
                ]
            },
            output=profile / "prepared",
            catalog_root=root,
        )
        assert report["status"] == "PREPARED_REFERENCES_ONLY"
        database = profile / "prepared/workload.sqlite"
        refs = list(prepared_rows(database))
        assert len(refs) == 8
        ids.append(refs[0]["record_id"])
        for i in [0, 1, 4, 7]:
            resolved = resolve_input(
                database, refs[i]["record_id"], data_root=path.parent
            )
            assert resolved.task["input"] == records[i].task["input"]
            assert resolved.outcome_record() == records[i].outcome_record()
        with sqlite3.connect(database) as connection:
            stored = json.loads(
                connection.execute(
                    "SELECT task_json FROM records WHERE record_id=?",
                    (refs[0]["record_id"],),
                ).fetchone()[0]
            )
            stored["input_ref"]["mapping_sha256"] = "wrong-direction-binding"
            connection.execute(
                "UPDATE records SET task_json=? WHERE record_id=?",
                (json.dumps(stored), refs[0]["record_id"]),
            )
        with pytest.raises(ValueError, match="direction/profile binding"):
            resolve_input(database, refs[0]["record_id"], data_root=path.parent)
    assert ids[0] != ids[1]


def test_same_or_missing_original_language_code_is_not_substituted(
    tmp_path, monkeypatch
):
    path, spec, _ = fixture(tmp_path / "raw")
    root = catalog(tmp_path, spec)
    monkeypatch.setattr(flores200, "dataset_spec", lambda **kwargs: spec)
    with pytest.raises(ValueError, match="distinct"):
        flores200.translation_mapping("eng_Latn", "eng_Latn", catalog_root=root)
    mapping = flores200.translation_mapping("arb_Latn", "eng_Latn", catalog_root=root)
    with pytest.raises(ValueError):
        bind_source(
            "FLORES200",
            data_root=path.parent,
            files=[path.name],
            revision=mapping["source"]["revision"],
            mapping=mapping,
            catalog_root=root,
        )


@pytest.mark.parametrize("failure", ["absolute", "parent", "symlink", "duplicate"])
def test_archive_safety_bounds_apply_before_any_rows(tmp_path, failure):
    path = tmp_path / "untrusted-fixture.tar"
    with tarfile.open(path, "w") as archive:
        name = (
            "/escape"
            if failure == "absolute"
            else "../escape"
            if failure == "parent"
            else "flores200_dataset/README"
        )
        member = tarfile.TarInfo(name)
        if failure == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
        archive.addfile(member, io.BytesIO(b""))
        if failure == "duplicate":
            archive.addfile(member, io.BytesIO(b""))
    spec = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_rows": {"dev": 4, "devtest": 4},
        "language_code_count": 2,
    }
    with pytest.raises(ValueError, match="unsafe|duplicate"):
        list(flores200.parallel_rows(path, dataset=spec))
