"""Invented fixtures test contracts, not original BBH execution or scores."""

from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data import iter_workload, provider
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_bbh_projection_separates_target_and_keeps_original_text():
    row = {"input": " fixture\ninput ", "target": "REFERENCE_SENTINEL"}
    before = deepcopy(row)
    source = {
        "source_id": "BBH",
        "revision": "unit-test-not-evidence",
        "file": "fixture.json",
        "file_sha256": "0" * 64,
    }
    record = adapt_row("BBH", row, source=source, spec=adapter_spec("BBH"), index=2)
    assert record.task["input"] == {"input": row["input"]}
    assert record.outcome_record() == {"target": "REFERENCE_SENTINEL"}
    assert "REFERENCE_SENTINEL" not in json.dumps(record.input_record())
    assert row == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("input", ""),
        ("input", []),
        ("input", None),
        ("target", ""),
        ("target", {}),
        ("target", None),
    ],
)
def test_invalid_original_field_types_fail_closed(field, value):
    row = {"input": "fixture", "target": "reference", field: value}
    with pytest.raises(ValueError):
        adapt_row("BBH", row, source={}, spec=adapter_spec("BBH"), index=0)


def synthetic_archive(tmp_path, monkeypatch):
    task = json.dumps({"examples": [{"input": "test", "target": "hidden"}]}).encode()
    prompt = (
        "fixture description\n-----\n"
        + "\n\n".join(f"Q: demo-{index}\nA: fixture answer" for index in range(3))
    ).encode()
    assets = {
        "boolean_expressions.json": task,
        "cot-prompts/boolean_expressions.txt": prompt,
    }
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("fixture-root/bbh/boolean_expressions.json", task)
        bundle.writestr("fixture-root/cot-prompts/boolean_expressions.txt", prompt)
        bundle.writestr("fixture-root/ignored/private.json", "not copied")
    spec = deepcopy(source_spec("BBH"))
    for asset in spec["files"]:
        if asset["path"] not in assets:
            continue
        asset["sha256"] = hashlib.sha256(assets[asset["path"]]).hexdigest()
        asset["bytes"] = len(assets[asset["path"]])
        if asset["role"] == "data":
            asset["rows"] = 1
    monkeypatch.setattr(
        provider, "BBH_ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(provider, "source_spec", lambda source_id: spec)
    return archive, spec, assets


def synthetic_all_bbh_archive(tmp_path, monkeypatch):
    original = source_spec("BBH")
    spec = deepcopy(original)
    task_assets = [asset for asset in original["files"] if asset["role"] == "data"]
    prompt_assets = [
        asset for asset in original["files"] if asset["role"] == "publisher_prompt"
    ]
    assets = {}
    for index, asset in enumerate(task_assets):
        assets[asset["path"]] = json.dumps(
            {
                "examples": [
                    {"input": f"fixture-{index}", "target": f"TARGET_SENTINEL_{index}"}
                ]
            }
        ).encode()
    for asset in prompt_assets:
        assets[asset["path"]] = (
            "fixture description\n-----\n"
            + "\n\n".join(f"Q: demo-{index}\nA: fixture answer" for index in range(3))
        ).encode()
    for asset in spec["files"]:
        asset["sha256"] = hashlib.sha256(assets[asset["path"]]).hexdigest()
        asset["bytes"] = len(assets[asset["path"]])
        if asset["role"] == "data":
            asset["rows"] = 1
    archive = tmp_path / "all-fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for asset in spec["files"]:
            bundle.writestr(
                "fixture-root/" + asset["archive_member"], assets[asset["path"]]
            )
    monkeypatch.setattr(
        provider, "BBH_ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(provider, "source_spec", lambda source_id: spec)
    return archive, spec, assets


def test_two_original_assets_copied_without_executing_or_overwriting(
    tmp_path, monkeypatch
):
    archive, spec, assets = synthetic_archive(tmp_path, monkeypatch)
    out = tmp_path / "prepared"
    result = provider.prepare_bbh_boolean(data_archive=archive, output=out)
    assert result["model_calls"] == result["scorer_runs"] == 0
    assert result["subtasks"] == ["boolean_expressions"]
    assert {
        p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
    } == set(assets)
    assert all((out / name).read_bytes() == raw for name, raw in assets.items())
    records = list(
        iter_workload(
            "BBH",
            data_root=out,
            files=["boolean_expressions.json"],
            manifest=spec,
        )
    )
    assert len(records) == 1
    assert records[0].task["input"] == {"input": "test"}
    assert records[0].outcome_record() == {"target": "hidden"}
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.pipeline import run, prepared_rows
    from vllm_hust_workload_data.consumer import resolve_input

    binding = bind_source(
        "BBH",
        data_root=out,
        files=["boolean_expressions.json"],
        revision="invented-unit-test-not-original-source",
    )
    manifest = tmp_path / "binding.json"
    manifest.write_text(json.dumps(binding), encoding="utf-8")
    expected = list(iter_workload("BBH", data_root=out, manifest=binding, limit=1))
    run(
        {
            "sources": [
                {
                    "source_id": "BBH",
                    "data_root": str(out),
                    "manifest": str(manifest),
                    "limit": 1,
                }
            ]
        },
        output=tmp_path / "index",
    )
    database = tmp_path / "index/workload.sqlite"
    refs = list(prepared_rows(database))
    assert len(refs) == 1
    actual = resolve_input(database, refs[0]["record_id"], data_root=out)
    assert actual.input_record() == expected[0].input_record()
    assert actual.outcome_record() == expected[0].outcome_record()
    assert actual.task["input"] == {"input": "test"}
    assert actual.outcome_record() == {"target": "hidden"}
    with pytest.raises(FileExistsError):
        provider.prepare_bbh_boolean(data_archive=archive, output=out)


def test_bad_archive_rejected_before_output_creation(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"unknown")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="archive identity"):
        provider.prepare_bbh_boolean(data_archive=archive, output=out)
    assert not out.exists()


def test_member_checksum_mismatch_rejected_before_output_creation(
    tmp_path, monkeypatch
):
    archive, spec, _ = synthetic_archive(tmp_path, monkeypatch)
    spec["files"][0]["sha256"] = "0" * 64
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="member identity"):
        provider.prepare_bbh_boolean(data_archive=archive, output=out)
    assert not out.exists()


def test_all_original_bbh_task_and_prompt_pairs_prepare_and_read_without_execution(
    tmp_path, monkeypatch
):
    archive, spec, assets = synthetic_all_bbh_archive(tmp_path, monkeypatch)
    out = tmp_path / "prepared-all"
    result = provider.prepare_bbh_original_tasks(data_archive=archive, output=out)
    assert result["task_file_count"] == result["publisher_prompt_file_count"] == 27
    assert result["original_rows"] == 27
    assert result["model_calls"] == result["scorer_runs"] == 0
    assert len(result["subtasks"]) == 27
    assert {
        path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()
    } == set(assets)
    assert all((out / name).read_bytes() == raw for name, raw in assets.items())

    records = list(iter_workload("BBH", data_root=out, manifest=spec))
    assert len(records) == 27
    assert all(
        record.task["input"]["input"].startswith("fixture-")
        and record.outcome_record()["target"].startswith("TARGET_SENTINEL_")
        and "TARGET_SENTINEL_" not in json.dumps(record.input_record())
        for record in records
    )
