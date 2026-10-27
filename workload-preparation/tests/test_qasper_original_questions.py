"""Synthetic relation fixtures only; original dataset stays outside the repo."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import io

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.qasper import question_rows
from vllm_hust_workload_data.registry import source_spec
from vllm_hust_workload_data.catalog import download_plan


def papers():
    return {
        "paper-one": {
            "title": "fixture title",
            "abstract": "complete abstract",
            "full_text": [
                {"section_name": "Intro", "paragraphs": ["whole paragraph", ""]}
            ],
            "figures_and_tables": [
                {"caption": "original caption", "file": "figure.png"}
            ],
            "qas": [
                {
                    "question_id": f"question-{index}",
                    "question": f"original question {index}?",
                    "question_writer": "fixture writer",
                    "answers": [
                        {
                            "annotation_id": f"annotation-{index}",
                            "worker_id": "fixture worker",
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["REFERENCE_SENTINEL"],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": ["EVIDENCE_SENTINEL"],
                                "highlighted_evidence": ["HIGHLIGHT_SENTINEL"],
                            },
                        }
                    ],
                }
                for index in range(3)
            ],
        }
    }


def test_questions_preserve_full_paper_and_all_reference_annotations(tmp_path):
    original = papers()
    file = tmp_path / "fixture.json"
    file.write_text(json.dumps(original), encoding="utf-8")
    rows = list(question_rows(file))
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert row["source_paper_row_index"] == 0
        assert row["source_question_index"] == index
        assert row["full_text"] == original["paper-one"]["full_text"]
        assert row["figures_and_tables"] == original["paper-one"]["figures_and_tables"]
        assert row["answers"] == original["paper-one"]["qas"][index]["answers"]
        assert "qas" not in row


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate_question",
        "missing_paper",
        "bad_question",
        "bad_reference",
        "bad_section",
    ],
)
def test_malformed_relations_fail_without_payloads(tmp_path, mode):
    original = papers()
    paper = original["paper-one"]
    if mode == "duplicate_question":
        paper["qas"][1]["question_id"] = paper["qas"][0]["question_id"]
    elif mode == "missing_paper":
        del paper["full_text"]
    elif mode == "bad_question":
        paper["qas"][0]["question"] = None
    elif mode == "bad_reference":
        paper["qas"][0]["answers"] = []
    else:
        paper["full_text"][0]["paragraphs"] = "SENTINEL"
    file = tmp_path / "fixture.json"
    file.write_text(json.dumps(original), encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(question_rows(file))
    assert "SENTINEL" not in str(error.value)


@pytest.mark.parametrize(
    "text", ['{"secret":1,"secret":2}', '{"secret": NaN}', '[{"paper":"wrong root"}]']
)
def test_json_structure_fails_closed(tmp_path, text):
    file = tmp_path / "fixture.json"
    file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(question_rows(file))


def test_pipeline_resolver_preserve_original_question_and_reference_separation(
    tmp_path,
):
    data = tmp_path / "data"
    data.mkdir()
    file = data / "fixture.json"
    file.write_text(json.dumps(papers()), encoding="utf-8")
    manifest = deepcopy(source_spec("QASPER"))
    manifest["revision"] = "synthetic-software-fixture-not-original"
    manifest["files"] = [
        {
            "path": file.name,
            "role": "data",
            "rows": 3,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "bytes": file.stat().st_size,
        }
    ]
    records = list(iter_workload("QASPER", data_root=data, manifest=manifest))
    assert len(records) == 3
    for record in records:
        assert "SENTINEL" not in json.dumps(record.input_record())
        assert record.outcome_record()["answers"][0]["answer"]["evidence"] == [
            "EVIDENCE_SENTINEL"
        ]
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    run(
        {
            "sources": [
                {
                    "source_id": "QASPER",
                    "data_root": str(data),
                    "manifest": str(binding),
                }
            ]
        },
        output=tmp_path / "index",
    )
    database = tmp_path / "index/workload.sqlite"
    for row, expected in zip(prepared_rows(database), records, strict=True):
        actual = resolve_input(database, row["record_id"], data_root=data)
        assert actual.input_record() == expected.input_record()
        assert actual.outcome_record() == expected.outcome_record()


def test_publisher_manifest_is_original_three_splits_not_longbench200():
    manifest = source_spec("QASPER")
    files = [asset for asset in manifest["files"] if asset["role"] == "data"]
    assert manifest["unit"] == "original_questions"
    assert manifest["revision"] == "publisher-v0.3"
    assert [asset["rows"] for asset in files] == [2593, 1005, 1451]
    assert sum(asset["original_papers"] for asset in files) == 1585
    assert sum(asset["rows"] for asset in files) == 5049
    assert (
        len(
            [
                asset
                for asset in manifest["files"]
                if asset["role"] == "publisher_scorer"
            ]
        )
        == 1
    )


def test_original_media_downloads_are_distinct_from_raw_json_members():
    plan = download_plan("QASPER")
    assert len(plan["media_archives"]) == 2
    assert sum(asset["bytes"] for asset in plan["media_archives"]) == 603112831
    assert plan["media_archives"][0]["archive_namespaces"] == [
        "train_dev_figures_and_tables/train",
        "train_dev_figures_and_tables/dev",
    ]
    assert plan["media_archives"][1]["archive_namespaces"] == [
        "test_figures_and_tables"
    ]
    assert not any(asset["path"].endswith(".tgz") for asset in plan["files"])


def test_prepare_helper_extracts_only_verified_members_without_execution(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "qasper_preparation", root / "scripts/prepare_qasper_original_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files, archives = [], {}
    for split, archive_name in [
        ("train", "qasper-train-dev-v0.3.tgz"),
        ("test", "qasper-test-and-evaluator-v0.3.tgz"),
    ]:
        body = json.dumps(papers()).encode()
        member = split + ".json"
        archive = tmp_path / archive_name
        with tarfile.open(archive, "w:gz") as bundle:
            info = tarfile.TarInfo(member)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
            info = tarfile.TarInfo("../unselected-danger.py")
            info.size = 0
            bundle.addfile(info, io.BytesIO())
        archives[archive_name] = (
            archive.stat().st_size,
            hashlib.sha256(archive.read_bytes()).hexdigest(),
        )
        files.append(
            {
                "path": member,
                "archive_member": member,
                "url": "https://example.invalid/" + archive_name,
                "role": "data",
                "split": split,
                "rows": 3,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    monkeypatch.setattr(module, "ARCHIVES", archives)
    monkeypatch.setattr(
        module, "source_spec", lambda _: {"revision": "fixture", "files": files}
    )
    output = tmp_path / "prepared"
    receipt = module.prepare(
        tmp_path / "qasper-train-dev-v0.3.tgz",
        tmp_path / "qasper-test-and-evaluator-v0.3.tgz",
        output,
    )
    assert receipt["model_calls"] == 0 and not receipt["native_benchmark_completed"]
    assert {file.name for file in output.iterdir()} == {
        "train.json",
        "test.json",
        "preparation-receipt.json",
    }
    assert not (tmp_path / "unselected-danger.py").exists()
    with pytest.raises(FileExistsError):
        module.prepare(
            tmp_path / "qasper-train-dev-v0.3.tgz",
            tmp_path / "qasper-test-and-evaluator-v0.3.tgz",
            output,
        )
