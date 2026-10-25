"""Small fabricated software fixtures, never real dataset validation."""

import csv
import hashlib
import json

import pytest

from vllm_hust_workload_data.narrativeqa import original_summary_tasks


def fixture(tmp_path, *, missing=False, duplicate=False, split=False):
    data = tmp_path / "data"
    catalog = tmp_path / "catalog"
    (catalog / "data/catalogs").mkdir(parents=True)
    tables = {
        "documents.csv": [
            {
                "document_id": "doc",
                "set": "test",
                "story_url": "https://example.invalid/story",
            }
        ],
        "qaps.csv": [
            {
                "document_id": "absent" if missing else "doc",
                "set": "test",
                "question": "fixture question",
                "answer1": "fixture answer",
                "answer2": "second answer",
            }
        ],
        "third_party/wikipedia/summaries.csv": [
            {
                "document_id": "doc",
                "set": "train" if split else "test",
                "summary": "fixture summary",
            }
        ],
    }
    if duplicate:
        tables["documents.csv"] *= 2
    files = []
    for name, rows in tables.items():
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        raw = path.read_bytes()
        files.append(
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        )
    (catalog / "data/catalogs/source_manifests_v1.json").write_text(
        json.dumps(
            {
                "sources": {
                    "NARRATIVEQA": {"revision": "software-fixture", "files": files}
                }
            }
        ),
        encoding="utf-8",
    )
    return data, catalog


def test_original_relations_and_answers_stay_separate(tmp_path):
    data, catalog = fixture(tmp_path)
    result = list(original_summary_tasks(data, catalog_root=catalog))
    assert len(result) == 1
    task = result[0]
    assert task["input"] == {
        "document_id": "doc",
        "question": "fixture question",
        "summary": "fixture summary",
    }
    assert task["outcome"]["answer1"] == "fixture answer"
    assert task["original_relations"]["document"]["story_url"].endswith("/story")
    assert "answer1" not in task["input"]
    assert task["setting"] == "publisher_summary_not_full_story"
    assert not task["native_execution_complete"]


@pytest.mark.parametrize(
    "option,reason",
    [
        ("missing", "unresolved"),
        ("duplicate", "duplicate"),
        ("split", "split relation"),
    ],
)
def test_bad_relations_fail_without_substitution(tmp_path, option, reason):
    data, catalog = fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=reason):
        list(original_summary_tasks(data, catalog_root=catalog))


def test_table_mutation_is_rejected_before_reading(tmp_path):
    data, catalog = fixture(tmp_path)
    (data / "qaps.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_summary_tasks(data, catalog_root=catalog))
