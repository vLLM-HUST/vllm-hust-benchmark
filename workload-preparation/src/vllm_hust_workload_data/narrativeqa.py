"""Resolve publisher NarrativeQA table relations, not a model/scoring runner.

Summary and full-story settings are distinct. Never substitute a summary for a
missing story, download URLs implicitly, or expose answer tables as model input.
"""

from copy import deepcopy

from .assets import asset_path, iter_rows, verify_file
from .registry import source_spec

FILES = ("documents.csv", "qaps.csv", "third_party/wikipedia/summaries.csv")


def original_summary_tasks(data_root, *, catalog_root=None, manifest=None):
    """Yield original question/document/summary relations in publisher QA order.

    All three registered files are verified before reading and after exhaustion.
    Consumers stopping early have initial verification only. Payloads stay in
    memory; this function neither copies raw data nor claims native completion.
    """
    source = source_spec("NARRATIVEQA", catalog_root) if manifest is None else manifest
    if source.get("source_id", "NARRATIVEQA") != "NARRATIVEQA" or not source.get(
        "revision"
    ):
        raise ValueError("original NarrativeQA source identity required")
    specs = {item["path"]: item for item in source["files"]}
    if any(name not in specs for name in FILES):
        raise ValueError("all three pinned original NarrativeQA tables required")
    for name in FILES:
        verify_file(data_root, specs[name])

    def rows(name):
        return iter_rows(asset_path(data_root, name))

    def index(name, required):
        result = {}
        for row in rows(name):
            if any(
                not isinstance(row.get(key), str) or not row[key] for key in required
            ):
                raise ValueError("missing original NarrativeQA table field")
            key = row["document_id"]
            if key in result:
                raise ValueError("duplicate NarrativeQA document identity")
            result[key] = row
        return result

    documents = index("documents.csv", ("document_id", "set", "story_url"))
    summaries = index(FILES[2], ("document_id", "set", "summary"))
    for position, question in enumerate(rows("qaps.csv")):
        if any(
            not isinstance(question.get(key), str) or not question[key]
            for key in ("document_id", "set", "question", "answer1", "answer2")
        ):
            raise ValueError("missing original NarrativeQA question field")
        key = question["document_id"]
        if key not in documents or key not in summaries:
            raise ValueError("unresolved original NarrativeQA document relation")
        document, summary = documents[key], summaries[key]
        if question["set"] != document["set"] or question["set"] != summary["set"]:
            raise ValueError("NarrativeQA split relation mismatch")
        yield {
            "source_id": "NARRATIVEQA",
            "revision": source["revision"],
            "setting": "publisher_summary_not_full_story",
            "question_row_index": position,
            "input": {
                "document_id": key,
                "question": question["question"],
                "summary": summary["summary"],
            },
            "outcome": {"answer1": question["answer1"], "answer2": question["answer2"]},
            "original_relations": deepcopy(
                {"question": question, "document": document, "summary": summary}
            ),
            "native_execution_complete": False,
        }
    for name in FILES:
        verify_file(data_root, specs[name])


def relation_spec(source, spec):
    """Bind original three-table view, leaving custom single-table mappings alone."""
    if source.get("source_id") != "NARRATIVEQA" or not set(FILES).issubset(
        {item["path"] for item in source["files"]}
    ):
        return spec
    return {
        **spec,
        "original_relation_source": deepcopy(source),
        "mapping": "native_fields",
        "input_fields": ["document_id", "summary", "question"],
        "required_fields": ["document_id", "summary", "question", "answer1", "answer2"],
        "outcome_fields": [
            "answer1",
            "answer2",
            "answer1_tokenized",
            "answer2_tokenized",
        ],
        "annotation_fields": [
            "original_document",
            "original_summary_metadata",
            "original_question_metadata",
        ],
        "native_structured_fields": [],
        "id_field": "",
    }


def joined_rows(data_root, source):
    """Original fields distributed without references leaking into input envelopes."""
    for task in original_summary_tasks(data_root, manifest=source):
        original = task["original_relations"]
        question = original["question"]
        answers = {
            key: value for key, value in question.items() if key.startswith("answer")
        }
        yield {
            **task["input"],
            **answers,
            "original_document": original["document"],
            "original_summary_metadata": {
                key: value
                for key, value in original["summary"].items()
                if key != "summary"
            },
            "original_question_metadata": {
                key: value
                for key, value in question.items()
                if key != "question" and not key.startswith("answer")
            },
        }
