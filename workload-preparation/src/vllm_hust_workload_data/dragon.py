"""Original same-release RAG task relations; no retrieval, prompts or research features."""

import ast
from copy import deepcopy
from pathlib import Path

from .assets import asset_path, iter_rows, verify_file
from .records import WorkloadRecord
from .registry import source_spec

KINDS = ("public_questions", "public_texts", "reference_qa", "reference_texts")
FIELDS = (
    {"id", "question"},
    {"id", "text"},
    {"id", "public_id", "text_ids", "text", "url", "question", "answer", "type"},
    {"id", "public_id", "text"},
)


def keyed(rows, fields, key="id"):
    result = {}
    for row in rows:
        if set(row) != fields or type(row.get(key)) is not int or row[key] in result:
            raise ValueError("DRAGOn original fields or unique integer IDs differ")
        result[row[key]] = row
    return result


def original_rows(path):
    path = Path(path).resolve(strict=True)
    root = path.parents[2]
    spec = source_spec("DRAGON_PUBLIC")
    expected = {kind: f"{kind}/data/train-00000-of-00001.parquet" for kind in KINDS}
    files = {row["path"]: row for row in spec["files"]}
    if set(files) != set(expected.values()) or path != asset_path(
        root, expected[KINDS[0]]
    ):
        raise ValueError("DRAGOn requires the complete fixed four-way release")
    for item in files.values():
        verify_file(root, item)
    try:
        values = [list(iter_rows(asset_path(root, expected[kind]))) for kind in KINDS]
        questions, texts, references, reference_texts = values
        question_ids = keyed(questions, FIELDS[0])
        text_ids = keyed(texts, FIELDS[1])
        keyed(references, FIELDS[2])
        keyed(reference_texts, FIELDS[3])
        qa_by_public = keyed(references, FIELDS[2], "public_id")
        text_by_public = keyed(reference_texts, FIELDS[3], "public_id")
        private_text_ids = {row["id"] for row in reference_texts}
        if set(qa_by_public) != set(question_ids) or set(text_by_public) != set(
            text_ids
        ):
            raise ValueError("DRAGOn original public/reference ID coverage differs")
        if any(
            row["text"] != text_ids[key]["text"] for key, row in text_by_public.items()
        ):
            raise ValueError("DRAGOn original public/reference corpus text differs")
        for question in questions:
            reference = qa_by_public[question["id"]]
            if reference["question"] != question["question"]:
                raise ValueError("DRAGOn original public/reference question differs")
            linked = ast.literal_eval(reference["text_ids"])
            if not isinstance(linked, list) or not linked:
                raise ValueError("DRAGOn original nonempty reference list required")
            flattened = []
            for value in linked:
                flattened.extend(value if isinstance(value, list) else [value])
            if not flattened or any(
                type(value) not in (str, int) or int(value) not in private_text_ids
                for value in flattened
            ):
                raise ValueError("DRAGOn original reference text ID missing")
        for offset, question in enumerate(questions):
            yield {
                "original_question": deepcopy(question),
                "original_reference_qa": deepcopy(qa_by_public[question["id"]]),
                "original_query_offset": offset,
                "corpus_file": expected["public_texts"],
                "reference_text_mapping_file": expected["reference_texts"],
                "release_tag": "1.15.0",
                "asset_references": [
                    {
                        key: item[key]
                        for key in (
                            "path",
                            "sha256",
                            "publisher_repo",
                            "publisher_revision",
                        )
                        if key in item
                    }
                    for item in files.values()
                ],
            }
    finally:
        for item in files.values():
            verify_file(root, item)


def adapt_case(row, *, source, index):
    original = deepcopy(row["original_question"])
    reference = deepcopy(row["original_reference_qa"])
    source = deepcopy(source)
    question_asset = next(
        item
        for item in row["asset_references"]
        if item["path"].startswith("public_questions/")
    )
    source.update(
        protocol_revision=source["revision"],
        dataset_repo=question_asset.get("publisher_repo"),
        dataset_revision=question_asset.get("publisher_revision"),
    )
    if (
        set(original) != FIELDS[0]
        or set(reference) != FIELDS[2]
        or type(original["id"]) is not int
        or reference["public_id"] != original["id"]
        or reference["question"] != original["question"]
    ):
        raise ValueError(
            "DRAGOn requires unchanged original question/reference identity"
        )
    return WorkloadRecord(
        source=source,
        task={
            "task_id": str(original["id"]),
            "identity_kind": "original_public_question_id",
            "native_unit": "original_same_release_rag_task",
            "input": original,
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Original full corpus, author retrieval/generation and scorer required; not a ready model prompt",
                "Reference QA and private/public text mapping are scorer-side only; no production update/delete events inferred",
            ],
            "corpus_file": row["corpus_file"],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "dragon_original_tasks",
            "source_row_index": index,
            "order": "original_publisher_file_order_not_request_arrival",
            "annotations": {
                key: row[key] for key in ("release_tag", "original_query_offset")
            },
            "asset_references": deepcopy(row["asset_references"]),
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "original_reference_qa": reference,
            "reference_text_mapping_file": row["reference_text_mapping_file"],
            "reference_kind": "original_same_release_qa_and_external_corpus_id_mapping_not_observed_output",
        },
    )
