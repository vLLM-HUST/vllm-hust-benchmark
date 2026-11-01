"""Original memory-question relations; no prompt rendering, features or models."""

from copy import deepcopy
from pathlib import Path

from .assets import _json, asset_path, iter_rows, verify_file
from .records import WorkloadRecord
from .registry import read_catalog

CORE = (
    "questions.jsonl",
    "trajectories.jsonl",
    "haystacks/lme_v2_small.json",
    "haystacks/lme_v2_medium.json",
)
INPUT_FIELDS = {"id", "domain", "environment", "question_type", "question", "image"}
REFERENCE_FIELDS = {"answer", "eval_function"}


def filemap(spec):
    files = spec.get("relation_files")
    if files is None:
        files = read_catalog("longmemeval_v2_original_files_v1.json")["files"]
    result = {item["path"]: item for item in files}
    if len(result) != len(files) or any(name not in result for name in CORE):
        raise ValueError(
            "original memory question/trajectory/both haystack files required"
        )
    return result


def trajectory_index(path):
    index = {}
    with Path(path).open("rb") as stream:
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            row = _json(raw.decode("utf-8"))
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not row["id"]
                or row.get("domain") not in {"web", "enterprise"}
                or not isinstance(row.get("states"), list)
            ):
                raise ValueError("original trajectory identity/domain/states required")
            if row["id"] in index:
                raise ValueError("duplicate original trajectory identity")
            index[row["id"]] = (offset, len(raw), row["domain"])
    return index


def original_rows(path, spec):
    path = Path(path)
    if path.name != CORE[0]:
        raise ValueError("original questions.jsonl view required")
    root, files = path.parent, filemap(spec)
    for name in CORE:
        verify_file(root, files[name])
    index = trajectory_index(asset_path(root, CORE[1]))
    haystacks = {
        tier: _json(
            asset_path(root, f"haystacks/lme_v2_{tier}.json").read_text("utf-8")
        )
        for tier in ("small", "medium")
    }
    questions = list(iter_rows(path))
    identities = [row.get("id") for row in questions]
    if any(
        not isinstance(identity, str) or not identity for identity in identities
    ) or len(set(identities)) != len(identities):
        raise ValueError("unique original question identities required")
    if any(
        not isinstance(value, dict) or set(value) != set(identities)
        for value in haystacks.values()
    ):
        raise ValueError("original question/haystack identity relation differs")
    try:
        for question in questions:
            if set(question) != INPUT_FIELDS | REFERENCE_FIELDS or question.get(
                "domain"
            ) not in {"web", "enterprise"}:
                raise ValueError("original eight-field question required")
            ordered = {tier: value[question["id"]] for tier, value in haystacks.items()}
            for ids in ordered.values():
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(not isinstance(tid, str) for tid in ids)
                    or len(ids) != len(set(ids))
                    or any(
                        tid not in index or index[tid][2] != question["domain"]
                        for tid in ids
                    )
                ):
                    raise ValueError("unresolved or cross-domain original haystack")
            assets = [deepcopy(files[name]) for name in CORE]
            image = question["image"]
            if image is not None:
                if (
                    not isinstance(image, str)
                    or image not in files
                    or not image.startswith("question_screenshots/")
                ):
                    raise ValueError("original registered question image required")
                verify_file(root, files[image])
                assets.append(deepcopy(files[image]))
            archive = "trajectory_screenshots/" + (
                "web_screenshots.tar.gz"
                if question["domain"] == "web"
                else "enterprise_screenshots_base.tar.gz"
            )
            if archive not in files:
                raise ValueError(
                    "original trajectory screenshot archive relation missing"
                )
            assets.append(deepcopy(files[archive]))
            yield {
                "original_question": question,
                "original_haystacks": ordered,
                "native_files": assets,
            }
    finally:
        for name in CORE:
            verify_file(root, files[name])


def adapt_case(row, *, source, index):
    if source.get("source_id") != "LONGMEMEVAL_V2" or set(row) != {
        "original_question",
        "original_haystacks",
        "native_files",
    }:
        raise ValueError("original joined memory task required")
    question = row["original_question"]
    if set(question) != INPUT_FIELDS | REFERENCE_FIELDS:
        raise ValueError("original question field separation required")
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": question["id"],
            "identity_kind": "original_question_id",
            "native_unit": "memory_question_with_ordered_trajectory_haystacks",
            "input": {
                "original_question": {
                    name: deepcopy(question[name]) for name in INPUT_FIELDS
                },
                "haystacks": deepcopy(row["original_haystacks"]),
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "native_files": deepcopy(row["native_files"]),
            "trajectory_media_availability": "archive_identity_registered_not_reverified_by_metadata_join",
            "requirements": [
                "Original data root, selected memory method, reader/embedding services and original evaluator required",
                "Historical trajectory actions/thoughts/outcomes remain original memory data, not new model outputs",
                "Question answers/evaluator are references, not model input; no prompt flattening",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "longmemeval_v2_original_relations",
            "source_row_index": index,
            "order": "original_question_order_and_both_original_haystack_orders",
            "synthetic_fallback": False,
        },
        outcome={
            "references": {name: deepcopy(question[name]) for name in REFERENCE_FIELDS}
        },
    )


def restore_original_question(record):
    if record.source.get("source_id") != "LONGMEMEVAL_V2":
        raise ValueError("original memory question record required")
    inputs, references = (
        record.task["input"]["original_question"],
        record.outcome["references"],
    )
    if (
        set(inputs) != INPUT_FIELDS
        or set(references) != REFERENCE_FIELDS
        or inputs["id"] != record.task["task_id"]
    ):
        raise ValueError("original question identity/reference separation lost")
    return deepcopy({**inputs, **references})


def iter_haystack_trajectories(record, *, tier, data_root):
    if tier not in {"small", "medium"}:
        raise ValueError("original small or medium tier required")
    root, question = Path(data_root), restore_original_question(record)
    files = {item["path"]: item for item in record.runtime["native_files"]}
    for name in CORE:
        verify_file(root, files[name])
    original = next(
        (
            row
            for row in iter_rows(asset_path(root, CORE[0]))
            if row["id"] == question["id"]
        ),
        None,
    )
    if original != question:
        raise ValueError("original question no longer matches record")
    ids = _json(asset_path(root, f"haystacks/lme_v2_{tier}.json").read_text("utf-8"))[
        question["id"]
    ]
    if ids != record.task["input"]["haystacks"][tier]:
        raise ValueError("original haystack order changed")
    path = asset_path(root, CORE[1])
    index = trajectory_index(path)
    try:
        with path.open("rb") as stream:
            for tid in ids:
                offset, size, domain = index[tid]
                if domain != question["domain"]:
                    raise ValueError("original memory domain mismatch")
                stream.seek(offset)
                row = _json(stream.read(size).decode("utf-8"))
                if row["id"] != tid:
                    raise ValueError("original trajectory index changed")
                yield row
    finally:
        for name in CORE:
            verify_file(root, files[name])
