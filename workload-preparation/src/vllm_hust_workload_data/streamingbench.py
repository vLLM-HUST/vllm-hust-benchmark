"""Lossless original video groups; no video decoding, prompt creation or models."""

from copy import deepcopy
from pathlib import Path

from .assets import _json
from .records import WorkloadRecord

GROUP_INPUT = {"time", "video_path", "video_categories"}
GROUP_REFERENCE = {"captions"}
QUESTION_INPUT = {
    "task_type",
    "question",
    "time_stamp",
    "options",
    "required_ability",
    "audio_path",
    "time",
}
QUESTION_REFERENCE = {
    "answer",
    "ground_truth_time_stamp",
    "ground_truth_output",
    "cause",
    "effect",
}


def original_rows(path):
    rows = _json(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("original StreamingBench top-level array required")
    for row in rows:
        # SQA's outer row is a whole ordered sequence, not separate questions.
        if not isinstance(row, (dict, list)):
            raise ValueError("original StreamingBench video group required")
        yield {"original_group": row}


def adapt_case(row, *, source, index):
    if source.get("source_id") != "STREAMINGBENCH" or set(row) != {"original_group"}:
        raise ValueError("original StreamingBench group envelope required")
    original = row["original_group"]
    groups = original if isinstance(original, list) else [original]
    if not groups:
        raise ValueError("complete nonempty original sequence required")
    inputs, references = [], []
    for group in groups:
        if (
            not isinstance(group, dict)
            or set(group) - GROUP_INPUT - GROUP_REFERENCE - {"questions"}
            or not isinstance(group.get("video_path"), str)
            or not group["video_path"]
            or not isinstance(group.get("questions"), list)
            or not group["questions"]
        ):
            raise ValueError("unrecognized or incomplete original video group")
        inp = {k: deepcopy(v) for k, v in group.items() if k in GROUP_INPUT}
        ref = {k: deepcopy(v) for k, v in group.items() if k in GROUP_REFERENCE}
        inp["questions"], ref["questions"] = [], []
        for question in group["questions"]:
            if (
                not isinstance(question, dict)
                or set(question) - QUESTION_INPUT - QUESTION_REFERENCE
                or any(
                    not isinstance(question.get(k), str)
                    for k in ("question", "time_stamp")
                )
            ):
                raise ValueError("unrecognized or incomplete original question")
            inp["questions"].append(
                {k: deepcopy(v) for k, v in question.items() if k in QUESTION_INPUT}
            )
            ref["questions"].append(
                {k: deepcopy(v) for k, v in question.items() if k in QUESTION_REFERENCE}
            )
        inputs.append(inp)
        references.append(ref)
    return WorkloadRecord(
        source=deepcopy(source),
        task={"task_id": index, "input": {"groups": inputs}},
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Restore complete original groups for the author runner; not ready chat requests",
                "Preserve original timestamps, media and all sequential/proactive stages",
                "Author SQA uses previous reference answers as context; native runner supplies that protocol",
                "Offline, stream and proactive subset files overlap; do not count them as disjoint tasks",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "streamingbench_original_groups",
            "source_row_index": index,
            "original_group_is_list": isinstance(original, list),
            "order": "original_file_and_nested_question_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={"references": {"groups": references}},
    )


def restore_original_case(record):
    if record.source.get("source_id") != "STREAMINGBENCH":
        raise ValueError("original StreamingBench record required")
    groups = deepcopy(record.task["input"]["groups"])
    for group, reference in zip(
        groups, record.outcome["references"]["groups"], strict=True
    ):
        labels = deepcopy(reference)
        for question, ref in zip(
            group["questions"], labels.pop("questions"), strict=True
        ):
            if set(question) & set(ref):
                raise ValueError("question input/reference overlap")
            question.update(ref)
        if set(group) & set(labels):
            raise ValueError("group input/reference overlap")
        group.update(labels)
    if record.replay["original_group_is_list"]:
        return groups
    if len(groups) != 1:
        raise ValueError("original single-group shape changed")
    return groups[0]
