"""Complete original spoken dialogues, never flattened chat requests or audio."""

from copy import deepcopy
from pathlib import Path

from .assets import _json
from .records import WorkloadRecord

INPUT_FIELDS = {"tag", "text", "words"}
REFERENCE_FIELDS = {"metadata", "dialog_act", "span_info"}


def original_rows(path):
    data = _json(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("original dialogue-ID keyed object required")
    for identity, dialogue in data.items():
        yield {"dialogue_id": identity, "original_dialogue": dialogue}


def adapt_case(row, *, source, index):
    if source.get("source_id") != "SPOKENWOZ" or set(row) != {
        "dialogue_id",
        "original_dialogue",
    }:
        raise ValueError("original SpokenWOZ envelope required")
    identity, original = row["dialogue_id"], row["original_dialogue"]
    if (
        not isinstance(identity, str)
        or not identity
        or not isinstance(original, dict)
        or set(original) != {"goal", "log"}
        or not isinstance(original["goal"], dict)
        or not isinstance(original["log"], list)
        or not original["log"]
        or len(original["log"]) % 2
    ):
        raise ValueError("complete original user/system dialogue required")
    inputs, references = [], []
    for position, turn in enumerate(original["log"]):
        if (
            not isinstance(turn, dict)
            or set(turn) != INPUT_FIELDS | REFERENCE_FIELDS
            or turn["tag"] != ("user" if position % 2 == 0 else "system")
            or not isinstance(turn["text"], str)
            or not isinstance(turn["words"], list)
            or not isinstance(turn["metadata"], dict)
            or not isinstance(turn["dialog_act"], dict)
            or not isinstance(turn["span_info"], list)
        ):
            raise ValueError("unrecognized original turn contract")
        inputs.append({k: deepcopy(v) for k, v in turn.items() if k in INPUT_FIELDS})
        references.append(
            {k: deepcopy(v) for k, v in turn.items() if k in REFERENCE_FIELDS}
        )
    return WorkloadRecord(
        source=deepcopy(source),
        task={"task_id": identity, "input": {"turns": inputs}},
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Complete stored dialogue is a task bundle, not a ready model prompt",
                "Native runner must restrict each call to its original visible history",
                "Goal/state/act/span labels are references, not automatic model input",
                "Original words/timestamps are not acquired audio or observed engine arrivals",
                "Select and run an explicit official DST/response/speech protocol separately",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "spokenwoz_original_dialogues",
            "source_row_index": index,
            "synthetic_fallback": False,
            "order": "original_dialogue_insertion_and_complete_turn_order",
        },
        outcome={
            "references": {"goal": deepcopy(original["goal"]), "turns": references}
        },
    )


def restore_original_case(record):
    if record.source.get("source_id") != "SPOKENWOZ":
        raise ValueError("original SpokenWOZ record required")
    turns = deepcopy(record.task["input"]["turns"])
    references = record.outcome["references"]
    for turn, labels in zip(turns, references["turns"], strict=True):
        if set(turn) & set(labels):
            raise ValueError("original input/reference fields overlap")
        turn.update(deepcopy(labels))
    return record.task["task_id"], {"goal": deepcopy(references["goal"]), "log": turns}
