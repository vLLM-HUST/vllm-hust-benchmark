"""Original conversational TTS utterances; no invented API requests or timing."""

from copy import deepcopy
from pathlib import Path
import re

from .records import WorkloadRecord


def original_rows(path):
    path = Path(path)
    if path.name not in {"train_phone.txt", "val_phone.txt"}:
        raise ValueError("original DailyTalk phoneme split required")
    rows, turns, identities = [], {}, set()
    with path.open("rb") as stream:
        for offset, raw in enumerate(stream):
            original = raw.decode("utf-8")
            fields = original.rstrip("\r\n").split("|")
            if len(fields) != 5 or not all(fields):
                raise ValueError("five nonempty original DailyTalk fields required")
            identity, speaker, phonemes, text, emotion = fields
            match = re.fullmatch(r"(\d+)_(\d+)_d(\d+)", identity)
            if not match or identity in identities:
                raise ValueError(
                    "distinct original DailyTalk utterance identity required"
                )
            turn, speaker_index, dialogue = match.groups()
            turn = int(turn)
            if speaker != speaker_index or turn in turns.setdefault(dialogue, {}):
                raise ValueError("original speaker identity or dialogue turn differs")
            identities.add(identity)
            turns[dialogue][turn] = identity
            rows.append(
                {
                    "id": identity,
                    "speaker": speaker,
                    "phonemes": phonemes,
                    "text": text,
                    "emotion": emotion,
                    "dialogue_id": dialogue,
                    "turn_index": turn,
                    "split": path.name.removesuffix("_phone.txt"),
                    "original_row": original,
                    "metadata_row_index": offset,
                    "reference_audio": {
                        "relative_path": f"dailytalk/data/{dialogue}/{identity}.wav",
                        "base": "original_audio_root",
                        "format": "wav",
                        "local_media_verified": False,
                    },
                }
            )
    if not rows:
        raise ValueError("nonempty original DailyTalk split required")
    for row in rows:
        # Missing source turns remain gaps, never reconstructed utterances.
        row["previous_turn_id"] = turns[row["dialogue_id"]].get(row["turn_index"] - 1)
        yield row


def adapt_case(row, *, source, index):
    annotations = {
        key: deepcopy(row[key])
        for key in (
            "id",
            "dialogue_id",
            "turn_index",
            "split",
            "original_row",
            "metadata_row_index",
            "previous_turn_id",
        )
    }
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["id"],
            "identity_kind": "source_id",
            "session_id": "d" + row["dialogue_id"],
            "parent_task_id": row["previous_turn_id"],
            "input": {
                key: row[key] for key in ("phonemes", "text", "speaker", "emotion")
            },
        },
        runtime={
            "kind": "task_dataset",
            "execution": "not_executed",
            "requirements": [
                "Original conversational TTS runner and complete dialogue context",
                "Original reference audio is not generated output or implicit input",
                "No streaming, overlap or arrival timing is inferred",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "dailytalk_original_utterances",
            "source_row_index": index,
            "order": "original_file_order; dialogue_turn_index_not_arrival_time",
            "annotations": annotations,
            "unmapped_field_names": [],
            "synthetic_fallback": False,
            "normalization": {},
        },
        outcome={"reference_audio": deepcopy(row["reference_audio"])},
    )
