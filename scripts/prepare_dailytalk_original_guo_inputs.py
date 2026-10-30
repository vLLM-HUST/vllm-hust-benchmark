"""Prepare fixed whole-dialogue Guo inputs, not synthetic prompts or embeddings.

Original phoneme/reference rows and maps stay byte-identical. The output records
the publisher's ten-turn history policy; no neural model is loaded or executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/dailytalk-original-audio-sample.json"
MAPS = ("speakers.json", "emotions.json", "stats.json")


def select_rows(raw, selected):
    rows, seen = [], set()
    for line in raw.splitlines(keepends=True):
        fields = line.decode("utf-8").rstrip("\r\n").split("|")
        if fields[0] not in selected:
            continue
        if len(fields) != 5 or fields[0] in seen:
            raise ValueError("duplicate or malformed original utterance row")
        seen.add(fields[0])
        rows.append((line, fields))
    if seen != selected:
        raise ValueError("original selected utterance rows missing")
    return rows


def history_plan(rows):
    grouped = {}
    for _, fields in rows:
        identity = fields[0]
        turn, _, dialogue = identity.split("_")
        grouped.setdefault(dialogue, {})[int(turn)] = identity
    for turns in grouped.values():
        if set(turns) != set(range(len(turns))):
            raise ValueError("whole original dialogue history is not contiguous")
    items = []
    for raw, fields in rows:
        identity, speaker = fields[:2]
        turn, _, dialogue = identity.split("_")
        turn = int(turn)
        items.append(
            {
                "utterance_id": identity,
                "speaker": speaker,
                "original_row_sha256": hashlib.sha256(raw).hexdigest(),
                "feature_path": f"text_emb/{speaker}-text_emb-{identity}.npy",
                "history_ids": [
                    grouped[dialogue][i] for i in range(max(0, turn - 10), turn)
                ],
            }
        )
    return items


def prepare(args):
    profile = json.loads(PROFILE.read_text("utf-8"))
    report = {
        "source_id": "DAILYTALK",
        "status": "PLAN_ONLY_NO_DATA_READ",
        "fixed_dialogue_ids": profile["selection"]["dialogue_ids"],
        "history_window_turns": 10,
        "history_encoder_embedding_dimension": 512,
        "model_calls": 0,
        "embedding_arrays_created": 0,
        "native_synthesis_or_scoring_executed": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    metadata, audio, output = (
        Path(x).resolve() for x in (args.metadata_root, args.audio_root, args.output)
    )
    if output.is_relative_to(ROOT) or any(
        output.is_relative_to(x) or x.is_relative_to(output) for x in (metadata, audio)
    ):
        raise ValueError("separate external preparation output required")
    if args.action == "prepare" and output.exists():
        raise ValueError("fresh preparation output required; use verify for reuse")
    sys.path.insert(0, str(ROOT / "scripts"))
    from prepare_dailytalk_original_audio import prepare as verify_audio
    from prepare_dailytalk_original_audio import selected_ids
    from prepare_dailytalk_original_metadata import prepare as verify_metadata

    verify_metadata(SimpleNamespace(data_root=metadata, action="verify"))
    verify_audio(
        SimpleNamespace(data_root=audio, metadata_root=metadata, action="verify")
    )
    selected = selected_ids(metadata, profile)
    source = metadata / "preprocessed_data/DailyTalk"
    rows = select_rows((source / "val_phone.txt").read_bytes(), selected)
    features = history_plan(rows)
    artifacts = {"val_phone.fixed3.private.txt": b"".join(x[0] for x in rows)}
    for name in MAPS:
        artifacts[f"preprocessed_data/DailyTalk/{name}"] = (source / name).read_bytes()
    contract = {
        "source_id": "DAILYTALK",
        "utterances": features,
        "raw_path": str(audio / "dailytalk"),
        "sub_dir_name": "data",
        "preprocessed_path": str(output / "preprocessed_data/DailyTalk"),
        "feature_rule": "publisher encode([raw_text])[0] using original distiluse-base-multilingual-cased-v1; one original utterance at a time",
        "history_window_turns": 10,
        "feature_dimension": 512,
        "phonemes_recomputed": False,
        "embeddings_generated": False,
    }
    artifacts["guo-input-contract.private.json"] = json.dumps(
        contract, ensure_ascii=False, indent=2
    ).encode("utf-8")
    if args.action == "prepare":
        output.mkdir(mode=0o700, parents=True)
        for name, raw in artifacts.items():
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod(0o600)
    for name, raw in artifacts.items():
        if (output / name).read_bytes() != raw:
            raise ValueError(f"original preparation artifact differs: {name}")
    report.update(
        status="ORIGINAL_GUO_INPUTS_VERIFIED_EMBEDDING_AND_SYNTHESIS_PENDING",
        original_utterances=len(rows),
        original_maps_copied=len(MAPS),
        original_rows_byte_identical=True,
        original_history_relationships_preserved=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-root", "audio-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
