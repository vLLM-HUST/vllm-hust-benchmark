"""Prepare original complete-dialogue Guo inputs; fixed first3 remain the default.

Original phoneme/reference rows and maps stay byte-identical. The output records
the publisher's ten-turn history policy; no neural model is loaded or executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/dailytalk-original-audio-sample.json"
MAPS = ("speakers.json", "emotions.json", "stats.json")


def cohort_rows(raw, dialogue_ids=None):
    """Select complete original dialogues, retaining source bytes and ordering."""
    all_ids = [
        line.decode("utf-8").rstrip("\r\n").split("|", 1)[0]
        for line in raw.splitlines(keepends=True)
    ]
    if any(not re.fullmatch(r"\d+_\d+_d\d+", identity) for identity in all_ids):
        raise ValueError("original utterance identities required")
    available = {identity.rsplit("_d", 1)[1] for identity in all_ids}
    if dialogue_ids is None:
        chosen = available
    else:
        chosen = set(dialogue_ids)
        if len(chosen) != len(dialogue_ids) or not chosen or not chosen <= available:
            raise ValueError(
                "distinct complete original dialogue IDs from this split required"
            )
    selected = {
        identity for identity in all_ids if identity.rsplit("_d", 1)[1] in chosen
    }
    rows = select_rows(raw, selected)
    history_plan(rows)
    return rows


def verify_cohort_audio(audio, rows, profile, original_archive=None):
    """Verify real source directory entries, never make empty WAV placeholders."""
    from prepare_dailytalk_original_audio import verify_bytes

    expected = {}
    for _, fields in rows:
        identity = fields[0]
        dialogue = identity.rsplit("_d", 1)[1]
        expected.setdefault(dialogue, set()).add(f"{identity}.wav")
    known = {item["path"]: item for item in profile["files"]}
    members = [
        f"dailytalk/data/{dialogue}/{name}"
        for dialogue, names in expected.items()
        for name in sorted(names)
    ]
    for dialogue, names in expected.items():
        directory = audio / "dailytalk/data" / dialogue
        if not directory.resolve().is_relative_to(audio.resolve()):
            raise ValueError("original dialogue audio directory escapes explicit root")
        if {path.name for path in directory.glob("*.wav")} != names:
            raise ValueError(
                "complete original audio directory entries required; no missing or extra turns"
            )
    if all(name in known for name in members):
        for name in members:
            path = audio / name
            if not path.resolve(strict=True).is_relative_to(audio.resolve()):
                raise ValueError("original reference audio escapes explicit root")
            verify_bytes(path.read_bytes(), known[name])
        return "publisher-pinned-selected-member-sha256-and-crc"
    if original_archive is None:
        raise ValueError(
            "provide the original publisher ZIP and extracted original audio for other dialogues"
        )
    archive_path = Path(original_archive)
    if archive_path.stat().st_size != profile["archive"]["bytes"]:
        raise ValueError("original publisher archive size differs")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("duplicate publisher archive members")
        for dialogue, expected_names in expected.items():
            prefix = f"dailytalk/data/{dialogue}/"
            original_names = {
                name[len(prefix) :]
                for name in names
                if name.startswith(prefix) and name.endswith(".wav")
            }
            if original_names != expected_names:
                raise ValueError("original archive/dialogue phoneme turn sets differ")
        for name in members:
            path = audio / name
            if (
                not path.resolve(strict=True).is_relative_to(audio.resolve())
                or not path.is_file()
            ):
                raise ValueError("contained original audio file required")
            info = archive.getinfo(name)
            if (
                info.is_dir()
                or stat.S_ISLNK(info.external_attr >> 16)
                or info.flag_bits & 1
                or path.stat().st_size != info.file_size
            ):
                raise ValueError("original archive member type or size differs")
            with archive.open(info) as original, path.open("rb") as actual:
                while chunk := original.read(1024**2):
                    if actual.read(len(chunk)) != chunk:
                        raise ValueError(
                            "extracted audio differs from original archive member"
                        )
                if actual.read(1):
                    raise ValueError("extra extracted audio bytes")
    return "provided-original-publisher-zip-member-bytes-and-crc-not-published-full-archive-digest"


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
    split = getattr(args, "split", "val")
    dialogue_ids = getattr(args, "dialogue_ids", None)
    all_dialogues = getattr(args, "all_dialogues", False)
    general = dialogue_ids is not None or all_dialogues
    if split not in {"train", "val"} or (dialogue_ids is not None and all_dialogues):
        raise ValueError(
            "one original split and one complete-dialogue selection required"
        )
    if not general and split != "val":
        raise ValueError("choose explicit train dialogue IDs or all-dialogues")
    if general:
        report.update(
            fixed_dialogue_ids=None,
            selected_dialogue_ids=dialogue_ids,
            all_dialogues=all_dialogues,
            original_split=split,
        )
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
    source = metadata / "preprocessed_data/DailyTalk"
    if general:
        rows = cohort_rows((source / f"{split}_phone.txt").read_bytes(), dialogue_ids)
        audio_verification = verify_cohort_audio(
            audio, rows, profile, getattr(args, "original_audio_archive", None)
        )
        source_name = f"{split}_phone.selected.private.txt"
    else:
        verify_audio(
            SimpleNamespace(data_root=audio, metadata_root=metadata, action="verify")
        )
        selected = selected_ids(metadata, profile)
        rows = select_rows((source / "val_phone.txt").read_bytes(), selected)
        audio_verification = "publisher-pinned-selected-member-sha256-and-crc"
        source_name = "val_phone.fixed3.private.txt"
    features = history_plan(rows)
    artifacts = {source_name: b"".join(x[0] for x in rows)}
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
        prepared_source_file=source_name,
        audio_verification=audio_verification,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metadata-root", "audio-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--dialogue-ids", nargs="+")
    selection.add_argument("--all-dialogues", action="store_true")
    parser.add_argument(
        "--original-audio-archive",
        type=Path,
        help="Original publisher ZIP for non-pinned dialogue media verification; extract genuine audio separately",
    )
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
