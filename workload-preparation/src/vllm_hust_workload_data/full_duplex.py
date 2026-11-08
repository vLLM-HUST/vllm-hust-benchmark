"""Original audio/annotation relationships; no model requests or speech inference."""

from copy import deepcopy
from pathlib import Path, PurePosixPath
import json
import zipfile

from .assets import asset_path, verify_file
from .records import WorkloadRecord


def relation_spec(source, spec):
    if (
        source.get("source_id") != "FULL_DUPLEX_BENCH"
        or source.get("revision") != spec["source_revision"]
    ):
        raise ValueError("original full-duplex archive revision mismatch")
    return {**spec, "original_archive_source": deepcopy(source)}


def original_rows(path, spec):
    path = Path(path)
    relative = "/".join(path.parts[-2:])
    assets = {x["path"]: x for x in spec["original_archive_source"]["files"]}
    if relative not in assets or assets[relative].get("role") != "data":
        raise ValueError("registered original audio archive required")
    yield from original_cases(path, assets[relative])


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["task_id"],
            "identity_kind": "original_archive_scoped_case_directory",
            "native_unit": "full_duplex_audio_case",
            **{
                key: row[key]
                for key in [
                    "original_case_id",
                    "original_case_directory",
                    "version",
                    "family",
                ]
            },
            "input": deepcopy(row["input"]),
        },
        runtime={
            "kind": "full_duplex_audio_case",
            "execution": "not_executed",
            "requirements": [
                "Original time-synchronous speech model,transcription and native evaluation remain required",
                "Paired clean input is a separate control,not a transcript or prompt substitute",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "full_duplex_original_archives",
            "source_row_index": index,
            "order": "original_zip_input_member_order_not_arrival_time",
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            **deepcopy(row["outcome"]),
            "reference_kind": "original_annotations_and_related_media_not_model_outputs",
        },
    )


V1 = {
    "candor_pause_handling": {"transcription.json", "pause.json"},
    "candor_turn_taking": {"turn_taking.json"},
    "icc_backchannel": {"transcription.json"},
    "synthetic_pause_handling": {"transcription.json", "pause.json"},
    "synthetic_user_interruption": {"interrupt.json", "context.wav", "interrupt.wav"},
}
V15 = {
    "background_speech": "background.wav",
    "talking_to_other": "current_turn.wav",
    "user_backchannel": "backchannel.wav",
    "user_interruption": "interrupt.wav",
}


def original_cases(path, asset):
    """Preserve original ZIP input-member order and every case JSON field."""
    path = Path(path)
    relative = PurePosixPath(asset["path"])
    if len(relative.parts) != 2 or relative.suffix != ".zip":
        raise ValueError("explicit original version/archive required")
    version = relative.parts[0]
    family = relative.stem
    if version == "v1.0" and family in V1:
        required = {"input.wav"} | V1[family]
    elif version == "v1.5" and family in V15:
        required = {
            "input.wav",
            "clean_input.wav",
            "context.wav",
            V15[family],
            "metadata.json",
            "input.json",
            "clean_input.json",
        }
    else:
        raise ValueError("registered original full-duplex archive family required")
    root = path.parents[1]
    if path.resolve() != asset_path(root, asset["path"]).resolve():
        raise ValueError("original version/archive binding differs")
    verify_file(root, asset)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if sum(x.file_size for x in entries) > 1024**3:
                raise ValueError("original archive uncompressed budget exceeded")
            files = {}
            order = []
            for entry in entries:
                name = entry.orig_filename
                member = PurePosixPath(name)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or "\\" in name
                    or ":" in name
                    or "\x00" in name
                    or name != entry.filename
                    or entry.flag_bits & 1
                    or (entry.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("unsafe original ZIP member")
                if entry.is_dir() or name.startswith("__MACOSX/"):
                    continue
                if name in files:
                    raise ValueError("duplicate original ZIP member")
                files[name] = entry
                if member.name == "input.wav":
                    if len(member.parts) != 3 or member.parts[0] != family:
                        raise ValueError("original case/input hierarchy differs")
                    order.append(str(member.parent))
            if not order or len(order) != len(set(order)):
                raise ValueError("distinct original audio cases required")
            if asset.get("rows") is not None and len(order) != asset["rows"]:
                raise ValueError("frozen observed case count differs")

            def reference(name):
                entry = files[name]
                return {
                    "archive_file": asset["path"],
                    "archive_sha256": asset["sha256"],
                    "archive_member": name,
                    "bytes": entry.file_size,
                    "zip_crc32": entry.CRC,
                }

            global_names = [
                name
                for name in files
                if not any(name.startswith(parent + "/") for parent in order)
            ]
            for parent in order:
                related = {
                    name.removeprefix(parent + "/"): name
                    for name in files
                    if name.startswith(parent + "/")
                }
                if not required <= related.keys():
                    raise ValueError(
                        "complete original case media/annotations required"
                    )
                annotations = {}
                for leaf, name in related.items():
                    if leaf.endswith(".json"):
                        if files[name].file_size > 8 * 1024**2:
                            raise ValueError("original annotation byte budget exceeded")
                        annotations[leaf] = json.loads(archive.read(name))
                yield {
                    "task_id": version + "/" + parent,
                    "original_case_id": PurePosixPath(parent).name,
                    "original_case_directory": parent,
                    "version": version,
                    "family": family,
                    "input": {"audio_stream": reference(related["input.wav"])},
                    "outcome": {
                        "original_annotations": annotations,
                        "annotation_assets": {
                            leaf: reference(name)
                            for leaf, name in related.items()
                            if leaf.endswith(".json")
                        },
                        "related_media": {
                            leaf: {
                                **reference(name),
                                "native_role": "paired_control_input"
                                if leaf == "clean_input.wav"
                                else "original_related_audio_component",
                            }
                            for leaf, name in related.items()
                            if leaf.endswith(".wav") and leaf != "input.wav"
                        },
                        "other_case_assets": {
                            leaf: reference(name)
                            for leaf, name in related.items()
                            if not leaf.endswith((".json", ".wav"))
                        },
                        "archive_level_assets": [reference(x) for x in global_names],
                    },
                    "native_execution_complete": False,
                }
    finally:
        verify_file(root, asset)
