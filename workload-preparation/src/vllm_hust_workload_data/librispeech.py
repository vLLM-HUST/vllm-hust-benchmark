"""Original OpenSLR utterance/transcript alignment, not synthetic speech requests."""

from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile

from .assets import asset_path, sha256
from .registry import source_spec


ARCHIVES = {"test-clean.tar.gz": "test-clean", "test-other.tar.gz": "test-other"}


def original_archive_rows(path):
    path = Path(path)
    if path.name not in ARCHIVES:
        raise ValueError("explicit original OpenSLR test archive required")
    split = ARCHIVES[path.name]
    audio, transcripts, seen = {}, {}, set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "LibriSpeech"
            ):
                raise ValueError("original archive member escapes corpus root")
            if member.name in seen or not (member.isfile() or member.isdir()):
                raise ValueError("duplicate or nonregular original archive member")
            seen.add(member.name)
            if member.isdir():
                continue
            if name.suffix == ".flac":
                if len(name.parts) != 5 or name.parts[1] != split:
                    raise ValueError("original utterance split/path mismatch")
                uid = name.stem
                pieces = uid.split("-")
                if (
                    len(pieces) != 3
                    or any(
                        not piece.isascii() or not piece.isdecimal() for piece in pieces
                    )
                    or pieces[:2] != list(name.parts[2:4])
                    or uid in audio
                ):
                    raise ValueError("original utterance identity/path mismatch")
                audio[uid] = str(name)
            elif name.name.endswith(".trans.txt"):
                if len(name.parts) != 5 or name.parts[1] != split:
                    raise ValueError("original transcript split/path mismatch")
                with archive.extractfile(member) as stream:
                    lines = stream.read().decode("utf-8").splitlines()
                for index, line in enumerate(lines):
                    if " " not in line:
                        raise ValueError(
                            "original transcript ID/text separator missing"
                        )
                    uid, text = line.split(" ", 1)
                    if (
                        uid in transcripts
                        or not text
                        or uid.split("-")[:2] != list(name.parts[2:4])
                    ):
                        raise ValueError(
                            "original transcript identity or content mismatch"
                        )
                    transcripts[uid] = (text, str(name), index)
    if not audio or audio.keys() != transcripts.keys():
        raise ValueError(
            "every original FLAC must have exactly one original transcript"
        )
    # Match the original TorchAudio LibriSpeech walker's ID ordering. This is
    # dataset iteration, not recorded request arrival or a fabricated timeline.
    for uid in sorted(audio):
        text, member, index = transcripts[uid]
        speaker, chapter, _ = uid.split("-")
        yield {
            "id": uid,
            "audio": {
                "archive_file": path.name,
                "archive_member": audio[uid],
                "format": "flac",
            },
            "text": text,
            "speaker_id": speaker,
            "chapter_id": chapter,
            "file": audio[uid],
            "transcript_member": member,
            "transcript_line_index": index,
        }


def read_original_audio(record, *, data_root):
    """Return the pinned original FLAC bytes, not a rewritten waveform or prompt."""
    if record.source.get("source_id") != "LIBRISPEECH":
        raise ValueError("original LibriSpeech record required")
    path = asset_path(Path(data_root), record.source["file"])
    if sha256(path) != record.source["file_sha256"]:
        raise ValueError("original audio archive identity mismatch")
    if path.name not in ARCHIVES:
        raise ValueError("original OpenSLR archive required for this accessor")
    uid = record.task["task_id"]
    pieces = uid.split("-")
    if len(pieces) != 3 or any(
        not piece.isascii() or not piece.isdecimal() for piece in pieces
    ):
        raise ValueError("original LibriSpeech utterance ID required")
    expected = f"LibriSpeech/{ARCHIVES[path.name]}/{pieces[0]}/{pieces[1]}/{uid}.flac"
    binding = record.task["input"]["audio"]
    if (
        binding.get("archive_file") != path.name
        or binding.get("archive_member") != expected
    ):
        raise ValueError("original utterance/media binding mismatch")
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember(expected)
        if not member.isfile():
            raise ValueError("regular original FLAC member required")
        with archive.extractfile(member) as stream:
            return stream.read()


def prepare_original_archives(*, data_root, output):
    """Unpack publisher bytes once for batch use; keep a private member hash index."""
    source = source_spec("LIBRISPEECH")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from archive root required")
    assets = [file for file in source["files"] if file["role"] == "data"]
    if {file["path"] for file in assets} != set(ARCHIVES):
        raise ValueError("both original test archives must be pinned")
    for asset in assets:
        archive_path = asset_path(data_root, asset["path"])
        if (
            archive_path.stat().st_size != asset["bytes"]
            or sha256(archive_path) != asset["sha256"]
        ):
            raise ValueError("original publisher archive identity mismatch")
        if sum(1 for _ in original_archive_rows(archive_path)) != asset["rows"]:
            raise ValueError("original utterance count mismatch")
    output.mkdir(parents=True, mode=0o700)
    manifest = {"schema_version": "openslr-original-preparation/v1", "archives": {}}
    for asset in assets:
        members = {}
        split = ARCHIVES[asset["path"]]
        with tarfile.open(asset_path(data_root, asset["path"]), "r|gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                if (
                    name.is_absolute()
                    or ".." in name.parts
                    or not name.parts
                    or name.parts[0] != "LibriSpeech"
                ):
                    raise ValueError("original member escapes preparation root")
                target = (output / split / str(name)).resolve()
                if not target.is_relative_to(output) or not (
                    member.isdir() or member.isfile()
                ):
                    raise ValueError("regular contained original member required")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with (
                    archive.extractfile(member) as stream,
                    target.open("xb") as destination,
                ):
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        destination.write(chunk)
                        digest.update(chunk)
                if target.stat().st_size != member.size:
                    raise ValueError("original member size mismatch")
                members[str(name)] = {
                    "sha256": digest.hexdigest(),
                    "bytes": member.size,
                }
        manifest["archives"][asset["path"]] = {
            "sha256": asset["sha256"],
            "split": split,
            "rows": asset["rows"],
            "members": members,
        }
    (output / "preparation.private.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return {
        "source_id": "LIBRISPEECH",
        "original_utterances": sum(file["rows"] for file in assets),
        "output": str(output),
        "waveform_or_transcript_rewritten": False,
        "native_asr_or_scoring_verified": False,
    }


def prepared_audio_path(record, *, prepared_root, preparation=None):
    """Avoid gzip rescanning; callers may preload the private manifest for batches."""
    root = Path(prepared_root).resolve()
    manifest = preparation
    if manifest is None:
        manifest = json.loads(
            asset_path(root, "preparation.private.json").read_text(encoding="utf-8")
        )
    if (
        manifest.get("schema_version") != "openslr-original-preparation/v1"
        or record.source.get("source_id") != "LIBRISPEECH"
    ):
        raise ValueError("original LibriSpeech prepared binding required")
    binding = record.task["input"]["audio"]
    name = Path(record.source["file"]).name
    archive = manifest["archives"][name]
    if (
        archive["sha256"] != record.source["file_sha256"]
        or binding.get("archive_file") != name
    ):
        raise ValueError("prepared archive identity differs from source reference")
    uid = record.task["task_id"]
    pieces = uid.split("-")
    if (
        name not in ARCHIVES
        or len(pieces) != 3
        or any(not piece.isascii() or not piece.isdecimal() for piece in pieces)
    ):
        raise ValueError("prepared original utterance identity mismatch")
    expected = f"LibriSpeech/{ARCHIVES[name]}/{pieces[0]}/{pieces[1]}/{uid}.flac"
    if binding.get("archive_member") != expected or archive["split"] != ARCHIVES[name]:
        raise ValueError("prepared original utterance identity mismatch")
    spec = archive["members"][expected]
    path = asset_path(root, archive["split"] + "/" + expected)
    if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
        raise ValueError("prepared original audio bytes changed")
    return path
