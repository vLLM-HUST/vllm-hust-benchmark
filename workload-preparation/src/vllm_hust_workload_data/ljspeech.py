"""Original LJSpeech text/reference-audio pairs; no synthesis or scoring."""

import csv
import io
from pathlib import Path, PurePosixPath
import re
import tarfile


def original_rows(path):
    path = Path(path)
    if path.name != "LJSpeech-1.1.tar.bz2":
        raise ValueError("original LJSpeech-1.1 archive required")
    seen, audio, metadata = set(), {}, None
    # Streaming avoids extraction and retaining waveforms in memory.
    with tarfile.open(path, "r|bz2") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not name.parts
                or name.is_absolute()
                or ".." in name.parts
                or name.parts[0] != "LJSpeech-1.1"
                or member.name in seen
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("unsafe or duplicate original archive member")
            seen.add(member.name)
            if member.isdir():
                continue
            if str(name) == "LJSpeech-1.1/metadata.csv":
                if metadata is not None:
                    raise ValueError("duplicate original metadata")
                with archive.extractfile(member) as stream:
                    metadata = list(
                        csv.reader(
                            io.StringIO(stream.read().decode("utf-8"), newline=""),
                            delimiter="|",
                            quoting=csv.QUOTE_NONE,
                        )
                    )
            elif name.suffix == ".wav":
                if (
                    len(name.parts) != 3
                    or name.parts[1] != "wavs"
                    or not re.fullmatch(r"LJ[0-9]{3}-[0-9]{4}", name.stem)
                    or name.stem in audio
                    or member.size <= 0
                ):
                    raise ValueError("original audio identity/path mismatch")
                audio[name.stem] = str(name)
    if not metadata:
        raise ValueError("original metadata required")
    identities = set()
    for row in metadata:
        if len(row) != 3 or not all(row) or row[0] not in audio or row[0] in identities:
            raise ValueError("original text/audio pair mismatch")
        identities.add(row[0])
    if identities != audio.keys():
        raise ValueError("unmatched original audio")
    # Preserve publisher CSV order, not an invented arrival timeline.
    for offset, (identity, text, normalized_text) in enumerate(metadata):
        yield {
            "id": identity,
            "text": text,
            "normalized_text": normalized_text,
            "audio": {
                "archive_file": path.name,
                "archive_member": audio[identity],
                "format": "wav",
            },
            "metadata_member": "LJSpeech-1.1/metadata.csv",
            "metadata_row_index": offset,
        }
