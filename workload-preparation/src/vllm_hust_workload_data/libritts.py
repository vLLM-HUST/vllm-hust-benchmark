"""Original LibriTTS text/reference-waveform pairs; never synthesize speech."""

import io
from pathlib import Path, PurePosixPath
import re
import tarfile
import wave


def original_rows(path):
    path = Path(path)
    if path.name != "test-other.tar.gz":
        raise ValueError("pinned original LibriTTS test-other archive required")
    seen, pairs, context = set(), {}, []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "LibriTTS"
                or member.name in seen
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError("unsafe or duplicate original archive member")
            seen.add(member.name)
            if member.isdir():
                continue
            suffix = next(
                (
                    s
                    for s in (".original.txt", ".normalized.txt", ".wav")
                    if member.name.endswith(s)
                ),
                None,
            )
            if suffix is None:
                context.append(str(name))
                continue
            if len(name.parts) != 5 or name.parts[1] != "test-other":
                raise ValueError("original utterance path required")
            identity = name.name[: -len(suffix)]
            if not re.fullmatch(r"[0-9]+_[0-9]+_[0-9]+_[0-9]+", identity):
                raise ValueError("original utterance identity required")
            speaker, chapter, segment, utterance = identity.split("_")
            if (speaker, chapter) != name.parts[2:4]:
                raise ValueError("speaker/chapter path mismatch")
            item = pairs.setdefault(identity, {"members": {}, "texts": {}})
            if suffix in item["members"] or member.size <= 0:
                raise ValueError("duplicate or empty original utterance asset")
            item["members"][suffix] = str(name)
            with archive.extractfile(member) as stream:
                if suffix == ".wav":
                    # wave seeks within headers; tar's streaming member cannot
                    # supply random access. Buffer only the bounded header,
                    # never the reference PCM or the whole archive.
                    with wave.open(io.BytesIO(stream.read(65536)), "rb") as audio:
                        if (
                            audio.getnchannels(),
                            audio.getsampwidth(),
                            audio.getframerate(),
                        ) != (1, 2, 24000) or audio.getnframes() <= 0:
                            raise ValueError(
                                "original mono16-bit24kHz reference required"
                            )
                else:
                    if member.size > 1024**2:
                        raise ValueError("original utterance text exceeds read budget")
                    item["texts"][suffix] = stream.read().decode("utf-8")
    if not pairs:
        raise ValueError("original utterance pairs required")
    required = {".original.txt", ".normalized.txt", ".wav"}
    if any(set(item["members"]) != required for item in pairs.values()):
        raise ValueError("unmatched original text/reference-audio pair")
    # Same lexical utterance-ID ordering as the official TorchAudio reader;
    # not a fabricated request timeline or a reconstructed conversation.
    for identity, item in sorted(pairs.items()):
        speaker, chapter, segment, utterance = identity.split("_")
        parent = PurePosixPath(item["members"][".wav"]).parent
        yield {
            "id": identity,
            "text": item["texts"][".original.txt"],
            "normalized_text": item["texts"][".normalized.txt"],
            "audio": {
                "archive_file": path.name,
                "archive_member": item["members"][".wav"],
                "format": "wav",
            },
            "speaker_id": speaker,
            "chapter_id": chapter,
            "segment_id": segment,
            "utterance_id": utterance,
            "original_text_member": item["members"][".original.txt"],
            "normalized_text_member": item["members"][".normalized.txt"],
            "context_metadata_members": [
                n
                for n in context
                if PurePosixPath(n).parent in (parent, PurePosixPath("LibriTTS"))
            ],
        }
