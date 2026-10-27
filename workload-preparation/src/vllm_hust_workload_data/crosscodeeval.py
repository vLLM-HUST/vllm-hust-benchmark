"""Read original CrossCodeEval archive views; never build prompts or run retrieval."""

from copy import deepcopy
import json
from pathlib import PurePosixPath
import tarfile

LANGUAGES = {"python", "java", "csharp", "typescript"}
PROFILES = {"line_completion"} | {
    f"line_completion_{setting}_{retriever}"
    for setting in ("rg1", "oracle")
    for retriever in ("bm25", "unixcoder_cosine_sim", "openai_cosine_sim")
}
PROVENANCE_FIELDS = {
    "source_archive_member",
    "source_member_row_index",
    "source_task_id",
    "source_language",
    "source_profile",
}


def original_rows(path):
    """Archive/member order stays original; repeated profile views are not new tasks."""
    with tarfile.open(path, "r:xz") as bundle:
        for member in bundle:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("uncontained/nonregular original archive member")
            if member.isdir() or not member.name.endswith(".jsonl"):
                continue
            if (
                len(name.parts) != 2
                or name.parts[0] not in LANGUAGES
                or name.stem not in PROFILES
            ):
                raise ValueError("unknown original language/profile, no coercion")
            if member.size > 2 * 1024**3:
                raise ValueError("original task member exceeds declared size budget")
            with bundle.extractfile(member) as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    if not isinstance(row, dict) or PROVENANCE_FIELDS & set(row):
                        raise ValueError("original task object or provenance collision")
                    if not all(
                        isinstance(row.get(key), str)
                        for key in ("prompt", "groundtruth", "right_context")
                    ):
                        raise ValueError(
                            "original code/reference fields must remain strings"
                        )
                    metadata = row.get("metadata")
                    if (
                        not isinstance(metadata, dict)
                        or not isinstance(metadata.get("task_id"), str)
                        or not metadata["task_id"]
                    ):
                        raise ValueError("original metadata task ID required")
                    if name.stem != "line_completion" and (
                        not isinstance(row.get("crossfile_context"), dict)
                        or not isinstance(row["crossfile_context"].get("text"), str)
                        or not isinstance(row["crossfile_context"].get("list"), list)
                    ):
                        raise ValueError(
                            "original retrieval context text/list required"
                        )
                    yield {
                        **deepcopy(row),
                        "source_archive_member": member.name,
                        "source_member_row_index": index,
                        "source_task_id": metadata["task_id"],
                        "source_language": name.parts[0],
                        "source_profile": name.stem,
                    }
