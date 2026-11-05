"""Original procedural browser task definitions, never fabricated prompt rows."""

from copy import deepcopy
import hashlib
from pathlib import Path, PurePosixPath
import zipfile

from .assets import asset_path, verify_file
from .records import WorkloadRecord

PREFIX = (
    "MiniWoB-plusplus-33c3b4ddef8c6eb67c57a29663d844b1eda7e614/miniwob/html/miniwob/"
)


def original_rows(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.orig_filename for entry in entries]
        if len(names) != len(set(names)) or len(entries) != len(
            set(archive.namelist())
        ):
            raise ValueError("duplicate original archive members")
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in name:
                raise ValueError("unsafe original archive member reference")
        for member in entries:
            name = member.filename
            if not name.startswith(PREFIX) or not name.endswith(".html"):
                continue
            suffix = name[len(PREFIX) :]
            if "/" in suffix:
                raise ValueError("nested task definition is not a native task name")
            raw = archive.read(member)
            if not raw.strip():
                raise ValueError("empty original task definition")
            yield {
                "task_name": "miniwob." + Path(suffix).stem,
                "definition_member": name,
                "definition_sha256": hashlib.sha256(raw).hexdigest(),
            }


def adapt_case(row, *, source, index):
    if source["source_id"] != "MINIWOB_PLUSPLUS":
        raise ValueError("original MiniWoB++ source required")
    return WorkloadRecord(
        source=deepcopy(source),
        task={"task_id": row["task_name"], "input": {"task_name": row["task_name"]}},
        runtime={
            "kind": "native_environment",
            "execution": "not_initialized",
            "definition_member": row["definition_member"],
            "definition_sha256": row["definition_sha256"],
            "requirements": [
                "Original full archive contains shared page assets, task generator and reward logic",
                "Initialize the original environment with an explicit experiment seed to obtain an actual instruction and observation",
                "Task definition is not a request, gold action or model prompt; no source code is injected into model input",
                "Check official BrowserGym registration before execution; 130 source definitions are not 130 supported runtime tasks",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "miniwob_original_html_definitions",
            "source_row_index": index,
            "order": "archive_definition_order_not_request_arrival",
            "synthetic_fallback": False,
        },
    )


def read_native_definition(record, *, data_root):
    if record.source["source_id"] != "MINIWOB_PLUSPLUS":
        raise ValueError("original MiniWoB++ record required")
    relative = record.source["file"]
    verify_file(data_root, {"path": relative, "sha256": record.source["file_sha256"]})
    with zipfile.ZipFile(asset_path(data_root, relative)) as archive:
        raw = archive.read(record.runtime["definition_member"])
    if hashlib.sha256(raw).hexdigest() != record.runtime["definition_sha256"]:
        raise ValueError("original task definition identity changed")
    return raw
