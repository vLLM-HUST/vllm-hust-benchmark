"""Resolve prepared row references through their pinned source adapter."""

from itertools import islice
import json
from pathlib import Path

from .adapters import adapt_row, iter_source_rows
from .assets import asset_path, verify_file
from .pipeline import connect_readonly


def resolve_input(database: Path, record_id: str, *, data_root: Path):
    """Return an original WorkloadRecord; outcomes remain separate. Never log it."""
    connection = connect_readonly(database)
    try:
        row = connection.execute(
            "SELECT r.source_json,r.task_json,s.adapter_json FROM records r JOIN sources s USING(source_id) WHERE record_id=?",
            (record_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise KeyError("unknown prepared record")
    source, task, spec = map(json.loads, row)
    reference = task["input_ref"]
    if source["source_id"] == "FLORES200" and spec["mapping"] == "source-mapping/v1":
        if reference.get("mapping_sha256") != spec["contract_sha256"]:
            raise ValueError("original translation direction/profile binding mismatch")
    verify_file(data_root, {"path": reference["file"], "sha256": reference["sha256"]})
    rows = iter_source_rows(asset_path(data_root, reference["file"]), spec)
    try:
        if spec.get("history_targets") is not None:
            from .conversations import history_record

            if (
                source["source_id"] != "OASST1"
                or reference.get("view") != "original-parent-history/v1"
                or task["task_id"] not in spec["history_targets"]
            ):
                raise ValueError("original history view identity mismatch")
            originals = list(rows)
            index = reference["row_index"]
            if (
                index >= len(originals)
                or originals[index].get("message_id") != task["task_id"]
            ):
                raise ValueError("original history target identity changed")
            result = history_record(
                originals, task["task_id"], source=source, index=index
            )
            result.replay["selection"] = task.get("selection")
            verify_file(
                data_root, {"path": reference["file"], "sha256": reference["sha256"]}
            )
            return result
        original = next(
            islice(rows, reference["row_index"], reference["row_index"] + 1), None
        )
        if original is None:
            raise ValueError("source row reference no longer exists")
        result = adapt_row(
            source["source_id"],
            original,
            source=source,
            spec=spec,
            index=reference["row_index"],
        )
        if task.get("selection") is not None:
            result.replay["selection"] = task["selection"]
        verify_file(
            data_root, {"path": reference["file"], "sha256": reference["sha256"]}
        )
        return result
    finally:
        rows.close()
