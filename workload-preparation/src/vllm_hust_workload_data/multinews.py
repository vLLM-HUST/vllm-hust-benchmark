"""Publisher Multi-News parallel files and original HF loader field semantics."""

from itertools import zip_longest
from pathlib import Path

from .assets import asset_path, verify_file
from .registry import source_spec


def original_rows(path):
    """Keep source/target line relationships; no prompt or fabricated arrival order.

    The publisher's multi_news.py strips line-edge whitespace and restores
    NEWLINE_CHAR in documents only. Original files remain unchanged and pinned.
    Bundle separators and reference text are not rewritten.
    """
    path = Path(path).resolve()
    manifest = source_spec("MULTINEWS")
    sources = {
        file["path"]: file for file in manifest["files"] if file["role"] == "data"
    }
    relative = "data/" + path.name
    if path.parent.name != "data" or relative not in sources:
        raise ValueError("explicit original Multi-News source filename required")
    root = path.parent.parent
    source = sources[relative]
    files = {file["path"]: file for file in manifest["files"]}
    target = files[source["paired_target"]]
    if target["role"] != "reference":
        raise ValueError("original paired target must be a reference, not an input")
    verify_file(root, source)
    verify_file(root, target)
    if path != asset_path(root, source["path"]):
        raise ValueError("original source path differs from registered binding")
    sentinel = object()
    count = 0
    with (
        path.open(encoding="utf-8") as src,
        asset_path(root, target["path"]).open(encoding="utf-8") as tgt,
    ):
        for index, (document, summary) in enumerate(
            zip_longest(src, tgt, fillvalue=sentinel)
        ):
            if document is sentinel or summary is sentinel:
                raise ValueError("original source/target line counts differ")
            count += 1
            yield {
                "document": document.strip().replace("NEWLINE_CHAR", "\n"),
                "summary": summary.strip(),
                "source_line_index": index,
                "target_file": target["path"],
                "target_sha256": target["sha256"],
            }
    if count != source["rows"] or count != target["rows"]:
        raise ValueError("original paired row count differs from publisher")
