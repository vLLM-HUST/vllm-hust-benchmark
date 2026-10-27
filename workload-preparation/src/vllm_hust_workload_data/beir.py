"""Original BEIR documents, queries and judgments; never fabricated chat requests."""

from copy import deepcopy
import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

from .assets import iter_rows, sha256
from .records import WorkloadRecord
from .registry import source_spec


SOURCE_IDS = {"BEIR_SCIFACT", "BEIR_NFCORPUS"}


def original_asset_rows(path):
    path = Path(path)
    if path.name in {"corpus.jsonl", "queries.jsonl"}:
        asset_type = (
            "corpus_document" if path.name == "corpus.jsonl" else "retrieval_query"
        )
        seen = set()
        for row in iter_rows(path):
            if (
                not isinstance(row.get("_id"), str)
                or not row["_id"]
                or not isinstance(row.get("text"), str)
            ):
                raise ValueError("original BEIR ID and text required")
            if asset_type == "corpus_document" and not isinstance(
                row.get("title"), str
            ):
                raise ValueError("original BEIR corpus title required")
            if row["_id"] in seen:
                raise ValueError("duplicate original BEIR ID within one asset")
            seen.add(row["_id"])
            yield {"native_asset_type": asset_type, "original_row": row}
    elif path.parent.name == "qrels" and path.suffix == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != ["query-id", "corpus-id", "score"]:
                raise ValueError("original BEIR judgment columns required")
            for row in reader:
                if None in row or any(
                    not isinstance(value, str) or not value for value in row.values()
                ):
                    raise ValueError("incomplete original relevance judgment")
                try:
                    int(row["score"])
                except ValueError:
                    raise ValueError(
                        "original integer relevance score required"
                    ) from None
                yield {"native_asset_type": "relevance_judgment", "original_row": row}
    else:
        raise ValueError("unknown original BEIR asset role; no prompt fallback")


def adapt_asset(row, *, source, index):
    asset_type, raw = row["native_asset_type"], row["original_row"]
    if asset_type == "corpus_document":
        fields = ["title", "text"]
    elif asset_type == "retrieval_query":
        fields = ["text"]
    elif asset_type == "relevance_judgment":
        fields = []
    else:
        raise ValueError("unreviewed original BEIR asset type")
    task_input = {key: deepcopy(raw[key]) for key in fields}
    outcome = {"original_judgment": deepcopy(raw)} if not fields else None
    annotations = (
        {key: deepcopy(value) for key, value in raw.items() if key not in fields}
        if fields
        else {}
    )
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": raw["_id"] if fields else f"{source['file_sha256']}:{index}",
            "identity_kind": "original_asset_id" if fields else "file_sha256_row_index",
            "native_unit": asset_type,
            "input": task_input,
        },
        runtime={
            "kind": "task_dataset",
            "execution": "not_executed",
            "asset_role": "reference_only"
            if not fields
            else "original_model_input_asset",
            "requirements": [
                "Original BEIR loader and retrieval/reranking; individual asset rows are not chat requests"
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "beir_original_assets",
            "source_row_index": index,
            "order": "original_asset_file_order_not_request_arrival",
            "annotations": annotations,
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome=outcome,
    )


def prepare_original_archive(*, source_id, data_archive, output):
    if source_id not in SOURCE_IDS:
        raise ValueError("unreviewed original BEIR source")
    source = source_spec(source_id)
    (archive_spec,) = [f for f in source["files"] if f["role"] == "source_archive"]
    data_archive, output = Path(data_archive), Path(output)
    if output.exists():
        raise ValueError("fresh BEIR output directory required")
    if (
        data_archive.stat().st_size != archive_spec["bytes"]
        or sha256(data_archive) != archive_spec["sha256"]
    ):
        raise ValueError("original BEIR publisher archive identity mismatch")
    verified = []
    with ZipFile(data_archive) as archive:
        member_names = [member.filename for member in archive.infolist()]
        if len(member_names) != len(set(member_names)):
            raise ValueError("duplicate original BEIR archive member")
        for spec in [f for f in source["files"] if f["role"] == "data"]:
            member = archive.getinfo(spec["archive_member"])
            if member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("regular original data member required")
            raw = archive.read(member)
            if (
                len(raw) != spec["bytes"]
                or hashlib.sha256(raw).hexdigest() != spec["sha256"]
            ):
                raise ValueError("original BEIR member identity mismatch")
            target = (output / spec["path"]).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError("BEIR member path escapes fresh output directory")
            verified.append((target, raw))
    output.mkdir(parents=True, mode=0o700)
    for target, raw in verified:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    report = validate_original_relations(source_id=source_id, data_root=output)
    return {
        "source_id": source_id,
        "output": str(output),
        "revision": source["revision"],
        **report,
        "native_execution_or_scoring_verified": False,
    }


def validate_original_relations(*, source_id, data_root):
    """Check every original judgment edge without filtering or manufacturing pairs."""
    if source_id not in SOURCE_IDS:
        raise ValueError("unreviewed original BEIR source")
    source = source_spec(source_id)
    root = Path(data_root)
    by_type = {}
    judgments = []
    for spec in [f for f in source["files"] if f["role"] == "data"]:
        from .assets import asset_path, verify_file

        verify_file(root, spec)
        rows = list(original_asset_rows(asset_path(root, spec["path"])))
        if len(rows) != spec["rows"]:
            raise ValueError("original BEIR asset row count mismatch")
        if rows and rows[0]["native_asset_type"] == "relevance_judgment":
            judgments.append((spec["path"], [r["original_row"] for r in rows]))
        else:
            by_type[Path(spec["path"]).name] = {
                r["original_row"]["_id"]: r["original_row"] for r in rows
            }
    corpus, queries = by_type["corpus.jsonl"], by_type["queries.jsonl"]
    splits = []
    for name, rows in judgments:
        if any(
            r["query-id"] not in queries or r["corpus-id"] not in corpus for r in rows
        ):
            raise ValueError(
                "original relevance edge points to missing query or document"
            )
        splits.append(
            {
                "file": name,
                "original_judgment_rows": len(rows),
                "original_judged_queries": len({r["query-id"] for r in rows}),
            }
        )
    return {
        "original_corpus_documents": len(corpus),
        "original_queries_all_splits": len(queries),
        "original_qrels_splits": splits,
        "original_metadata_preserved": True,
        "query_or_corpus_filtered_or_rewritten": False,
    }
