"""Original passage reranking query pools; never turn them into chat prompts."""

from copy import deepcopy
import sqlite3
import tarfile
from pathlib import Path

from .assets import verify_file
from .records import WorkloadRecord


def relation_spec(manifest, adapter):
    files = {item["path"]: item for item in manifest["files"]}
    if set(files) != {"qrels.dev.tsv", "top1000.dev.tar.gz"}:
        raise ValueError("original MS MARCO dev candidates and qrels required")
    if (
        files["top1000.dev.tar.gz"]["role"] != "data"
        or files["qrels.dev.tsv"]["role"] != "reference"
    ):
        raise ValueError("original candidates and reference roles must stay separate")
    return {
        **adapter,
        "original_qrels": files["qrels.dev.tsv"],
        "original_archive": files["top1000.dev.tar.gz"],
    }


def original_rows(path, spec):
    path = Path(path)
    if path.name != "top1000.dev.tar.gz":
        raise ValueError("original dev candidate archive required")
    reference = verify_file(path.parent, spec["original_qrels"])
    references = {}
    with (path.parent / "qrels.dev.tsv").open(encoding="utf-8", newline="") as stream:
        for line in stream:
            fields = line.removesuffix("\n").removesuffix("\r").split("\t")
            if (
                len(fields) != 4
                or not fields[0].isdigit()
                or not fields[2].isdigit()
                or fields[1] != "0"
                or fields[3] != "1"
            ):
                raise ValueError(
                    "original four-field positive dev qrel required; payload suppressed"
                )
            references.setdefault(fields[0], []).append(fields)
    query_limit = spec.get("diagnostic_query_limit")
    if query_limit is not None and (type(query_limit) is not int or query_limit <= 0):
        raise ValueError("diagnostic query count must be positive or omitted")
    # The publisher interleaves queries. Scan the complete archive before emitting
    # any pool. SQLite keeps aggregation overhead bounded without writing raw data
    # to an implicit disk cache. A limited run stores candidates for selected queries
    # only; a full run needs RAM for the complete original candidate text.
    queries = {}
    database = sqlite3.connect(":memory:")
    try:
        yield from _group_original_candidates(
            path, spec, references, reference, query_limit, queries, database
        )
    finally:
        database.close()


def _group_original_candidates(
    path, spec, references, reference, query_limit, queries, database
):
    database.execute(
        "CREATE TABLE candidates (query_index INTEGER, position INTEGER, passage_id TEXT, passage TEXT, PRIMARY KEY(query_index, position)) WITHOUT ROWID"
    )
    members = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name != "top1000.dev" or members:
                raise ValueError("one original regular top1000.dev member required")
            expected = spec["original_archive"].get("archive_member_bytes")
            if expected is not None and member.size != expected:
                raise ValueError("original candidate member size differs")
            members += 1
            with archive.extractfile(member) as stream:
                for raw in stream:
                    try:
                        fields = (
                            raw.decode("utf-8")
                            .removesuffix("\n")
                            .removesuffix("\r")
                            .split("\t", 3)
                        )
                    except UnicodeError:
                        raise ValueError(
                            "invalid original UTF-8 candidate; payload suppressed"
                        ) from None
                    if (
                        len(fields) != 4
                        or not fields[0].isdigit()
                        or not fields[1].isdigit()
                        or not fields[2]
                        or not fields[3]
                    ):
                        raise ValueError(
                            "original query/passage candidate row required; payload suppressed"
                        )
                    qid, pid, query, passage = fields
                    if qid not in queries:
                        queries[qid] = [query, len(queries), 0]
                    original_query, query_index, position = queries[qid]
                    if original_query != query:
                        raise ValueError(
                            "original query text differs within its candidate pool"
                        )
                    if query_limit is None or query_index < query_limit:
                        database.execute(
                            "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                            (query_index, position, pid, passage),
                        )
                    queries[qid][2] += 1
    if members != 1 or not queries:
        raise ValueError("original dev candidate archive is empty")
    for qid, (query, query_index, count) in queries.items():
        if query_limit is not None and query_index >= query_limit:
            break
        candidates = [
            {
                "passage_id": pid,
                "passage": passage,
                "original_position_within_query": position,
            }
            for position, pid, passage in database.execute(
                "SELECT position, passage_id, passage FROM candidates WHERE query_index=? ORDER BY position",
                (query_index,),
            )
        ]
        if len(candidates) != count:
            raise ValueError("original query pool lost candidates")
        yield {
            "query_id": qid,
            "query": query,
            "candidates": candidates,
            "original_qrel_rows": deepcopy(references.get(qid, [])),
            "qrels_sha256": reference["sha256"],
        }


def adapt_case(row, *, source, index):
    fields = {"query_id", "query", "candidates", "original_qrel_rows", "qrels_sha256"}
    if (
        source.get("source_id") != "MSMARCO_PASSAGE"
        or not isinstance(row, dict)
        or set(row) != fields
    ):
        raise ValueError("original MS MARCO query pool required")
    if (
        not isinstance(row["query_id"], str)
        or not row["query_id"].isdigit()
        or not isinstance(row["query"], str)
        or not row["query"]
        or not isinstance(row["candidates"], list)
        or not row["candidates"]
    ):
        raise ValueError("original query and nonempty complete candidate pool required")
    for position, candidate in enumerate(row["candidates"]):
        if (
            not isinstance(candidate, dict)
            or set(candidate)
            != {"passage_id", "passage", "original_position_within_query"}
            or candidate["original_position_within_query"] != position
            or not isinstance(candidate["passage_id"], str)
            or not candidate["passage_id"].isdigit()
            or not isinstance(candidate["passage"], str)
            or not candidate["passage"]
        ):
            raise ValueError("original candidate fields and ordering required")
    if not isinstance(row["original_qrel_rows"], list) or any(
        not isinstance(qrel, list)
        or len(qrel) != 4
        or qrel[0] != row["query_id"]
        or qrel[1] != "0"
        or not isinstance(qrel[2], str)
        or not qrel[2].isdigit()
        or qrel[3] != "1"
        for qrel in row["original_qrel_rows"]
    ):
        raise ValueError("original qrel rows must remain separate from model input")
    return WorkloadRecord(
        source={**deepcopy(source), "qrels_sha256": row["qrels_sha256"]},
        task={
            "task_id": row["query_id"],
            "identity_kind": "original_query_id",
            "native_unit": "passage_reranking_query_pool",
            "input": {key: deepcopy(row[key]) for key in ["query", "candidates"]},
        },
        runtime={
            "kind": "task_dataset",
            "execution": "not_executed",
            "requirements": [
                "Rank the complete original candidate pool with a suitable reranker; use original Microsoft MRR evaluator",
                "Candidate and query text are native inputs, not chat requests; no synthetic negatives or reference leakage",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "msmarco_original_dev_query_pools",
            "source_row_index": index,
            "order": "original_query_pool_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={
            "references": {"original_qrel_rows": deepcopy(row["original_qrel_rows"])},
            "reference_kind": "original_four_column_qrels_not_model_input",
        },
    )
