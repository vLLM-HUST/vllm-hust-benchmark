"""Original keyed scientific tasks, references and data-file relations; no prompts."""

from copy import deepcopy
from itertools import islice
import json
from pathlib import Path, PurePosixPath
import re

from .assets import asset_path, iter_rows, verify_file
from .records import WorkloadRecord
from .registry import read_catalog

KEY = "answer_key/answer_key_real.csv"
KEY_FIELDS = {"dataset", "metadataid", "query_id", "gold_hypo"}


def filemap(spec):
    files = spec.get("relation_files")
    if files is None:
        files = read_catalog("discoverybench_original_real_files_v1.json")["files"]
    mapped = {item["path"]: item for item in files}
    if len(mapped) != len(files) or KEY not in mapped:
        raise ValueError("unique original files and real answer key required")
    return mapped


def queries(value):
    if isinstance(value, list):
        for child in value:
            yield from queries(child)
    elif isinstance(value, dict) and "qid" in value and "question" in value:
        yield value
    else:
        raise ValueError("original nested query relation malformed")


def original_rows(path, spec):
    path = Path(path)
    if path.name != "answer_key_real.csv" or path.parent.name != "answer_key":
        raise ValueError("original real answer-key CSV required")
    root, files = path.parent.parent, filemap(spec)
    for entry in files.values():
        verify_file(root, entry)
    seen = set()
    try:
        for key in iter_rows(path):
            if (
                set(key) != KEY_FIELDS
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", key["dataset"])
                or key["dataset"] in {".", ".."}
                or not all(key[name].isdigit() for name in ("metadataid", "query_id"))
            ):
                raise ValueError("original four-field keyed task required")
            identity = (key["dataset"], int(key["metadataid"]), int(key["query_id"]))
            if identity in seen:
                raise ValueError("duplicate original scientific task identity")
            seen.add(identity)
            name = f"discoverybench/real/test/{identity[0]}/metadata_{identity[1]}.json"
            assets, blockers, metadata, question, material = (
                [deepcopy(files[KEY])],
                [],
                None,
                None,
                None,
            )
            if name not in files:
                blockers.append("ORIGINAL_METADATA_NOT_REGISTERED")
            else:
                assets.append(deepcopy(files[name]))
                try:
                    metadata = json.loads(asset_path(root, name).read_text("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    blockers.append("ORIGINAL_METADATA_DECODE_FAILED")
                if metadata is not None:
                    matches = [
                        q
                        for q in queries(metadata["queries"])
                        if int(q["qid"]) == identity[2]
                    ]
                    if len(matches) != 1 or not isinstance(matches[0]["question"], str):
                        raise ValueError("original question relation not unique")
                    question = matches[0]
                    material = {
                        "datasets": deepcopy(metadata["datasets"]),
                        "hypotheses": {
                            "intermediate": deepcopy(
                                metadata["hypotheses"]["intermediate"]
                            )
                        },
                    }
                    for optional in ("domain", "domain_knowledge", "workflow_tags"):
                        if optional in metadata:
                            material[optional] = deepcopy(metadata[optional])
                    for item in metadata["datasets"]:
                        relative = str(PurePosixPath(name).parent / item["name"])
                        if relative not in files:
                            blockers.append(
                                "ORIGINAL_DATA_REFERENCE_UNRESOLVED:" + relative
                            )
                        else:
                            assets.append(deepcopy(files[relative]))
            yield {
                "original_key": key,
                "original_query": question,
                "original_prompt_material": material,
                "metadata_path": name,
                "native_files": assets,
                "blockers": blockers,
            }
    finally:
        for entry in files.values():
            verify_file(root, entry)


def adapt_case(row, *, source, index):
    if source.get("source_id") != "DISCOVERYBENCH":
        raise ValueError("original DiscoveryBench source required")
    key = row["original_key"]
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": f"{key['dataset']}/{int(key['metadataid'])}/{int(key['query_id'])}",
            "identity_kind": "original_dataset_metadata_query_tuple",
            "native_unit": "scientific_discovery_with_original_data",
            "input": {
                "original_task_key": {
                    name: key[name] for name in ("dataset", "metadataid", "query_id")
                },
                "original_query": deepcopy(row["original_query"]),
                "original_prompt_material": deepcopy(row["original_prompt_material"]),
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "native_input_status": "BLOCKED_ORIGINAL_ASSETS"
            if row["blockers"]
            else "READY_ORIGINAL_INPUT_NOT_EXECUTED",
            "original_metadata_path": row["metadata_path"],
            "native_files": deepcopy(row["native_files"]),
            "blockers": deepcopy(row["blockers"]),
            "requirements": [
                "Use original Agent and scoring with explicitly configured models; no rendered/synthetic prompt",
                "Original metadata main hypotheses and answer-key gold are not model input",
                "Blocked publisher assets stay registered; no decoder guess, alias or task filtering",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "discoverybench_original_real_tasks",
            "source_row_index": index,
            "order": "original_real_answer_key_order",
            "synthetic_fallback": False,
        },
        outcome={"references": {"gold_hypo": key["gold_hypo"]}},
    )


def restore_original_case(record, *, data_root, spec=None):
    if record.source.get("source_id") != "DISCOVERYBENCH":
        raise ValueError("original DiscoveryBench record required")
    if record.runtime["blockers"]:
        raise ValueError(
            "original publisher task blocked; inspect original asset references"
        )
    rows = original_rows(Path(data_root) / KEY, spec or {})
    try:
        row = next(islice(rows, record.replay["source_row_index"], None))
    finally:
        rows.close()
    expected = adapt_case(
        row, source=record.source, index=record.replay["source_row_index"]
    )
    if (
        expected.task != record.task
        or expected.outcome != record.outcome
        or expected.runtime != record.runtime
    ):
        raise ValueError("original task/reference/file relation changed")
    metadata = json.loads(
        asset_path(data_root, row["metadata_path"]).read_text("utf-8")
    )
    return {
        "answer_key_row": deepcopy(row["original_key"]),
        "metadata": metadata,
        "question": deepcopy(row["original_query"]),
    }
