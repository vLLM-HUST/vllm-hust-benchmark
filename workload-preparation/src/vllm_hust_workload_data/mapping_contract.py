"""Portable source-mapping/v1 contract: explicit paths, no executable expressions."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

from .input_views import VIEW_TYPES
from .identity import canonical, key
from .records import WorkloadRecord
from .registry import read_catalog

VERSION = "source-mapping/v1"
PROTECTED_SOURCES = {"AGENTBENCH", "GAIA", "TAUBENCH"}
ORACLE_FIELDS = {
    "answer",
    "answers",
    "answerkey",
    "answer_index",
    "has_answer",
    "private_answer",
    "gold",
    "gold_answer",
    "gold_program_name",
    "reference_code",
    "canonical_solution",
    "cot_content",
    "reward",
    "score",
    "scorer",
    "solution",
}
TYPES = {"string", "number", "boolean", "object", "array", "schema"}
MISSING = object()


def _keys(obj, required, optional=()):
    if (
        not isinstance(obj, dict)
        or set(obj) - set(required) - set(optional)
        or set(required) - set(obj)
    ):
        raise ValueError("mapping has missing or unknown fields")


def _path(path):
    if not isinstance(path, list) or any(
        not (isinstance(p, str) and p or type(p) is int and p >= 0) for p in path
    ):
        raise ValueError("path must be an explicit array of keys/nonnegative indexes")


def _rule(rule, prefix=(), input_side=False):
    _keys(rule, ("path", "type", "required"), ("fields", "items"))
    _path(rule["path"])
    if (
        not isinstance(rule["type"], str)
        or rule["type"] not in TYPES
        or type(rule["required"]) is not bool
    ):
        raise ValueError("invalid mapping type or required flag")
    location = (*prefix, *rule["path"])
    if input_side and any(
        isinstance(p, str) and p.casefold() in ORACLE_FIELDS for p in location
    ):
        raise ValueError("outcome/answer path cannot enter workload input")
    if rule["type"] == "object":
        if (
            not isinstance(rule.get("fields"), dict)
            or not rule["fields"]
            or "items" in rule
        ):
            raise ValueError("objects require nonempty explicit field projections")
        leaves = []
        for target, child in rule["fields"].items():
            if input_side and target.casefold() in ORACLE_FIELDS:
                raise ValueError("outcome field cannot enter workload input")
            leaves.extend(_rule(child, location, input_side))
        return leaves
    if rule["type"] == "array":
        if "items" not in rule or "fields" in rule:
            raise ValueError("arrays require an explicit item projection")
        return _rule(rule["items"], (*location, "*"), input_side)
    if "fields" in rule or "items" in rule:
        raise ValueError("scalar/schema rules cannot have field/item projections")
    return [location]


def _overlap(a, b):
    return all(x == y or x == "*" or y == "*" for x, y in zip(a, b))


def validate_mapping(value):
    _keys(
        value,
        (
            "mapping_version",
            "source",
            "families",
            "records_path",
            "identity",
            "input",
            "features",
            "outcome",
            "runtime",
            "replay",
            "evidence",
        ),
    )
    if value["mapping_version"] != VERSION:
        raise ValueError("unsupported mapping version")
    source = value["source"]
    _keys(
        source,
        (
            "source_id",
            "revision",
            "unit",
            "publisher",
            "locator",
            "license",
            "provenance_kind",
        ),
    )
    if any(not isinstance(v, str) or not v.strip() for v in source.values()):
        raise ValueError(
            "source metadata must be explicit strings; use 'unknown' where unknown"
        )
    sid = source["source_id"]
    if not re.fullmatch(r"[A-Z][A-Z0-9_-]*", sid) or sid in PROTECTED_SOURCES:
        raise ValueError("invalid source ID or protected QBI source override")
    if source["revision"].lower() in {"unknown", "main", "master", "latest"}:
        raise ValueError(
            "bind a concrete revision or named local snapshot, not a floating branch"
        )
    families = value["families"]
    if (
        not isinstance(families, list)
        or not families
        or any(
            not isinstance(f, str)
            or f
            not in {
                row["family_id"]
                for row in read_catalog("workload_taxonomy_v1.json")[
                    "canonical_families"
                ]
            }
            for f in families
        )
        or len(set(families)) != len(families)
    ):
        raise ValueError("families require distinct canonical family IDs")
    _path(value["records_path"])
    _keys(value["identity"], ("task_id",), ("parent_task_id", "session_id"))
    identity_paths = []
    for rule in value["identity"].values():
        if rule is not None:
            identity_paths.extend(_rule(rule, input_side=True))
            if rule["type"] not in {"string", "number"}:
                raise ValueError("task identifiers must be scalar")
    inputs, outputs = [], []
    for section in ("input", "features", "outcome"):
        fields = value[section]
        if not isinstance(fields, dict) or (section == "input" and not fields):
            raise ValueError("mapping sections must be objects; input cannot be empty")
        for target, rule in fields.items():
            if not isinstance(target, str) or not target or target == "feature_inputs":
                raise ValueError("invalid/reserved target field")
            if section != "outcome" and target.casefold() in ORACLE_FIELDS:
                raise ValueError("outcome target cannot enter input")
            paths = _rule(rule, input_side=section != "outcome")
            if section == "features" and (
                target not in VIEW_TYPES
                or rule.get("type")
                != ("schema" if target == "output_schema" else VIEW_TYPES[target])
            ):
                raise ValueError("unknown feature view or wrong view type")
            (outputs if section == "outcome" else inputs).extend(paths)
    if any(_overlap(a, b) for a in inputs + identity_paths for b in outputs):
        raise ValueError("input/feature paths overlap outcome paths")
    runtime = value["runtime"]
    _keys(
        runtime,
        ("kind", "requirements", "environment_ref", "termination_ref", "scorer_ref"),
    )
    if runtime["kind"] not in {"task_dataset", "request_replay", "trace_only"}:
        raise ValueError(
            "interactive environments must reuse a reviewed external adapter"
        )
    if not isinstance(runtime["requirements"], list) or any(
        not isinstance(s, str) for s in runtime["requirements"]
    ):
        raise ValueError("requirements must be strings")
    if any(
        v is not None and not isinstance(v, str)
        for k, v in runtime.items()
        if k.endswith("_ref")
    ):
        raise ValueError("runtime references must be strings or null")
    if runtime["kind"] == "request_replay" and "messages" not in value["input"]:
        raise ValueError("request replay needs explicit original messages")
    _keys(value["replay"], ("order", "artificial_load"))
    if value["replay"] != {
        "order": "source_file_order_not_assumed_arrival_order",
        "artificial_load": False,
    }:
        raise ValueError("v1 does not synthesize load or infer arrival order")
    _keys(value["evidence"], ("field_reference", "review_status"))
    if (
        not isinstance(value["evidence"]["field_reference"], str)
        or not value["evidence"]["field_reference"].strip()
        or value["evidence"]["review_status"]
        not in {"needs_review", "field_contract_reviewed"}
    ):
        raise ValueError("field evidence and review status are required")
    canonical(value)
    return deepcopy(value)


def load_mapping(path):
    return validate_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def mapping_spec(mapping):
    mapping = validate_mapping(mapping)
    return {
        "kind": mapping["runtime"]["kind"],
        "mapping": VERSION,
        "required_fields": [],
        "requirements": mapping["runtime"]["requirements"],
        "records_path": mapping["records_path"],
        "status": "DECLARATIVE_MAPPING_LOCAL_VALIDATION_REQUIRED",
        "contract_sha256": key(mapping),
        "contract": mapping,
    }


def _get(row, path):
    value = row
    for part in path:
        if isinstance(value, dict) and isinstance(part, str):
            value = value.get(part, MISSING)
        elif isinstance(value, list) and type(part) is int and part < len(value):
            value = value[part]
        else:
            return MISSING
    return value


def _no_oracle(value):
    if isinstance(value, dict):
        if any(k.casefold() in ORACLE_FIELDS for k in value):
            raise ValueError("unprojected oracle field found in input schema")
        for v in value.values():
            _no_oracle(v)
    elif isinstance(value, list):
        for v in value:
            _no_oracle(v)


def project(row, rule, *, input_side=True):
    value = _get(row, rule["path"])
    if value is MISSING or value is None:
        if rule["required"]:
            raise ValueError(
                "required mapped value unavailable; source payload suppressed"
            )
        return MISSING
    expected = {
        "string": str,
        "boolean": bool,
        "object": dict,
        "array": list,
        "schema": (dict, bool),
        "number": (int, float),
    }[rule["type"]]
    if (
        not isinstance(value, expected)
        or rule["type"] == "number"
        and isinstance(value, bool)
    ):
        raise ValueError("mapped value has wrong type; source payload suppressed")
    if rule["type"] == "object":
        return _fields(value, rule["fields"], input_side=input_side)
    if rule["type"] == "array":
        result = [project(v, rule["items"], input_side=input_side) for v in value]
        if any(v is MISSING for v in result):
            raise ValueError("array mapping cannot silently drop unavailable elements")
        return result
    if input_side and rule["type"] == "schema":
        _no_oracle(value)
    canonical(value)
    return deepcopy(value)


def _fields(row, rules, *, input_side=True):
    result = {}
    for name, rule in rules.items():
        value = project(row, rule, input_side=input_side)
        if value is not MISSING:
            result[name] = value
    return result


def adapt_mapping(row, *, source, spec, index):
    mapping = spec["contract"]
    if (
        spec["contract_sha256"] != key(mapping)
        or source["source_id"] != mapping["source"]["source_id"]
        or source["revision"] != mapping["source"]["revision"]
    ):
        raise ValueError("mapping identity mismatch")
    data = _fields(row, mapping["input"])
    if not data:
        raise ValueError("record has no mapped input")
    if mapping["runtime"]["kind"] == "request_replay":
        if (
            not isinstance(data.get("messages"), list)
            or not data["messages"]
            or any(
                not isinstance(m, dict) or not isinstance(m.get("role"), str)
                for m in data["messages"]
            )
        ):
            raise ValueError("invalid mapped request messages")
        data = {"request_body": data}
    data["feature_inputs"] = _fields(row, mapping["features"])
    task = {"input": data}
    for name, rule in mapping["identity"].items():
        if rule is not None:
            value = project(row, rule)
            if value is not MISSING:
                task[name] = str(value)
    if not task.get("task_id"):
        task["task_id"] = f"{source['file_sha256']}:{index}"
        task["identity_kind"] = "file_sha256_row_index"
    else:
        task["identity_kind"] = "source_id"
    return WorkloadRecord(
        source=deepcopy(source),
        task=task,
        runtime={**mapping["runtime"], "execution": "not_executed"},
        replay={
            "adapter_version": VERSION,
            "mapping": VERSION,
            "mapping_sha256": spec["contract_sha256"],
            "source_row_index": index,
            "order": mapping["replay"]["order"],
            "annotations": {},
            "synthetic_fallback": False,
        },
        outcome=_fields(row, mapping["outcome"], input_side=False) or None,
    )


def example_mapping():
    def text(path, required=True):
        return {"path": path, "type": "string", "required": required}

    return {
        "mapping_version": VERSION,
        "source": {
            "source_id": "NEW_SOURCE",
            "revision": "REPLACE_WITH_PINNED_REVISION",
            "unit": "source_rows",
            "publisher": "unknown",
            "locator": "unknown",
            "license": "unknown",
            "provenance_kind": "unknown",
        },
        "families": ["standalone-reasoning"],
        "records_path": [],
        "identity": {"task_id": text(["id"], False)},
        "input": {"prompt": text(["question"])},
        "features": {"prompt": text(["question"])},
        "outcome": {"reference": text(["answer"], False)},
        "runtime": {
            "kind": "task_dataset",
            "requirements": [],
            "environment_ref": None,
            "termination_ref": None,
            "scorer_ref": None,
        },
        "replay": {
            "order": "source_file_order_not_assumed_arrival_order",
            "artificial_load": False,
        },
        "evidence": {
            "field_reference": "REPLACE_WITH_PUBLISHER_FIELD_DOCUMENTATION",
            "review_status": "needs_review",
        },
    }
