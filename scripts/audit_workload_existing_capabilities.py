"""Compare the frozen 45-definition baseline with data-only provider fixtures.

No raw data, network, model, native environment, feature extraction or analysis.
The caller explicitly selects trusted research source code for this audit.
"""

import importlib
import json
import sys
import types
import copy
import ast
import hashlib
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(
    description="Offline source-to-provider migration audit; invented fixtures, no data acquisition or models."
)
parser.add_argument("--research-component", type=Path, required=True)
parser.add_argument(
    "--target-component",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "workload-preparation",
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error("choose a fresh receipt; existing evidence is immutable")
base = args.research_component.resolve()
destination = args.target_component.resolve()
sys.path.insert(0, str(destination / "src"))
pkg = types.ModuleType("migration_origin")
pkg.__path__ = [str(base / "src/llm_serving_workloads")]
sys.modules["migration_origin"] = pkg
origin = importlib.import_module("migration_origin.adapters")
target = importlib.import_module("vllm_hust_workload_data.adapters")
registry = importlib.import_module("vllm_hust_workload_data.registry")
adapter_spec = registry.adapter_spec

specs = {}
for f in ("adapter_extensions_v1.json", "adapter_registry_v1.json"):
    specs.update(
        json.loads((base / "data/catalogs" / f).read_text(encoding="utf-8"))["adapters"]
    )
rows = []
for sid, spec in specs.items():
    if sid in {"AGENTBENCH", "GAIA", "TAUBENCH"}:
        continue
    row = {key: "fixture" for key in spec.get("required_fields", [])}
    for k in spec.get("input_fields", []):
        row.setdefault(k, "fixture")
    for k in spec.get("outcome_fields", []):
        row[k] = "REFERENCE_SENTINEL"
    for k in ("messages", "conversation", "conversations"):
        if k in row:
            row[k] = [
                {
                    "role": "user",
                    "content": "fixture",
                    "from": "human",
                    "value": "fixture",
                },
                {
                    "role": "assistant",
                    "content": "reference",
                    "from": "gpt",
                    "value": "reference",
                },
            ]
    for k in (
        "images",
        "hash_ids",
        "haystack_dates",
        "haystack_session_ids",
        "haystack_sessions",
        "Context",
    ):
        if k in row:
            row[k] = []
    if "options" in row:
        row["options"] = ["a", "b"]
    if "choices" in row:
        row["choices"] = {"text": ["a", "b"], "label": ["A", "B"]}
    for k in ("image", "audio"):
        if k in row:
            row[k] = {"bytes": b"fixture", "path": None}
    if "request_body" in row:
        row["request_body"] = json.dumps(
            {"messages": [{"role": "user", "content": "fixture"}]}
        )
    if sid == "OASST1":
        row.update(parent_id=None, role="user")
    if sid == "PAIO-LONGTEXT-4000":
        row["query_text"] = "fixture query"
    for k in (
        "Timestamp",
        "Request tokens",
        "Response tokens",
        "round_index",
        "timestamp",
        "input_length",
        "output_length",
        "ContextTokens",
        "GeneratedTokens",
    ):
        if k in row:
            row[k] = 1
    if sid == "STRUCTEVAL":
        row["feature_requirements"] = []
    source = {
        "source_id": sid,
        "revision": "software-fixture",
        "file_sha256": "f" * 64,
        "file": "fixture.jsonl",
    }
    before = copy.deepcopy(row)
    a = origin.adapt_row(sid, row, source=source, spec=spec, index=0)
    b = target.adapt_row(sid, row, source=source, spec=adapter_spec(sid), index=0)
    ok = (
        a.task["input"] == b.task["input"]
        and a.outcome == b.outcome
        and a.runtime == b.runtime
        and row == before
    )
    rows.append({"source_id": sid, "input_outcome_runtime_equal_and_no_mutation": ok})
assert len(rows) == 42 and all(
    x["input_outcome_runtime_equal_and_no_mutation"] for x in rows
)
checks = {}


def catalog(root, name):
    return json.loads((root / "data/catalogs" / name).read_text(encoding="utf-8"))


all_adapters = registry.all_adapters

assert set(specs) <= set(all_adapters())
checks["baseline_adapter_ids_missing"] = sorted(set(specs) - set(all_adapters()))
source_ids = {
    s["source_id"] for s in catalog(base, "authoritative_datasets_v1.json")["sources"]
}
target_ids = {
    s["source_id"]
    for s in catalog(destination, "authoritative_datasets_v1.json")["sources"]
}
assert source_ids <= target_ids
checks["baseline_public_sources"] = len(source_ids)
checks["public_source_ids_missing"] = sorted(source_ids - target_ids)
origin_sources = catalog(base, "source_manifests_v1.json")["sources"]
target_sources = catalog(destination, "source_manifests_v1.json")["sources"]
fixed = 0
for sid, spec in origin_sources.items():
    for asset in spec["files"]:
        if asset["role"] != "data":
            continue
        assert any(
            f["path"] == asset["path"] and f["sha256"] == asset["sha256"]
            for f in target_sources[sid]["files"]
        )
        fixed += 1
checks["baseline_fixed_data_file_identities_covered"] = fixed
checks["taxonomy_v1_json_equal"] = catalog(
    base, "workload_taxonomy_v1.json"
) == catalog(destination, "workload_taxonomy_v1.json")
assert checks["taxonomy_v1_json_equal"]
a = catalog(base, "workload_taxonomy_v2.json")
b = catalog(destination, "workload_taxonomy_v2.json")
for entry in b["replacements"]:
    if entry["family_id"] == "text-summarization":
        entry["candidate_source_ids"] = [
            sid
            for sid in entry["candidate_source_ids"]
            if sid != "NATURAL_INSTRUCTIONS"
        ]
assert a == b
checks["taxonomy_v2_delta"] = "ONLY_ADDED_NATURAL_INSTRUCTIONS_TO_SUMMARIZATION"
checks["original_source_tools_ast_equal"] = {}
for name in (
    "materialize_public_datasets.py",
    "verify_public_sources.py",
    "validate_paio_assets.py",
):
    same = ast.dump(
        ast.parse((base / "analysis" / name).read_text(encoding="utf-8")),
        include_attributes=False,
    ) == ast.dump(
        ast.parse((destination / "source_tools" / name).read_text(encoding="utf-8")),
        include_attributes=False,
    )
    assert same
    checks["original_source_tools_ast_equal"][name] = same
report = {
    "schema_version": "workload-migration-capability-audit/v1",
    "status": "PASS_SOFTWARE_FIXTURES_NOT_REAL_TASKS",
    "baseline_adapter_definitions": 45,
    "compared_data_adapters": 42,
    "intentionally_excluded_qbi_definitions": ["AGENTBENCH", "GAIA", "TAUBENCH"],
    "model_calls": 0,
    "native_tasks_run": 0,
    "new_datasets_acquired": 0,
    "checks": checks,
    "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "rows": rows,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))
