"""Every published MOD result is attached to a canonical catalog entry."""

import json
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website-publication" / "data"


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_result_bindings_are_complete_and_schema_valid():
    evidence = load("mod-test-evidence.json")
    bindings = load("mod-result-bindings.json")
    schema = load("schemas/mod_result_bindings_v1.schema.json")
    jsonschema.validate(bindings, schema)

    for evidence_key, binding_key in (
        ("integration_records", "integration_records"),
        ("combination_effects", "combination_effects"),
        ("negative_results", "negative_results"),
    ):
        evidence_ids = {row["id"] for row in evidence[evidence_key]}
        assert set(bindings[binding_key]) == evidence_ids


def test_bindings_only_reference_catalog_components():
    catalog_ids = {row["id"] for row in load("ecosystem.json")["components"]}
    bindings = load("mod-result-bindings.json")
    referenced = {
        component_id
        for group in ("integration_records", "combination_effects", "negative_results")
        for component_ids in bindings[group].values()
        for component_id in component_ids
    }
    assert referenced <= catalog_ids


def test_new_family_repositories_have_requested_ownership():
    components = {row["id"]: row for row in load("ecosystem.json")["components"]}
    for component_id in (
        "gdn-optimizations",
        "runtime-micro-optimizations",
        "ascend-attention-optimizations",
    ):
        component = components[component_id]
        assert component["maintainers"] == ["ShuhaoZhangTony"]
        assert component["advisors"] == []
        assert component["repository_visibility"] == "public"
        assert component["canonical_repository"].startswith(
            "https://github.com/vLLM-HUST/"
        )
