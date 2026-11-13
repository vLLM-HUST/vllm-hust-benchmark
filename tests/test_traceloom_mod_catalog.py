"""Keep independent MOD identities and explicitly declared public ownership."""

import importlib.util
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "website-publication/data"


def test_independent_public_cards_keep_bilingual_summaries_after_projection():
    root = DATA.parents[1]
    spec = importlib.util.spec_from_file_location(
        "public_projection", root / "scripts/build_public_site.py"
    )
    projection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projection)
    source = json.loads((DATA / "ecosystem.json").read_text(encoding="utf-8"))
    public = projection.project_json(source, set())
    components = {item["id"]: item for item in public["components"]}
    for key in ("graph-coverage", "prefill-admission", "sglang-prepared-dispatch"):
        item = components[key]
        assert item["summary_en"] and item["summary_zh"]
        assert item["maintainers"] == ["ShuhaoZhangTony"]
        assert item["advisors"] == []


def test_three_independent_traceloom_optimizations_have_declared_ownership():
    catalog = json.loads((DATA / "ecosystem.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (DATA / "plugin-workshop-metadata.json").read_text(encoding="utf-8")
    )
    taxonomy = json.loads((DATA / "mod-taxonomy.json").read_text(encoding="utf-8"))
    navigation = json.loads(
        (DATA / "plugin-workload-navigation.json").read_text(encoding="utf-8")
    )
    portfolio = json.loads(
        (DATA / "repository-portfolio.json").read_text(encoding="utf-8")
    )
    components = {item["id"]: item for item in catalog["components"]}
    expected = {
        "graph-coverage": "vllm-hust-graph-coverage",
        "prefill-admission": "vllm-hust-prefill-admission",
        "sglang-prepared-dispatch": "sglang-hust-prepared-dispatch",
    }
    for key, repository in expected.items():
        item = components[key]
        url = "https://github.com/vLLM-HUST/" + repository
        assert item["canonical_repository"] == url
        assert item["public_surface"] is True
        assert item["maintainers"] == ["ShuhaoZhangTony"]
        assert item["maintainer_profiles"] == [
            {"login": "ShuhaoZhangTony", "name": "张书豪"}
        ]
        assert item["advisors"] == []
        public = metadata["plugins"][key]
        assert [(p["login"], p["name"]) for p in public["maintainers"]] == [
            ("ShuhaoZhangTony", "张书豪")
        ]
        assert public["advisors"] == []
        assert navigation["plugins"][key]
        (record,) = [r for r in portfolio["repositories"] if r["name"] == repository]
        assert record["component_ids"] == [key]
        assert record["url"] == url
    assert taxonomy["components"]["graph-coverage"]["kind"] == "tool_mod"
    assert taxonomy["components"]["prefill-admission"]["kind"] == "tool_mod"
    assert (
        taxonomy["components"]["sglang-prepared-dispatch"]["lifecycle"]
        == "qualification_pending"
    )
