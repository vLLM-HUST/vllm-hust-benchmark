import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website-publication/data"
PATTERN = re.compile(
    r"https://github.com/vLLM-HUST/vllm-hust-benchmark/(?:blob|tree)/"
    r"[0-9a-f]{40}/([^#]+)(?:#.*)?$"
)


def check(url):
    match = PATTERN.fullmatch(url)
    assert match, url
    assert (ROOT / match[1]).exists(), url


def test_all_performance_panel_sources_are_pinned_benchmark_links():
    entries = json.loads((DATA / "plugin-performance.json").read_text())["entries"]
    for entry in entries:
        check(entry["url"])
        for observation in entry["observations"]:
            if "url" in observation:
                check(observation["url"])


def test_measured_card_sources_and_access_limits():
    components = json.loads((DATA / "ecosystem.json").read_text())["components"]
    for component in components:
        if component.get("public_effect_status") in {
            "measured",
            "beneficial",
            "not-beneficial-in-tested-cell",
            "inconclusive",
        }:
            url = component["public_effect_url"]
            if url.startswith("./"):
                assert component["id"] == "betterscale"
            else:
                check(url)
    by_id = {row["id"]: row for row in components}
    for name in ("latchmoe", "request-throttling-controller"):
        assert "not publicly retrievable" in by_id[name]["public_effect_en"]
        assert "无法公开访问" in by_id[name]["public_effect_zh"]
    check(by_id["pyramidkv-ascend-migration"]["public_effect_url"])
