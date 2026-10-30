"""Current cards retain measurements while reporting later correctness findings."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "website-publication/data"


def read(path):
    return json.loads(path.read_text())


def test_simllm_current_effect_matches_retained_diagnostic():
    result = read(ROOT / "reports/simllm-issue3-targeted-20261030/comparisons.json")
    card = next(
        c
        for c in read(DATA / "ecosystem.json")["components"]
        if c["id"] == "simllm-migration"
    )
    assert result["correct_by_arm"]["approximate"] == 5
    assert result["correct_by_arm"]["baseline"] == result["request_count"] == 12
    for lang in ("en", "zh"):
        assert "5/12" in card[f"public_effect_{lang}"]
        assert "12/12" in card[f"public_effect_{lang}"]
        assert "910B3" in card[f"public_effect_{lang}"]
    assert card["public_effect_status"] == "not-beneficial-in-tested-cell"
    assert card["public_effect_url"].endswith(
        "reports/simllm-issue3-targeted-20261030/README.md"
    )
    historical = next(
        e
        for e in read(DATA / "plugin-performance.json")["entries"]
        if e["id"] == "simllm-migration"
    )["observations"][0]
    assert historical["diagnostic_only"] is True
    assert [(r["baseline"], r["candidate"]) for r in historical["comparisons"]] == [
        (5.1964, 6.2558),
        (5.2563, 6.2435),
    ]


def test_all_effect_summaries_have_bilingual_scope_and_evidence():
    for card in read(DATA / "ecosystem.json")["components"]:
        if card.get("public_effect_status"):
            assert card["public_effect_en"] and card["public_effect_zh"]
            assert card["public_effect_url"]
