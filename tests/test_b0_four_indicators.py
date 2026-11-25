"""Keep B0 four-indicator publication bound to the actual source files."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_b0_indicators", ROOT / "scripts/b0_four_indicators.py"
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


def test_b0_four_indicators_match_qualified_frozen_sources():
    actual = publication.build(publication.PUBLIC)
    published = publication.PUBLIC / publication.OUTPUT
    assert published.read_text(encoding="utf-8") == publication.encode(actual)
    for baseline, values in actual["baselines"].items():
        assert set(values) == {"4.1", "4.2", "4.3", "4.4"}
        assert len(values["4.2"]["profiles"]) == 5
        assert len(values["4.3"]["phases"]) == 4
        assert ("repair" in values["4.1"]["identity"]) == (
            baseline != actual["official_v2_baseline_id"]
        )
