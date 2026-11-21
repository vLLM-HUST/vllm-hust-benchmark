"""Candidate publication must preserve the original baseline without inventing pairs."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fixed_baseline", ROOT / "scripts/build_fixed_baseline_observations.py"
)
producer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(producer)


def test_original_baseline_is_independent_and_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_fixed_baseline_observations.py"),
            "--check",
        ],
        check=True,
    )
    value = producer.build()
    assert [r["arm"]["requests"] for r in value["profiles"]] == [200, 200, 160, 64, 48]
    assert all(r["arm"]["role"] == "B0" for r in value["profiles"])
    assert len(value["context"]["observations"]) == 4
    assert all("B1" not in r for r in value["context"]["observations"])
    assert len(value["compute"]["arm"]["repetitions"]) == 3
    catalog = json.loads((ROOT / producer.CATALOG).read_text())
    candidate = next(e for e in catalog["entries"] if e["id"] == "r9")
    original = {row["id"]: row["arm"] for row in value["profiles"]}
    # Existing paired observations must use the independent fixed B0 values.
    # Candidate completeness never controls whether the baseline is preserved.
    for row in candidate["measurements"]:
        if row["id"] not in original:
            continue
        assert row["control"] == "B0"
        for key in (
            "role",
            "requests",
            "output_tokens",
            "output_tokens_per_second",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "source_seal_sha256",
        ):
            assert row["arms"][0][key] == original[row["id"]][key]
    assert catalog["baseline"]["observations"]["sha256"] == producer.sha(
        producer.render(value)
    )


def test_future_empty_candidate_does_not_clear_baseline(tmp_path):
    for name in [producer.CATALOG, *producer.SOURCES.values()]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    p = tmp_path / producer.CATALOG
    catalog = json.loads(p.read_bytes())
    catalog["entries"] = [
        dict(id="r100", measurements=[], baseline_id=catalog["baseline"]["id"])
    ]
    catalog["order"] = ["r100"]
    catalog["default_release"] = "r100"
    p.write_text(json.dumps(catalog))
    assert producer.build(tmp_path) == producer.build()


def test_optimized_arm_cannot_be_relabelled_as_the_fixed_baseline(tmp_path):
    for name in [producer.CATALOG, *producer.SOURCES.values()]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    p = tmp_path / producer.SOURCES["profiles"]
    data = json.loads(p.read_bytes())
    data["measurements"][0]["arms"].reverse()
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        producer.build(tmp_path)
