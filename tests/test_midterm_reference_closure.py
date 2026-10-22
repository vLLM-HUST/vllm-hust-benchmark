import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
EVIDENCE = ROOT / "m01-recovery/reference-closure-20261022"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def test_archive_identity_and_no_formal_result_promotion():
    manifest = read("archive.json")
    assert len(manifest["archives"]) == 6
    assert len(manifest["selected_members"]) == 8
    assert manifest["formal_b1_admission"] is False
    assert manifest["m01_attainment"] is manifest["model_effective_flops"] is None
    for item in manifest["selected_members"]:
        assert (
            hashlib.sha256((EVIDENCE / item["name"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
        assert item["archive"] in {a["name"] for a in manifest["archives"]}


@pytest.mark.parametrize(
    "name,cases,calls",
    [
        ("recurrence.json", 872, 26520),
        ("wy.json", 135, 4230),
        ("convolution.json", 1013, 30750),
    ],
)
def test_component_coverage(name, cases, calls):
    data = read(name)
    assert data["calls_per_rank"] == calls
    assert {r["rank"] for r in data["ranks"]} == {0, 1}
    for rank in data["ranks"]:
        assert rank["cases"] == cases
        assert len(rank["weighted_reference_ms"]) == 3
        assert rank["median_ms"] == statistics.median(rank["weighted_reference_ms"])


def test_resource_sums_use_all_repetitions():
    data = read("resource-terms.json")
    assert data["m01_attainment"] is None
    rows = data["compute_components"]
    assert len({(r["rank"], r["component"]) for r in rows}) == len(rows) == 36
    for summary in data["compute_rank_sums"]:
        selected = [r for r in rows if r["rank"] == summary["rank"]]
        assert summary["components"] == 18
        expected = (
            sum(statistics.mean(r["weighted_repetition_ms"]) for r in selected) / 1000
        )
        assert summary["sum_mean_seconds"] == pytest.approx(expected, rel=1e-12)
        assert summary["sum_min_seconds"] <= expected <= summary["sum_max_seconds"]


def test_physical_mix_mismatch_is_not_hidden():
    audit = read("hbm-mix-review.json")
    for row in audit["repetitions"]:
        assert row["direction_constrained_reference_seconds"] == max(
            row["model_direction_reference_seconds"].values()
        )
        assert (
            row["direction_constrained_reference_seconds"]
            > row["pooled_reference_seconds"]
        )
        assert all(
            0.03 < value < 0.04 for value in row["relative_read_write_mix_error"]
        )
    assert audit["m01_attainment"] is None


def test_captured_maps_close_old_gap():
    audit = read("preprocess-maps.json")
    assert (
        audit["status"]
        == "captured_device_final_preprocess_maps_and_semantic_extents_verified"
    )
    for rank in audit["ranks"]:
        assert rank["contexts"] == 884
        assert rank["mixed_groups"] + rank["decode_groups"] == 884
        assert rank["compared_map_elements"] == 709032
        assert rank["valid_token_layers"] == 10197480


def test_corrupt_archive_rejected_before_output(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "reference_package", ROOT / "package_reference_closure_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ARCHIVES", {"broken": "0" * 64})
    (tmp_path / "broken-20261022.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest mismatch"):
        module.package(tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
