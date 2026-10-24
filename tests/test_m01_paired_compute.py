import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pair_compute", ROOT / "scripts/build_m01_paired_compute.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_paired_compute_is_reproducible_and_not_reference_eta():
    actual = json.loads((BUILDER.DEST / "summary.json").read_text())
    assert actual == BUILDER.build()
    assert actual["rows"]["B0"]["F"] == 1829161955258989
    assert actual["rows"]["B1"]["F"] == 1812905151088034
    assert 84.97 < actual["C_gain_percent"] < 84.98
    assert actual["eta_percent"] is actual["attainment"] is None
    assert actual["reference_seconds"] is actual["reference_tflops_per_chip"] is None
    assert actual["baseline_is_reference_ceiling"] is False
    assert actual["measurement"]["same_run_F_T"]


def test_wrong_F_cannot_be_silently_published(tmp_path):
    import shutil
    import pytest

    for name in BUILDER.FILES:
        shutil.copyfile(BUILDER.DEST / name, tmp_path / name)
    path = tmp_path / BUILDER.FILES[0]
    data = json.loads(path.read_text())
    data["rows"]["B0"]["F"] += 1000000
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        BUILDER.build(tmp_path)
