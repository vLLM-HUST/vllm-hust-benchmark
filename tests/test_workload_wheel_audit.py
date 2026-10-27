"""Small fabricated software-wheel fixtures only."""

import importlib.util
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wheel_audit", ROOT / "scripts/audit_workload_data_wheel.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, *, change=False, extra=None):
    root = tmp_path / "software"
    package = root / "src/vllm_hust_workload_data"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""software fixture"""\n', encoding="utf-8")
    (root / "MIGRATION_PROVENANCE.json").write_text(
        '{"raw_datasets_included":false}', encoding="utf-8"
    )
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as bundle:
        for name, source in runner.expected_payload(root).items():
            bundle.writestr(name, b"changed" if change else source.read_bytes())
        bundle.writestr("vllm_hust_workload_data-0.1.0.dist-info/METADATA", "fixture")
        if extra:
            bundle.writestr(extra, "unexpected fixture")
    return wheel, root


def test_exact_current_source_bytes_verified_without_installation(tmp_path):
    wheel, root = fixture(tmp_path)
    result = runner.audit(wheel, root=root)
    assert len(result["verified_payloads"]) == 2
    assert not result["dataset_validation"]
    assert not result["installed_consumer_verified"]
    assert not result["source_checkbox_eligible"]


def test_changed_payload_is_rejected(tmp_path):
    wheel, root = fixture(tmp_path, change=True)
    with pytest.raises(ValueError, match="bytes differ"):
        runner.audit(wheel, root=root)


@pytest.mark.parametrize(
    "extra",
    [
        "vllm_hust_workload_data/research.py",
        "vllm_hust_workload_data/raw.jsonl",
        "vllm_hust_workload_data-0.1.0.dist-info/raw.csv",
        "../escape.dist-info/METADATA",
    ],
)
def test_unexpected_or_disguised_data_payload_rejected(tmp_path, extra):
    wheel, root = fixture(tmp_path, extra=extra)
    with pytest.raises(ValueError):
        runner.audit(wheel, root=root)
