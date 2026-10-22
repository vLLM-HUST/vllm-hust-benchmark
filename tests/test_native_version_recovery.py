import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/native-version-recovery-20261022"
SPEC = importlib.util.spec_from_file_location(
    "native_version_recovery", REPORT / "audit.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_actual_archives_match_mapping():
    expected = json.loads((REPORT / "version-mapping.json").read_text())
    assert AUDIT.build([ROOT]) == expected
    assert len(expected["records"]) == 5
    assert all(not row["designated_018_baseline"] for row in expected["records"])


def test_rejects_018_relabeling():
    with pytest.raises(ValueError):
        AUDIT.parse_versions(b"vllm 0.18.0\nvllm_ascend 0.18.0\n")


def test_rejects_missing_and_duplicate_identity():
    valid = "".join(
        f"{key} {value}\n" for key, value in AUDIT.EXPECTED.items()
    ).encode()
    with pytest.raises(ValueError):
        AUDIT.parse_versions(b"vllm_path=/some/path\n")
    with pytest.raises(ValueError):
        AUDIT.parse_versions(valid + valid)
