import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/vllm-hust-midterm-20261019"


def test_original_probe_bytes_match_preexisting_report_seal():
    receipt = json.loads((REPORT / "long-context-originals.json").read_text())
    source = json.loads((REPORT / "results.json").read_text())
    raw = (REPORT / receipt["archive"]).read_bytes()
    assert len(raw) == receipt["bytes"]
    assert hashlib.sha256(raw).hexdigest() == receipt["sha256"]
    expected = {r["path"]: r for r in receipt["members"]}
    assert len(expected) == 172
    reports = []
    with tarfile.open(REPORT / receipt["archive"]) as archive:
        assert set(archive.getnames()) == set(expected)
        for member in archive.getmembers():
            assert member.isfile()
            raw = archive.extractfile(member).read()
            sealed = expected[member.name]
            assert len(raw) == sealed["bytes"]
            assert hashlib.sha256(raw).hexdigest() == sealed["sha256"]
            assert (
                source["source_sha256"]["/root/gennova-evidence/" + member.name]
                == sealed["sha256"]
            )
            if member.name.endswith("/long-context-report.json"):
                reports.append(json.loads(raw))
    assert len(reports) == 4
