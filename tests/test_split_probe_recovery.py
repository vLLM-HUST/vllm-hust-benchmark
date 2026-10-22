import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "archive/mod-git-20261022"


def test_recovery_preserves_identity_and_missing_originals():
    audit = json.loads(
        (FOLDER / "split-probe-recovery/local-reference-audit.json").read_bytes()
    )
    previous = (ROOT / audit["previous_audit"]["path"]).read_bytes()
    assert hashlib.sha256(previous).hexdigest() == audit["previous_audit"]["sha256"]
    old = json.loads(previous)["references"]
    assert len(old) == len(audit["references"]) == 39
    recovered = 0
    for before, after in zip(old, audit["references"], strict=True):
        for key in ("original_reference", "source_file", "location", "declared_sha256"):
            assert before[key] == after[key]
        if "recovery" not in after:
            assert before == after
            continue
        recovered += 1
        record = after["recovery"]
        assert before["canonical_raw_url"] is None
        assert after["status"] == "recovered-by-source-map-and-sha256"
        for kind in ("raw", "proof"):
            data = (ROOT / record[kind]["path"]).read_bytes()
            assert len(data) == record[kind]["bytes"]
            assert hashlib.sha256(data).hexdigest() == record[kind]["sha256"]
        mapping = json.loads((ROOT / record["proof"]["path"]).read_bytes())
        for part in record["json_pointer"].lstrip("/").split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            mapping = mapping[int(key)] if isinstance(mapping, list) else mapping[key]
        assert mapping[record["mapping_key"]] == after["original_reference"]
        assert mapping["sha256"] == record["raw"]["sha256"]
        assert after["canonical_raw_url"] == (
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
            + record["commit"]
            + "/"
            + record["raw"]["path"]
        )
        assert record["download_verified"] is True
    assert recovered == 4
    assert sum(r["canonical_raw_url"] is None for r in audit["references"]) == 33
