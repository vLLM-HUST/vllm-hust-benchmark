"""GenNova candidate evidence is distinct from an accepted B1 replacement."""

import hashlib
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1] / "reports/gennova-mmlu-pro-20261023"


def test_gennova_capsules_preserve_member_bytes():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["schema"] == "gennova-independent-engine-evidence/v1"
    assert manifest["b1_replacement_approved"] is False
    assert manifest["full_dataset_quality_score"] is None
    labels = set()
    for row in manifest["archives"]:
        labels.add(row["run"])
        archive = ROOT / row["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["sha256"]
        with tarfile.open(archive) as capsule:
            files = {member.name: member for member in capsule if member.isfile()}
            assert files.keys() == row["members"].keys()
            for name, expected in row["members"].items():
                assert ".." not in Path(name).parts
                assert not Path(name).is_absolute()
                with capsule.extractfile(files[name]) as stream:
                    assert hashlib.sha256(stream.read()).hexdigest() == expected
            if row["run"] != "protocol":
                receipt = json.load(
                    capsule.extractfile(files[row["run"] + "/receipt.json"])
                )
                assert receipt["all_devices_released"]
                assert receipt["status"] == row["status"]
    assert {
        "native-r1",
        "checkpoint-r1",
        "checkpoint-r2",
        "engine-r1",
        "b1-r1",
        "protocol",
    } <= labels


def test_candidate_validity_does_not_claim_b1_superiority():
    with tarfile.open(ROOT / "protocol-and-tools.tar.gz") as capsule:
        adjudication = json.load(capsule.extractfile("protocol/adjudication.json"))
    assert adjudication["b1-r2"]["invalid_as_current_best_b1_comparison"] is True
    assert (
        adjudication["comparison-unqualified-r2.json"]["status"]
        == "rejected-for-b1-comparison"
    )
    with tarfile.open(ROOT / "engine-r1.tar.gz") as capsule:
        ranks = [
            json.load(capsule.extractfile(f"engine-r1/rank-{i}.json")) for i in (0, 1)
        ]
    assert ranks[0]["outputs"] == ranks[1]["outputs"]
    for rank in ranks:
        assert rank["status"] == "completed"
        assert rank["total_input_tokens"] == 282265
        assert rank["total_output_tokens"] == 51200
        assert len(rank["outputs"]) == 200
        assert all(len(tokens) == 256 for tokens in rank["outputs"].values())
        assert rank["http_comparable"] is False
