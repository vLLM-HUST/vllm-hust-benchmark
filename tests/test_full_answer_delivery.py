"""Offline consistency of the immutable V1.03 delivery receipt, not a live check."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def receipt():
    return json.loads(
        (ROOT / "reports/full-answer-joint-20261030/delivery-v103.json").read_text()
    )


def test_delivery_matches_published_payload():
    data = receipt()
    assert data["version"] == "V1.03"
    raw = (ROOT / "website-publication/data/midterm-v103.json").read_bytes()
    assert data["production_bytes"] == [
        {
            "path": "data/midterm-v103.json",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
    ]
    assert data["production_browser_combinations"] == 24
    assert data["results_first_combinations"] == 16


def test_delivery_success_and_custody_are_bound():
    data = receipt()
    for run in (data["pages"], data["production_browser_run"]):
        assert run["conclusion"] == "success"
        assert run["headSha"] == data["website"]
    for pr in data["pull_requests"]:
        assert pr["state"] == "MERGED"
        assert pr["statusCheckRollup"]
        assert all(c["conclusion"] == "SUCCESS" for c in pr["statusCheckRollup"])
    assert data["archive"]["state"] == "uploaded"
    assert data["archive"]["digest"] == "sha256:" + data["archive_sha256"]
    assert data["release_download_sha256"] == data["archive_sha256"]
    assert data["private_storage_readback"]["sha256"] == data["archive_sha256"]
    assert data["private_storage_readback"]["verified_manifest_members"] == 45
    assert len(data["remaining"]) == 2


def test_live_manifest_capture_is_original_and_bound():
    data = receipt()
    capture = data["full_manifest_capture"]
    raw = (ROOT / capture["manifest_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == capture["manifest_sha256"]
    assert len(raw) == capture["manifest_bytes"]
    assert capture["run"]["conclusion"] == "success"
    assert capture["publication_identity"]["benchmark_commit"] == data["benchmark"]
    # The immutable delivery is bound to its captured manifest, not the mutable
    # navigation snapshot, which legitimately follows later deployments.
    manifest = json.loads(raw)
    assert manifest["schema"] == "benchmark-website-publication/v1"
    assert manifest["repository"] == "vLLM-HUST/vllm-hust-benchmark"
    assert manifest["commit"] == data["benchmark"]
    assert manifest["status"] == "published-source"
    published_file = capture["publication_identity"]["file"]
    assert published_file in manifest["files"]
    assert (
        capture["publication_identity"]["manifest"]["sha256"]
        == capture["manifest_sha256"]
    )
    assert capture["publication_identity"]["manifest"]["bytes"] == len(raw)
    assert capture["previous_local_transport"]["prior_urllib_error"]
