"""Restore incremental evidence and validate its relationship to frozen r19."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import pytest

from test_midterm_r19_logits import evidence as base_evidence  # noqa: F401

BASE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019"
    / "m01-recovery/r19-preprocess-supplement"
)
SOURCE = "root/gennova-evidence/20261019-midterm-aligned"
RUN = "workspace/gennova-measurements/microflow-token-accounting-r19"
PROBE = "workspace/gennova-measurements/publication-fence-v1-qualification"


@pytest.fixture(scope="module")
def recovered(request):
    base_root = request.getfixturevalue("base_evidence")
    index = json.loads((BASE / "archive.json").read_text())
    prior_path = BASE.parent / "r19-direct-logits/archive.json"
    assert (
        hashlib.sha256(prior_path.read_bytes()).hexdigest()
        == index["base_manifest_sha256"]
    )
    prior = json.loads(prior_path.read_text())
    assert prior["archive_sha256"] == index["base_archive_sha256"]
    old = {item["name"]: item for item in prior["members"]}
    assert all(old[item["name"]] == item for item in index["base_references"])
    packed = BASE / index["archive"]
    assert packed.stat().st_size == index["archive_bytes"]
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == index["archive_sha256"]
    expected = {item["name"]: item for item in index["members"]}
    assert not set(expected).intersection(old)
    seen = set()
    with tarfile.open(packed) as archive:
        for item in archive:
            path = PurePosixPath(item.name)
            assert item.isfile() and not path.is_absolute() and ".." not in path.parts
            assert item.name not in seen
            seen.add(item.name)
            data = archive.extractfile(item).read()
            assert len(data) == expected[item.name]["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected[item.name]["sha256"]
            target = base_root / item.name
            assert not target.exists()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    assert seen == set(expected)
    return base_root


def read(root, name):
    return json.loads((root / SOURCE / name).read_text())


def test_new_ledgers_provenance_and_scope(recovered):
    for name in (
        "r19-gdn-preprocess-semantics.json",
        "r19-accounted-families.json",
        "r19-host-mapping-qualification.json",
    ):
        item = read(recovered, name)
        assert item["model_effective_flops"] is item["m01_attainment"] is None
        for path, digest in item["source_sha256"].items():
            assert (
                hashlib.sha256((recovered / path.lstrip("/")).read_bytes()).hexdigest()
                == digest
            )
    result = read(recovered, "r19-accounted-families.json")
    assert (
        len({item["family"] for item in result["families"]})
        == len(result["families"])
        == 10
    )
    assert (
        sum(item["global_basic_operations"] for item in result["families"])
        == result["accounted_basic_operation_subtotal"]
    )


def test_preprocess_recomputed_from_device_cohorts(recovered, monkeypatch):
    monkeypatch.syspath_prepend(str(recovered / SOURCE))
    from audit_gdn_preprocess_semantics import operations, basic, verify_source

    verify_source(
        (
            recovered / SOURCE / "payload/betterscale/models/qwen35/preprocess.py"
        ).read_text()
    )
    raw = read(recovered, "r19-gdn-publication-summary.json")
    output = read(recovered, "r19-gdn-preprocess-semantics.json")
    for rank in raw["ranks"]:
        target = next(row for row in output["ranks"] if row["pid"] == rank["pid"])
        for name, counts in rank["cohorts"].items():
            active = counts.get("prefill_tokens_layers", 0) + counts.get(
                "device_active_verify_tokens_layers", 0
            )
            skipped = counts.get("device_skipped_verify_tokens_layers", 0)
            for scope, rows in [
                ("logical_valid_rows", active + skipped),
                ("state_update_eligible_rows", active),
            ]:
                actual = target["cohorts"][name]["scopes"][scope]
                assert actual["token_layers"] == rows
                assert actual["operations"] == operations(rows)
                assert (
                    actual["basic_arithmetic_subtotal"]
                    == basic(operations(rows))
                    == rows * 6192
                )


def test_partition_totals_from_original_publications(recovered):
    output = read(recovered, "r19-host-mapping-qualification.json")
    assert output["actual_device_map_contents_observed"] is False
    for rank in output["ranks"]:
        raw = json.loads(
            (recovered / RUN / f"gdn-publication-{rank['pid']}.json").read_text()
        )
        counts = dict(direct_tokens=0, mapped_prefill_tokens=0, mapped_verify_tokens=0)
        for row in raw["records"]:
            meta = row["metadata"]
            cu = [item[0] for item in row["device_fields"]["cu"]]
            assert cu == meta["host_cu"]
            lengths = [b - a for a, b in zip(cu, cu[1:])]
            assert sorted(meta["prefill_ids"] + meta["verify_ids"]) == list(
                range(len(lengths))
            )
            if meta["decode_only"]:
                assert not meta["prefill_ids"]
                counts["direct_tokens"] += sum(lengths)
            else:
                counts["mapped_prefill_tokens"] += sum(
                    lengths[i] for i in meta["prefill_ids"]
                )
                counts["mapped_verify_tokens"] += sum(
                    lengths[i] for i in meta["verify_ids"]
                )
        assert (
            counts == rank["tokens"]
            and len(raw["records"]) == rank["publications"] == 876
        )


def test_real_npu_result_does_not_overclaim_double_bank_contention(recovered):
    result = json.loads((recovered / PROBE / "device-result.json").read_text())
    receipt = json.loads((recovered / PROBE / "receipt.json").read_text())
    assert (
        receipt["status"] == "passed"
        and receipt["exit_code"] == 0
        and receipt["all_four_idle_after"]
    )
    for path, digest in result["source_sha256"].items():
        assert (
            hashlib.sha256((recovered / path.lstrip("/")).read_bytes()).hexdigest()
            == digest
        )
    assert result["model_run"] is result["performance_publication"] is False
    single, double = result["cases"]
    for case in (single, double):
        assert case["publications"] == 129 and case["payload_bytes"] == 65536
        assert (
            case["all_payload_bytes_matched"]
            and case["compute_stream"] not in case["publication_streams"]
        )
    assert single["pending_uploads_observed_before_acquire"] > 0
    assert single["pending_consumers_observed_before_reuse"] > 0
    assert (
        double["pending_uploads_observed_before_acquire"]
        == double["pending_consumers_observed_before_reuse"]
        == 0
    )
