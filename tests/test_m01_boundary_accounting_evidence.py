import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/vllm-hust-midterm-20261019/m01-recovery"


def test_boundary_accounting_identity_and_arithmetic():
    evidence = BASE / "boundary-accounting-20261023"
    manifest = json.loads((evidence / "manifest.json").read_text())
    files = {}
    for item in manifest["archives"]:
        archive = evidence / item["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["sha256"]
        members = {}
        with tarfile.open(archive) as bundle:
            for member in bundle:
                assert not Path(member.name).is_absolute()
                assert ".." not in Path(member.name).parts
                assert member.isdir() or member.isfile()
                if member.isfile():
                    assert member.name not in files
                    raw = bundle.extractfile(member).read()
                    files[member.name] = raw
                    members[member.name] = hashlib.sha256(raw).hexdigest()
        assert members == item["members"]
    records = {
        Path(name).name: json.loads(raw)
        for name, raw in files.items()
        if name.endswith(".json")
    }
    measured = BASE / "boundary-measured-20261023"
    measured_manifest = json.loads((measured / "manifest.json").read_text())
    with tarfile.open(measured / measured_manifest["archive"]) as bundle:
        raw = bundle.extractfile("m01-boundary-b1-r1/mmlu-pro.json").read()
    bench = json.loads(raw)
    join = records["boundary-workload-join-r1.json"]
    assert (
        hashlib.sha256(raw).hexdigest()
        == join["source_sha256"]["/workspace/m01-boundary-b1-r1/mmlu-pro.json"]
    )
    lengths = [
        i + o - 1
        for i, o in zip(bench["input_lens"], bench["output_lens"], strict=True)
    ]
    assert sum(lengths) == join["candidate_target_positions"]
    assert sum(n * (n + 1) // 2 for n in lengths) == join["candidate_causal_pairs"]
    assert not join["old_draft_count_reused"]
    assert not join["old_execution_shape_counts_reused"]
    matrix = records["boundary-matrix-demand-r1.json"]
    assert matrix["target_projection_flops"] == sum(lengths) * sum(
        matrix["coefficients_per_position"].values()
    )
    assert matrix["partial_target_matrix_flops"] == sum(
        matrix[k]
        for k in (
            "target_projection_flops",
            "causal_qk_av_flops",
            "target_output_vocabulary_flops",
        )
    )
    scalar = records["boundary-scalar-demand-r1.json"]
    assert scalar["listed_target_subtotal"] == (
        matrix["partial_target_matrix_flops"] + sum(scalar["basic_parts"].values())
    )
    gdn = records["boundary-gdn-prefill-partition-bounds-r1.json"]
    rows = gdn["per_request_single_head_layer"]
    assert [r["length"] for r in rows] == bench["input_lens"]
    for field in ("minimum", "maximum"):
        assert gdn[field + "_basic_flops"] == sum(r[field] for r in rows) * 30 * 32
    assert (
        gdn["interval_width"] == gdn["maximum_basic_flops"] - gdn["minimum_basic_flops"]
    )
    norm = records["boundary-norm-consumers-r1.json"]
    assert norm["applied_exclusion_flops"] == 0
    assert (
        norm["hypothetical_logits_only_exclusion_basic_flops"]
        == (sum(lengths) - sum(bench["output_lens"])) * 10241
    )
    for record in [manifest, *records.values()]:
        assert record["effective_flops"] is None
        assert record["efficiency_percent"] is None
    assert manifest["m01_attainment"] is None
