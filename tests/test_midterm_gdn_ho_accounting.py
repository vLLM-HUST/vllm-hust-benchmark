"""Verify H/O equations, source provenance and r9 chunk-state coverage."""

from collections import Counter
import hashlib
import json
import subprocess
import sys

from test_midterm_vocab_prefill import BASE, archive_members


def test_archived_ho_equations_and_count_boundaries(tmp_path):
    members = archive_members(BASE / "gdn-ho-accounting-r9")
    for name, content in members.items():
        if name.startswith("source/") and name.count("/") == 1 and name.endswith(".py"):
            (tmp_path / name.split("/")[-1]).write_bytes(content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_gdn_ho_accounting.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_r9_chunk_states_and_native_source_manifest():
    root = BASE / "gdn-ho-accounting-r9"
    members = archive_members(root)
    summary = json.loads((root / "accounting.json").read_text())
    source_data = members["gdn-native-source-r8/source-receipt.json"]
    assert hashlib.sha256(source_data).hexdigest() == summary["source_receipt_sha256"]
    for name, digest in json.loads(source_data)["source_sha256"].items():
        assert (
            hashlib.sha256(members["gdn-native-source-r8/" + name]).hexdigest()
            == digest
        )
    assert (
        hashlib.sha256(
            members["source/payload/betterscale/models/qwen35/mixed_layout.py"]
        ).hexdigest()
        == summary["scan_source_sha256"]
    )
    raw = archive_members(BASE / "device-routing-r9")
    for rank in summary["ranks"]:
        data = raw["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(data).hexdigest() == rank["trace_sha256"]
        histogram = Counter()
        for event in map(json.loads, data.splitlines()):
            if event["event"] != "forward_enter" or event["role"] != "target":
                continue
            if (
                not summary["request_phase"]["start_wall"]
                <= event["wall"]
                <= summary["request_phase"]["end_wall"]
            ):
                continue
            for group in event["owned_gdn"]:
                if group["decode_only"]:
                    continue
                for i, (a, b) in enumerate(zip(group["cu"], group["cu"][1:])):
                    if i in group["verify_ids"]:
                        continue
                    for start in range(0, b - a, 64):
                        histogram[
                            min(64, b - a - start),
                            bool(group["initial"][i]) or start > 0,
                        ] += len(group["layers"])
        assert histogram == {
            (r["length"], r["has_state"]): r["chunk_layers"]
            for r in rank["chunk_catalog"]
        }
        expected = Counter()
        for (n, state), count in histogram.items():
            pairs = sum(range(1, n + 1))
            terms = {
                "prefix_scan_add": n - 1,
                "h_wh_dot": 32768 * n if state else 0,
                "h_value_residual_subtract": 128 * n if state else 0,
                "h_decay_difference": n,
                "h_value_decay_multiply": 128 * n,
                "h_kv_dot": 32768 * n,
                "h_state_decay_and_add": 32768 if state else 0,
                "o_qk_causal_dot": 256 * pairs,
                "o_gate_difference_and_multiply": 2 * pairs,
                "o_qh_dot": 32768 * n if state else 0,
                "o_av_causal_dot": 256 * pairs,
                "o_state_decay_and_add": 256 * n if state else 0,
                "o_output_scale": 128 * n,
                "exp_evaluations": n + pairs + (n + 1 if state else 0),
                "clamp_comparisons": pairs,
            }
            expected.update({k: v * count * 16 for k, v in terms.items()})
        assert expected == rank["operations"]
        assert rank["basic_arithmetic_subtotal"] == sum(
            v
            for k, v in expected.items()
            if k not in ("exp_evaluations", "clamp_comparisons")
        )
    combined = json.loads((root / "same-run-catalog.json").read_text())
    prior_data = members["source/same-run-partial-accounting-r9-v4.json"]
    assert hashlib.sha256(prior_data).hexdigest() == combined["previous_catalog_sha256"]
    prior = json.loads(prior_data)
    assert (
        combined["request_phase"] == prior["request_phase"] == summary["request_phase"]
    )
    for key, value in prior["components"].items():
        assert combined["components"][key] == value
    assert (
        combined["components"]["gdn_prefill_ho_and_scan_basic_ops"]
        == summary["basic_arithmetic_subtotal"]
    )
    assert (
        sum(combined["components"].values())
        == combined["covered_basic_semantic_subtotal"]
    )
    assert combined["model_effective_flops"] is combined["attainment_percent"] is None
