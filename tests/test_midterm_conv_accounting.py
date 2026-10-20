"""Cross-check useful convolution arithmetic against immutable r9 evidence."""

from collections import Counter
import hashlib
import json
import subprocess
import sys

from test_midterm_vocab_prefill import BASE, archive_members


def test_archived_convolution_numerics(tmp_path):
    members = archive_members(BASE / "conv-accounting-r9")
    for name, content in members.items():
        if name.startswith("source/") and name.count("/") == 1 and name.endswith(".py"):
            (tmp_path / name.split("/")[-1]).write_bytes(content)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_conv_accounting.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_convolution_trajectory_and_disjoint_catalog():
    root = BASE / "conv-accounting-r9"
    members = archive_members(root)
    summary = json.loads((root / "accounting.json").read_text())
    combined = json.loads((root / "same-run-catalog.json").read_text())
    raw = archive_members(BASE / "device-routing-r9")
    for name, digest in summary["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    for rank in summary["ranks"]:
        trace = raw["microflow-token-accounting-r9/" + rank["trace"]]
        assert hashlib.sha256(trace).hexdigest() == rank["trace_sha256"]
        counts, histogram = Counter(), Counter()
        multiply = add = silu = 0
        for e in map(json.loads, trace.splitlines()):
            if e["event"] != "forward_enter" or e["role"] != "target":
                continue
            if (
                not summary["request_phase"]["start_wall"]
                <= e["wall"]
                <= summary["request_phase"]["end_wall"]
            ):
                continue
            counts["target_contexts"] += 1
            for group in e["owned_gdn"]:
                factor = len(group["layers"])
                for i, (a, b) in enumerate(zip(group["cu"], group["cu"][1:])):
                    role = "verify" if i in group["verify_ids"] else "prefill"
                    initial = role == "verify" or group["initial"][i]
                    n = b - a
                    histogram[role, n, initial] += factor
                    counts[role + "_token_layers"] += n * factor
                    counts["zero_initial_sequence_layers"] += int(not initial) * factor
                    taps = sum(4 if initial else min(t + 1, 4) for t in range(n))
                    multiply += taps * 4096 * factor
                    add += (taps - n) * 4096 * factor
                    silu += n * 4096 * factor
        assert counts == rank["counts"]
        assert histogram == {
            (c["role"], c["length"], c["initial"]): c["sequence_layers"]
            for c in rank["sequence_catalog"]
        }
        assert multiply == rank["operations"]["convolution_multiply"]
        assert add == rank["operations"]["convolution_add"]
        assert silu == rank["operations"]["silu_evaluations"]
        assert multiply + add == rank["basic_arithmetic_subtotal"]
    prior = json.loads(raw["source/same-run-partial-accounting-r9-v3.json"])
    assert (
        hashlib.sha256(raw["source/same-run-partial-accounting-r9-v3.json"]).hexdigest()
        == combined["previous_catalog_sha256"]
    )
    for key, value in prior["components"].items():
        assert combined["components"][key] == value
    assert (
        combined["components"]["gdn_causal_convolution_basic_ops"]
        == summary["basic_arithmetic_subtotal"]
    )
    for name, digest in combined["extension_input_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    assert (
        sum(combined["components"].values())
        == combined["covered_basic_semantic_subtotal"]
    )
    assert combined["model_effective_flops"] is combined["attainment_percent"] is None
