"""Rebuild useful triangular work from the sealed r9 schedule and sources."""

from collections import Counter
import hashlib
import json
import subprocess
import sys

from test_midterm_vocab_prefill import BASE, archive_members


def test_archived_scalar_and_independent_inverse_checks(tmp_path):
    members = archive_members(BASE / "gdn-triangular-r9")
    for name, content in members.items():
        if name.startswith("source/") and name.count("/") == 1 and name.endswith(".py"):
            (tmp_path / name.split("/")[-1]).write_bytes(content)
    for name in ("recover_model_hardware.py", "integrate_hbm_samples.py"):
        (tmp_path / name).write_bytes((BASE.parent / name).read_bytes())
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test_gdn_wy_accounting.py",
            "test_gdn_inverse_accounting.py",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_r9_histogram_source_hashes_and_totals():
    members = archive_members(BASE / "gdn-triangular-r9")
    summary = json.loads((BASE / "gdn-triangular-r9/accounting.json").read_text())
    raw = archive_members(BASE / "device-routing-r9")
    inverse_module = {}
    exec(members["source/gdn_inverse_accounting.py"], inverse_module)
    for name, digest in summary["source_sha256"].items():
        assert hashlib.sha256(members["source/" + name]).hexdigest() == digest
    ranks = {rank["rank"]: rank for rank in summary["ranks"]}
    for name, content in raw.items():
        if not name.startswith("microflow-token-accounting-r9/token-accounting-"):
            continue
        events = list(map(json.loads, content.splitlines()))
        rank = next(e["rank"] for e in events if e["event"] == "observer_installed")
        assert hashlib.sha256(content).hexdigest() == ranks[rank]["trace_sha256"]
        histogram = Counter()
        for event in events:
            if event["event"] != "forward_enter" or event["role"] != "target":
                continue
            if (
                not summary["request_phase"]["start_wall"]
                <= event["wall"]
                <= summary["request_phase"]["end_wall"]
            ):
                continue
            assert sum(len(g["layers"]) for g in event["owned_gdn"]) == 30
            for group in event["owned_gdn"]:
                if group["decode_only"]:
                    continue
                for a, b in zip(group["prefill_cu"], group["prefill_cu"][1:]):
                    for offset in range(0, b - a, 64):
                        histogram[str(min(64, b - a - offset))] += len(group["layers"])
        assert histogram == ranks[rank]["chunk_token_layers_histogram"]
        totals = Counter()
        for text, count in histogram.items():
            n = int(text)
            strict = sum(i for i in range(n))
            inclusive = sum(i + 1 for i in range(n))
            per_head = {
                "kkt_dot": 256 * strict,
                "kkt_gate_difference_and_scaling": 3 * strict,
                "wy_value_beta": 128 * n,
                "wy_key_beta_gate": 256 * n,
                "wy_lower_triangular_dot": 512 * inclusive,
                "exp_evaluations": strict + n,
                **inverse_module["inverse_arithmetic"](n),
            }
            totals.update({k: v * count * 16 for k, v in per_head.items()})
        assert totals == ranks[rank]["operations"]
        assert (
            sum(v for k, v in totals.items() if k != "exp_evaluations")
            == ranks[rank]["basic_arithmetic_subtotal"]
        )
    assert summary["basic_arithmetic_subtotal"] == sum(
        r["basic_arithmetic_subtotal"] for r in ranks.values()
    )
    assert summary["model_effective_flops"] is None
    assert summary["attainment_percent"] is None
