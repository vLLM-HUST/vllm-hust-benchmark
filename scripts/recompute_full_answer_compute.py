"""Recompute the identifiable target-matrix subtotal; never impute total F or eta."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = "reports/full-answer-compute-20261101"
PAIR = "reports/full-answer-joint-20261030/joint-summary.json"
PUBLICATION = "website-publication/data/full-answer-compute-20261101.json"


def positive_integer(value):
    if type(value) is not int or value <= 0:
        raise ValueError("Positive integer extent required")
    return value


def coefficients(config):
    """Global TP work counted once; 2MNK, activated experts, unpadded shapes."""
    c = config["text_config"]
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "head_dim",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
        "num_experts",
        "vocab_size",
    )
    for key in keys:
        positive_integer(c[key])
    if c.get("attn_output_gate") is not True or c.get("attention_bias") is not False:
        raise ValueError("This derivation requires gated, bias-free Qwen3.5")
    layers = c["num_hidden_layers"]
    fa = c["layer_types"].count("full_attention")
    gdn = c["layer_types"].count("linear_attention")
    if fa + gdn != layers or c["num_experts_per_tok"] > c["num_experts"]:
        raise ValueError("Unsupported layers or expert count")
    h = c["hidden_size"]
    q = c["num_attention_heads"] * c["head_dim"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    k = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    v = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    return {
        "projection_per_position": {
            "full_attention": fa * 2 * h * (3 * q + 2 * kv),
            "linear_attention": gdn
            * 2
            * h
            * (2 * k + 3 * v + 2 * c["linear_num_value_heads"]),
            "moe_and_router": layers
            * 2
            * h
            * (
                3
                * (
                    c["num_experts_per_tok"] * c["moe_intermediate_size"]
                    + c["shared_expert_intermediate_size"]
                )
                + c["num_experts"]
                + 1
            ),
        },
        "causal_qk_av_per_pair": 4 * fa * q,
        "vocabulary_per_output": 2 * h * c["vocab_size"],
    }


def calculate(pair, config):
    co = coefficients(config)
    if (
        pair["protocol"]["reserved_cards"] != 2
        or pair["protocol"]["tensor_parallel"] != 2
    ):
        raise ValueError("The audited TP2/two-chip protocol is required")
    if [a["role"] for a in pair["arms"]] != ["B0", "B1"]:
        raise ValueError("Ordered B0/B1 pair required")
    arms = []
    identities = []
    for arm in pair["arms"]:
        rows = arm["rows"]
        if arm["completed"] != len(rows) or arm["failed"] != 0 or not rows:
            raise ValueError("Complete original cohort required")
        ids = [row["case_id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate request identity")
        inputs = [positive_integer(row["prompt_tokens"]) for row in rows]
        outputs = [positive_integer(row["output_tokens"]) for row in rows]
        identities.append(list(zip(ids, inputs, strict=True)))
        if sum(outputs) != arm["successful_output_tokens"]:
            raise ValueError("Output count mismatch")
        seconds = arm["phase_seconds"]
        if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Finite positive same-run client time required")
        positions = [i + o - 1 for i, o in zip(inputs, outputs, strict=True)]
        n = sum(positions)
        causal_pairs = sum(x * (x + 1) // 2 for x in positions)
        parts = {
            "target_projections": sum(co["projection_per_position"].values()) * n,
            "target_causal_qk_av": co["causal_qk_av_per_pair"] * causal_pairs,
            "target_vocabulary": co["vocabulary_per_output"] * sum(outputs),
        }
        subtotal = sum(parts.values())
        arms.append(
            {
                "role": arm["role"],
                "run": arm["run"],
                "requests": len(rows),
                "input_tokens": sum(inputs),
                "output_tokens": sum(outputs),
                "necessary_target_positions": n,
                "causal_pairs": causal_pairs,
                "matrix_flops_parts": parts,
                "matrix_flops_subtotal": subtotal,
                "client_seconds": seconds,
                "chips": 2,
                "matrix_subtotal_tflops_per_chip": subtotal / (2 * seconds * 1e12),
                "effective_flops_total": None,
                "effective_tflops_per_chip": None,
                "reference_seconds": None,
                "efficiency_percent": None,
                "outline_pass": None,
            }
        )
    if identities[0] != identities[1]:
        raise ValueError("Paired input identities and extents must match")
    return {
        "schema": "full-answer-compute-subtotal/v1",
        "status": "target_matrix_subtotal_recomputed_total_requires_measurement",
        "coefficients": co,
        "arms": arms,
        "matrix_subtotal_rate_gain_percent": 100
        * (
            arms[1]["matrix_subtotal_tflops_per_chip"]
            / arms[0]["matrix_subtotal_tflops_per_chip"]
            - 1
        ),
        "timing_basis": "Original client submission-to-response-drain interval, all overhead retained; no synchronized worker boundary receipt.",
        "missing_total_work": [
            "Same-run retained MTP outputs and their necessary KV/hidden dependency closure",
            "Same-run GDN prefill chunk/state ledger and remaining basic scalar operations",
        ],
        "missing_acceptance": [
            "Matched synchronized uninstrumented timing",
            "Independent calibrated dependency/resource reference for each full-answer workload/path",
        ],
        "convention": "I+O-1 target positions, triangular causal pairs, top8 activated experts; vocabulary for O target decisions. Count global TP2 work once. Exclude padding and rejected speculative work. This subtotal is not complete F or outline C.",
        "report_path": REPORT + "/README.md",
    }


def build(root):
    paths = [PAIR, REPORT + "/model-config.json", REPORT + "/source-audit.json"]
    raw = {path: (root / path).read_bytes() for path in paths}
    result = calculate(json.loads(raw[PAIR]), json.loads(raw[paths[1]]))
    audit = json.loads(raw[paths[2]])
    if audit["model_config_sha256"] != hashlib.sha256(raw[paths[1]]).hexdigest():
        raise ValueError("Frozen model configuration identity mismatch")
    for arm, record in zip(result["arms"], audit["arms"], strict=True):
        if (
            arm["run"] != record["run"]
            or arm["requests"] != record["matched_cases"]
            or not record["request_salts_unique"]
        ):
            raise ValueError("Original request audit mismatch")
    result["sources"] = {p: hashlib.sha256(b).hexdigest() for p, b in raw.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("Choose --check or --write")
    result = build(ROOT)
    raw = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode()
    for name in (REPORT + "/summary.json", PUBLICATION):
        path = ROOT / name
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        elif args.check and path.read_bytes() != raw:
            raise SystemExit("Stale derived result: " + name)
    print(json.dumps({"status": result["status"], "arms": result["arms"]}, indent=2))


if __name__ == "__main__":
    main()
