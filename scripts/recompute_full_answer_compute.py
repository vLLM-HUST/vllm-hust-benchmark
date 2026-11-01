"""Recompute the identifiable target-matrix subtotal; never impute total F or eta."""

import argparse
from collections import Counter
import gzip
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


def draft_dependencies(rows):
    """MTP2 version closure in linear space; terminal outputs clip accepted roots."""
    total = Counter()
    for row in rows:
        output = positive_integer(row["output_tokens"])
        selected = positive_integer(row["prompt_tokens"]) - 1
        emitted, kv0, first, second, pairs = 1, 0, 0, 0, 0
        accepted, drafted = row["per_step_accepted"], row["per_step_drafted"]
        if len(accepted) != len(drafted):
            raise ValueError("MTP sequence lengths differ")
        for count, proposed in zip(accepted, drafted, strict=True):
            if type(count) is not int or count not in (0, 1, 2) or proposed != 2:
                raise ValueError("Ordered MTP2 counts required")
            if emitted >= output:
                raise ValueError("MTP steps after terminal output")
            keep = min(count + 1, output - emitted)
            retained = min(count, keep)
            if retained:
                kv0 = selected + 1
                first += 1
                pairs += selected + 1
            if retained == 2:
                second += 1
                pairs += selected + 2
            emitted += keep
            selected += count + 1
        if emitted != output:
            raise ValueError("MTP sequence does not close emitted output")
        total.update(
            kv_rows=kv0 + second,
            full_query_rows=first + second,
            retained_vocab_rows=first + second,
            attention_pairs=pairs,
            first_retained=first,
            second_retained=second,
            draft_steps=len(accepted),
            raw_accepted=sum(accepted),
        )
    return dict(total)


def partition_bounds(lengths, initial, continuation):
    """Exact extrema over all compositions into positive chunks of at most 64."""
    maximum = max(positive_integer(n) for n in lengths)
    if set(initial) != set(range(1, 65)) or set(continuation) != set(range(1, 65)):
        raise ValueError("Complete 1..64 chunk catalog required")
    low, high = [0], [0]
    for n in range(1, maximum + 1):
        low.append(
            min(
                initial[k] if n == k else low[n - k] + continuation[k]
                for k in range(1, min(64, n) + 1)
            )
        )
        high.append(
            max(
                initial[k] if n == k else high[n - k] + continuation[k]
                for k in range(1, min(64, n) + 1)
            )
        )
    return [sum(low[n] for n in lengths), sum(high[n] for n in lengths)]


def extended_b1(pair, result, rows, catalog):
    original = pair["arms"][1]
    if len(rows) != len(original["rows"]):
        raise ValueError("Incomplete native-metrics recovery")
    for a, b in zip(rows, original["rows"], strict=True):
        for key in ("case_id", "prompt_tokens", "output_tokens", "raw_sse_sha256"):
            if a[key] != b[key]:
                raise ValueError("Native metrics do not match original request")
    arm = result["arms"][1]
    n, pairs = arm["necessary_target_positions"], arm["causal_pairs"]
    if any(r["prompt_tokens"] + r["output_tokens"] - 1 < 4 for r in rows):
        raise ValueError("This convolution correction requires at least four positions")
    c = catalog["target_scalar"]
    scalar = n * sum(
        c[k]
        for k in (
            "normalization_residual_gate",
            "rope_moe_router_attention_gate",
            "gdn_preprocessing",
            "causal_convolution_per_position",
            "attention_softmax_per_position",
        )
    )
    scalar += c["causal_convolution_initial_correction_per_request"] * len(rows)
    scalar += c["attention_softmax_per_pair"] * pairs
    scalar += c["gdn_recurrence_per_decode_row"] * (arm["output_tokens"] - len(rows))
    dep = draft_dependencies(rows)
    dc = catalog["draft"]
    draft = sum(
        dc[k] * dep[v]
        for k, v in (
            ("kv_row", "kv_rows"),
            ("full_query_row", "full_query_rows"),
            ("retained_vocab_row", "retained_vocab_rows"),
            ("attention_pair", "attention_pairs"),
        )
    )
    costs = [
        {int(k): v for k, v in catalog["gdn_prefill_chunks"][kind].items()}
        for kind in ("initial", "continuation")
    ]
    prefill = partition_bounds([r["prompt_tokens"] for r in rows], *costs)
    base = arm["matrix_flops_subtotal"] + scalar + draft
    total = [base + value for value in prefill]
    return dict(
        status="basic_work_interval_over_gdn_partitions_client_timing",
        run=arm["run"],
        recovered_native_requests=len(rows),
        dependencies=dep,
        target_scalar_flops=scalar,
        necessary_draft_flops=draft,
        gdn_prefill_flops_interval=prefill,
        basic_work_flops_interval=total,
        client_tflops_per_chip_interval=[
            f / (2 * arm["client_seconds"] * 1e12) for f in total
        ],
        timing_basis=result["timing_basis"],
        synchronized_effective_tflops_per_chip=None,
        reference_seconds=None,
        efficiency_percent=None,
        outline_pass=None,
        interval_meaning="Exact arithmetic extrema over all 1..64-token prefill partitions under the frozen 64-chunk GDN algorithm, structural-zero initial state and continued state thereafter; not a statistical confidence interval.",
        dependency_basis="Ordered native MTP2 acceptance, terminal output clipping, unique no-reuse request paths and qualified logical KV/hidden dependencies. Rejected roots and dead KV versions excluded.",
        remaining="Actual GDN partition ledger for a point F; matched synchronized uninstrumented T and independent full-answer reference for outline eta. B0 has no detailed native MTP sequence in its SSE records.",
    )


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
    for name in ("b1-native-acceptance.json.gz", "operation-catalog.json"):
        path = REPORT + "/" + name
        raw[path] = (root / path).read_bytes()
    catalog = json.loads(raw[REPORT + "/operation-catalog.json"])
    if catalog["model_config_sha256"] != audit["model_config_sha256"]:
        raise ValueError("Operation catalog model binding mismatch")
    result["b1_basic_work_interval"] = extended_b1(
        json.loads(raw[PAIR]),
        result,
        json.loads(gzip.decompress(raw[REPORT + "/b1-native-acceptance.json.gz"])),
        catalog,
    )
    result["missing_total_work"][0] = (
        "B0 detailed MTP sequence absent; B1 native sequence recovered and dependency work reconstructed"
    )
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "arms": result["arms"],
                "b1_basic_work_interval": result["b1_basic_work_interval"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
