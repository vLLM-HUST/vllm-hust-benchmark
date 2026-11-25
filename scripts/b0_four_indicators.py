import json
import math
import sys
from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1] / "website-publication"
OUTPUT = "data/b0-four-indicators-20261125-v1/report.json"
PROOF = "data/b0-four-indicators-20261125-v1/0.28-prefix-repair-4.1.json"
FIXED = "data/fixed-b0-original018-v1.json"
EDITED = "data/b0-mtp-kv-prefix-repair-v1/report.json"
RENTAL = "data/b0-independent-rental-observations-v1.json"
TARGETS = "data/r10-35b-paired-full-20261125/acceptance.json"
SUITE = ("mmlu", "hle", "frontier", "swe", "terminal")
CONTEXT = ("4096-formal", "16384-formal", "32768-formal", "multi-tenant-formal")
IDS = (
    "original018-cann8.5.1",
    "official-0.23.0",
    "candidate-0.28.0rc1-source",
    "official-0.30.0",
)


def publication(root, path):
    raw = (root / path).read_bytes()
    return json.loads(raw), {
        "path": path,
        "sha256": sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def performance(arm):
    keys = (
        "requests",
        "failed",
        "output_tokens",
        "duration_seconds",
        "output_tokens_per_second",
    )
    row = {name: arm[name] for name in keys}
    assert 0 <= row["failed"] < row["requests"] and row["output_tokens"] > 0
    assert math.isclose(
        row["output_tokens"] / row["duration_seconds"],
        row["output_tokens_per_second"],
        rel_tol=1e-9,
    )
    row["successful"] = row["requests"] - row["failed"]
    row["ttft_mean_ms"] = arm.get("mean_ttft_ms", arm.get("ttft_ms", {}).get("mean"))
    row["tpot_mean_ms"] = arm.get("mean_tpot_ms", arm.get("tpot_ms", {}).get("mean"))
    assert all(
        isinstance(row[field], (float, int)) and row[field] >= 0
        for field in ("ttft_mean_ms", "tpot_mean_ms")
    )
    return row


def context_row(phase, arm, seal):
    row = performance(arm)
    row.update(
        phase=phase,
        ttft_p95_ms=arm["ttft_ms"]["p95"],
        tpot_p95_ms=arm["tpot_ms"]["p95"],
        source_seal_sha256=seal,
    )
    return row


def workload_cost(row, rate):
    assert row["failed"] == 0
    return (
        1e6 * rate * 2 * row["duration_seconds"] / (30 * 86400 * row["output_tokens"])
    )


def compute(raw, source, identity, reference):
    fixed = raw.get(
        "efficiency_lower_percent", raw.get("reference_normalized_lower_percent")
    )
    rates = raw.get(
        "achieved_tflops_per_card_interval", raw.get("tflops_per_card_interval")
    )
    flops = [int(n) for n in raw["F_interval"]]
    assert len(flops) == 2 and 0 < flops[0] <= flops[1] and len(raw["repetitions"]) == 3
    with localcontext() as ctx:
        ctx.prec = 40
        denominator = Decimal(reference)
        seconds = Decimal(raw["T_seconds"])
        recomputed = [Decimal(value) / (2 * seconds * 10**12) for value in flops]
        for left, right in zip(recomputed, rates, strict=True):
            assert abs(left - Decimal(right)) < Decimal("1e-32")
        assert abs(100 * recomputed[0] / denominator - Decimal(fixed)) < Decimal(
            "1e-32"
        )
    return {
        "identity": identity,
        "workload": "MMLU-Pro fixed 200, native out256",
        "formal_repetitions": 3,
        "F_interval": flops,
        "T_seconds": str(seconds),
        "tflops_per_card_interval": rates,
        "reference_tflops_per_card_upper": reference,
        "efficiency_lower_percent": fixed,
        "source": source,
    }


def build(root):
    original, fixed_source = publication(root, FIXED)
    repairs, edited_source = publication(root, EDITED)
    repaired28, source28 = publication(root, PROOF)
    rental, rental_source = publication(root, RENTAL)
    targets, targets_source = publication(root, TARGETS)
    catalog, _ = publication(root, "data/engine-release-history-v1.json")
    quote = targets["targets"]["4.4"]["rental_quote"]
    assert quote == rental["rental_quote"]
    reference = repairs["original_reference"]["C_upper_tflops_per_card"]
    assert (
        reference
        == repaired28["reference_tflops_per_card_upper"]
        == original["compute"]["arm"]["reference_tflops_per_card_upper"]
    )
    assert (
        repaired28["source_and_timing_qualified"]
        and repaired28["upstream_source_patched"]
    )
    compute_arms = (
        (original["compute"]["arm"], fixed_source, "original-0.18"),
        (
            repairs["versions"]["0.23.0"],
            edited_source,
            "0.23.0-selected-kv-prefix-repair",
        ),
        (repaired28, source28, "0.28.0rc1-source-selected-kv-prefix-repair"),
        (
            repairs["versions"]["0.30.0"],
            edited_source,
            "0.30.0-selected-kv-prefix-repair",
        ),
    )
    entries = {entry["id"]: entry for entry in catalog["comparison_baselines"]}
    rows = {}
    for baseline, (ft, ft_source, ft_id) in zip(IDS, compute_arms, strict=True):
        if baseline == IDS[0]:
            source = fixed_source
            profiles = {row["id"]: row for row in original["profiles"]}
            windows = {
                row["phase"]: row["B0"] for row in original["context"]["observations"]
            }
            phases = [
                context_row(
                    name, windows[name]["metrics"], original["context"]["seal_sha256"]
                )
                for name in CONTEXT
            ]
            utilization = windows[CONTEXT[-1]]["utilization"]
        else:
            entry = entries[baseline]
            obs, source = publication(root, entry["observations"]["path"])
            assert source == entry["observations"] and obs["baseline_id"] == baseline
            assert (
                obs["model"] == original["model"] and obs["tensor_parallel_size"] == 2
            )
            profiles = {row["id"]: row for row in obs["profiles"]}
            windows = {name: profiles["real-context-" + name] for name in CONTEXT}
            phases = [
                context_row(
                    name,
                    windows[name]["arm"],
                    windows[name]["arm"]["source_seal_sha256"],
                )
                for name in CONTEXT
            ]
            utilization = windows[CONTEXT[-1]]["utilization"]
        suite = [
            dict(
                id=name,
                source_seal_sha256=profiles[name]["arm"]["source_seal_sha256"],
                **performance(profiles[name]["arm"]),
            )
            for name in SUITE
        ]
        costs = {
            name: workload_cost(row, float(quote["cny_per_card_month"]))
            for name, row in (("mmlu", suite[0]), ("c24", phases[-1]))
        }
        if baseline != IDS[0]:
            verified = rental["baselines"][baseline]
            assert verified["observations"] == source
            for name, key in (
                ("mmlu", "mmlu"),
                ("c24", "real-context-multi-tenant-formal"),
            ):
                assert math.isclose(
                    costs[name],
                    verified["workloads"][key][
                        "estimated_cny_per_million_successful_output_tokens"
                    ],
                )
            assert math.isclose(
                utilization["mean_percent"],
                verified["workloads"]["real-context-multi-tenant-formal"][
                    "utilization"
                ]["mean_percent"],
            )
        c24 = {
            "mean_ai_core_percent": utilization["mean_percent"],
            "per_card_percent": utilization["per_card_percent"],
            "covered_seconds": utilization["covered_seconds"],
            "missing_seconds": utilization["missing_seconds"],
        }
        assert len(c24["per_card_percent"]) == 2 and c24["missing_seconds"] == 0
        assert math.isclose(
            c24["covered_seconds"], phases[-1]["duration_seconds"], rel_tol=1e-9
        )
        rows[baseline] = {
            "4.1": compute(ft, ft_source, ft_id, reference),
            "4.2": {"source": source, "profiles": suite},
            "4.3": {
                "source": source,
                "phases": phases,
                "ttft_p95_16k_over_4k": phases[1]["ttft_p95_ms"]
                / phases[0]["ttft_p95_ms"],
                "tpot_p95_16k_over_4k": phases[1]["tpot_p95_ms"]
                / phases[0]["tpot_p95_ms"],
            },
            "4.4": {
                "source": source,
                "rental_projection": rental_source if baseline != IDS[0] else None,
                "estimated_cny_per_million_successful_output_tokens": costs,
                "c24_device_utilization": c24,
            },
        }
    return {
        "schema": "vllm-hust-b0-four-indicators/v1",
        "model": original["model"],
        "tensor_parallel_size": 2,
        "precision": "BF16",
        "accelerator": "Ascend 910B2",
        "scope": "standalone-b0",
        "official_v2_baseline_id": IDS[0],
        "rental_quote": quote,
        "rental_quote_source": targets_source,
        "baselines": rows,
    }


def encode(data):
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


if __name__ == "__main__":
    output = PUBLIC / OUTPUT
    actual = encode(build(PUBLIC))
    if "--check" in sys.argv:
        assert output.read_text(encoding="utf-8") == actual
    else:
        output.write_text(actual, encoding="utf-8", newline="\n")
