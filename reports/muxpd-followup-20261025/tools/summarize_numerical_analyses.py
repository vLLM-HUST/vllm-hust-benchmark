"""Summarize frozen numerical analyses without recomputing or relaxing scopes."""

import argparse
import hashlib
import json
from pathlib import Path


def file_identity(path):
    path = Path(path).resolve()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def verify_identity(expected):
    actual = file_identity(expected["path"])
    if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError(f"Source changed: {expected['path']}")


def summarize_pairs(pairs, left_hashes=None, right_hashes=None):
    decisions = [d for pair in pairs for d in pair["decisions"]]
    numeric = [d for d in decisions if d["logits_compared"]]
    if any(
        not d["same_request_history"] or not d["same_semantic_batch"] for d in numeric
    ):
        raise ValueError("Analysis compared logits after semantic context diverged")
    return {
        "request_pairs": len(pairs),
        "exact_token_sequence_pairs": sum(
            p["exact_sequence_match"] is True for p in pairs
        ),
        "paired_token_positions": len(decisions),
        "shared_request_prefix_positions": sum(
            d["same_request_history"] is True for d in decisions
        ),
        "shared_request_and_peer_context_positions": sum(
            d["same_request_history"] and d["same_semantic_batch"] for d in decisions
        ),
        "compared_logit_vectors": len(numeric),
        "strict_batch_descriptor_positions": sum(
            d["strict_context_match"] is True for d in numeric
        ),
        "exact_logit_value_equal_vectors": sum(
            d["logits_byte_equal"] is True for d in numeric
        ),
        "byte_identical_npy_files_on_compared_positions": sum(
            left_hashes.get((p["left_phase"], p["lane"], d["output_token_index"]))
            == right_hashes.get((p["right_phase"], p["lane"], d["output_token_index"]))
            and left_hashes.get((p["left_phase"], p["lane"], d["output_token_index"]))
            is not None
            for p in pairs
            for d in p["decisions"]
            if d["logits_compared"]
        )
        if left_hashes is not None and right_hashes is not None
        else None,
        "max_abs_delta_on_compared_positions": max(
            (d["max_abs_delta"] for d in numeric if d["max_abs_delta"] is not None),
            default=None,
        ),
        "all_compared_native_argmax_agree_with_cpu": (
            all(
                d["left_native_argmax_agrees"] and d["right_native_argmax_agrees"]
                for d in numeric
            )
            if numeric
            else None
        ),
        "first_divergences": [
            {
                "left_phase": p["left_phase"],
                "right_phase": p["right_phase"],
                "lane": p["lane"],
                "decision": p["first_observed_token_difference"],
            }
            for p in pairs
            if p["first_observed_token_difference"] is not None
        ],
    }


def logit_hash_records(sources):
    runs = [entry for entry in sources if Path(entry["path"]).name == "run.json"]
    if len(runs) != 1:
        raise ValueError("Expected exactly one run receipt")
    report = json.loads(Path(runs[0]["path"]).read_text())
    positions = {
        rid: (phase, lane)
        for phase, row in enumerate(report["phases"])
        for lane, rid in enumerate(row["request_ids"])
    }
    source_hashes = {
        str(Path(entry["path"]).resolve()): entry["sha256"] for entry in sources
    }
    records = {}
    for entry in sources:
        path = Path(entry["path"])
        if not path.name.startswith("rank-0-") or path.suffix != ".jsonl":
            continue
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "native-sample-observed":
                continue
            for row in event["rows"]:
                receipt = row.get("logits")
                if receipt is None:
                    continue
                artifact = str((path.parent / receipt["file"]).resolve())
                if source_hashes.get(artifact) != receipt["sha256"]:
                    raise ValueError(
                        "Vector hash is not bound by verified analysis sources"
                    )
                key = (*positions[row["request_id"]], row["output_token_index"])
                if key in records:
                    raise ValueError("Duplicate sample identity")
                records[key] = receipt["sha256"]
    return records


def summarize(path, verified):
    value = json.loads(path.read_text())
    if value["schema"] != "muxpd-fixed-numerics-comparison-v1":
        raise ValueError(f"Not a numerical analysis: {path}")
    for entry in [
        value["producer"],
        *value["left_sources"],
        *value.get("right_sources", []),
    ]:
        identity = (entry["path"], entry["bytes"], entry["sha256"])
        if identity not in verified:
            verify_identity(entry)
            verified.add(identity)
    result = {"analysis": file_identity(path), "scope": value["scope"]}
    for name in (
        "left_collection_passed",
        "right_collection_passed",
        "left_consistency",
        "right_consistency",
        "same_driver_source",
        "engine_argument_differences",
    ):
        if name in value:
            result[name] = value[name]
    hashes = {
        side: logit_hash_records(value[side + "_sources"])
        for side in ("left", "right")
        if side + "_sources" in value
    }
    for name in ("left_warm_pairs", "right_warm_pairs", "cross_pairs"):
        if name in value:
            left = "right" if name == "right_warm_pairs" else "left"
            right = "left" if name == "left_warm_pairs" else "right"
            result[name] = summarize_pairs(value[name], hashes[left], hashes[right])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    verified = set()
    rows = [summarize(path.resolve(), verified) for path in args.analysis]
    value = {
        "schema": "muxpd-numerical-summary-v1",
        "scope": "Derived counts only; source bindings reverified. No acceptance tolerance, quality, throughput or cause inferred.",
        "producer": file_identity(__file__),
        "verified_source_identities": len(verified),
        "analyses": rows,
    }
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        "Analysis | comparison | exact sequences | comparable vectors | byte equal | max delta"
    )
    for row in rows:
        for key in ("left_warm_pairs", "right_warm_pairs", "cross_pairs"):
            if key not in row:
                continue
            p = row[key]
            print(
                f"{Path(row['analysis']['path']).name} | {key} | "
                f"{p['exact_token_sequence_pairs']}/{p['request_pairs']} | "
                f"{p['compared_logit_vectors']}/{p['paired_token_positions']} | "
                f"{p['byte_identical_npy_files_on_compared_positions']} | {p['max_abs_delta_on_compared_positions']}"
            )


if __name__ == "__main__":
    main()
