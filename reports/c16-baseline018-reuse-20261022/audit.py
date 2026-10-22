"""Offline request/weight identity audit; does not run inference or award acceptance."""

import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_archive(blob):
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        return {
            name: archive.extractfile(name).read()
            for name in ("config.json", "summary.json", "requests.jsonl")
        }


def audit(root=ROOT):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    b0 = read_archive(
        b"".join((root / p).read_bytes() for p in manifest["native_parts"])
    )
    recovered = root.parent / "frontier-raw-recovery-20261020"
    runs = json.loads((recovered / "manifest.json").read_text())["runs"]
    run = next(r for r in runs if r["run_id"] == "fa0e01a32eef456bab4bebe51e4aa7dd")
    b1_blob = b"".join(
        (recovered / p["path"]).read_bytes() for p in run["archive_parts"]
    )
    assert hashlib.sha256(b1_blob).hexdigest() == run["archive_sha256"]
    b1 = read_archive(b1_blob)
    configs = [json.loads(x["config.json"]) for x in (b0, b1)]
    records = [
        [json.loads(line) for line in x["requests.jsonl"].splitlines()]
        for x in (b0, b1)
    ]
    for k in (
        "workload_sha256",
        "concurrency",
        "duration",
        "chips",
        "window_policy",
        "cache_policy",
    ):
        assert configs[0][k] == configs[1][k], k
    assert (
        configs[0]["tokenizer"]["fingerprint"] == configs[1]["tokenizer"]["fingerprint"]
    )

    def key(row):
        return row["lane"], row["play"], row["turn"]

    mapped = {key(r): r for r in records[1]}
    assert len(mapped) == len(records[1])
    assert len({key(r) for r in records[0]}) == len(records[0])
    window_tokens = []
    for archive, rows, config in zip((b0, b1), records, configs):
        summary = json.loads(archive["summary.json"])
        tokens = sum(
            count
            for row in rows
            for elapsed, count in row["chunks"]
            if 0 <= elapsed < config["duration"]
        )
        assert summary["valid"] and summary["failed_requests"] == 0
        assert tokens == summary["observed_output_tokens_in_window"]
        assert tokens / config["duration"] == summary["output_tokens_per_second"]
        window_tokens.append(tokens)
    differences = dict.fromkeys(
        ("trajectory_id", "prompt_sha256", "prompt_tokens", "expected_output_tokens"), 0
    )
    first_turn_hash_differences = 0
    for row in records[0]:
        other = mapped[key(row)]
        for field in differences:
            differences[field] += row[field] != other[field]
        if row["turn"] == 0:
            first_turn_hash_differences += (
                row["prompt_sha256"] != other["prompt_sha256"]
            )
    hf = json.loads((root / "weights-huggingface.json").read_text())
    ms = json.loads((root / "weights-modelscope.json").read_text())
    assert len(hf["weights"]) == len(ms["weights"]) == 14
    assert hf["weights"] == ms["weights"]
    return {
        "b0_requests": len(records[0]),
        "b1_requests": len(records[1]),
        "matched_b0_requests": len(records[0]),
        "field_differences": differences,
        "first_turn_hash_differences": first_turn_hash_differences,
        "identical_upstream_weight_shards": len(hf["weights"]),
        "window_tokens": window_tokens,
        "descriptive_ratio": window_tokens[1] / window_tokens[0],
        "descriptive_gain_percent": (window_tokens[1] / window_tokens[0] - 1) * 100,
        "scope": "Input identity audit, not quality acceptance or isolated MOD attribution",
    }


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
