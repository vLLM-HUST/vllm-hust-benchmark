"""Package measured projection references with pinned raw release locators."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

PINS = {
    "dense": "8836c00685b2001b462da0637fe979acc6b53870dcd4f62da974ed2145aeca47",  # pragma: allowlist secret
    "moe": "2e4a93d7f33e81df43c8e8d67b501242418df60c19b8cce556f192393ba9dd77",  # pragma: allowlist secret
    "vocab": "f4761f8c6801d2fd3a47972c3e812264d99a75a2295820e3682b6a0f958b1a1f",  # pragma: allowlist secret
}
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-directory", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "m01-recovery/projection-references-20261021"
    destination.mkdir(exist_ok=False)
    selected, sources, summaries = {}, {}, {}
    for family, expected in PINS.items():
        name = f"m01-{family}-reference-recovery-r1-evidence.tar.gz"
        path = args.raw_directory / name
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == expected
        members = {}
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isdir():
                    continue
                assert member.isfile()
                raw = archive.extractfile(member).read()
                members[member.name] = {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
                # Large per-call samples remain in the immutable raw release.
                if (
                    member.name.endswith(".jsonl")
                    or member.name == "recovery-moe-reference-plan-r1.json"
                ):
                    continue
                selected[f"{family}/{member.name}"] = raw
                if member.name == f"{family}-recovery-reference-summary-r1.json":
                    summaries[family] = json.loads(raw)
        sources[family] = {
            "url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
            "m01-recovery-20261021-r7/" + name,
            "sha256": expected,
            "bytes": path.stat().st_size,
            "members": members,
        }
    assert set(summaries) == set(PINS)
    assert (
        len(
            {json.dumps(s["request_phase"], sort_keys=True) for s in summaries.values()}
        )
        == 1
    )
    for family, summary in summaries.items():
        assert summary["all_four_idle_after"]
        assert (
            summary["model_effective_flops"] is None
            and summary["m01_attainment"] is None
        )
        receipt_name = f"{family}/m01-{family}-reference-recovery-r1/receipt.json"
        receipt = json.loads(selected[receipt_name])
        assert receipt["probe_exit_code"] == receipt["cleanup_exit_code"] == 0
        assert (
            receipt["all_four_idle_after"] and receipt["prior_cache_imported"] is False
        )
    selected["summaries.json"] = (json.dumps(summaries, indent=2) + "\n").encode()
    packed = destination / "evidence.tar.xz"
    with tarfile.open(packed, "w:xz") as archive:
        for name, raw in sorted(selected.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            archive.addfile(member, io.BytesIO(raw))
    index = {
        "schema": "m01-projection-references/v1",
        "archive": packed.name,
        "archive_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
        "members": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in sorted(selected.items())
        },
        "raw_archives": sources,
        "model_effective_flops": None,
        "m01_attainment": None,
    }
    (destination / "archive.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps({"portable_bytes": packed.stat().st_size, "members": len(selected)})
    )


if __name__ == "__main__":
    main()
