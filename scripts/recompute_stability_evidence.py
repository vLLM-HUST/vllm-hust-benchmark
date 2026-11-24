"""Verify original child seals and recompute stability from the diagnostic Release pack."""

import argparse
import json
from pathlib import Path

from stability_assessment import assess, digest


def recompute(pack, summary):
    pack = pack.resolve()
    index = json.loads((pack / "STAGE_INDEX.json").read_bytes())
    if len(index) != 16:
        raise ValueError("Expected the original sixteen completed child stages")
    for stage in index:
        root = (pack / stage["path"]).resolve()
        if not root.is_relative_to(pack):
            raise ValueError("Stage path outside evidence pack")
        seal = json.loads((root / "checksums.json").read_bytes())
        actual = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlink in sealed child")
            if path.is_file() and path != root / "checksums.json":
                actual[str(path.relative_to(root))] = digest(path)
        if actual != seal:
            raise ValueError("Original child seal differs: " + stage["path"])
    roots = {
        **{
            "continuous-" + role: pack / "native/c8" / ("continuous-" + role)
            for role in ("B0", "B1")
        },
        "mmlu-pair": pack / "native/text/mmlu-pair",
        **{
            name + "-" + phase + "-" + role: pack
            / ("native/text" if name == "frontier" else "native/hle")
            / (name + "-" + phase + "-" + role)
            for name in ("frontier", "hle")
            for phase in ("solver", "judge")
            for role in ("B0", "B1")
        },
        **{
            name + "-trace-" + role: pack / "native/trace" / (name + "-trace-" + role)
            for name in ("swe", "terminal")
            for role in ("B0", "B1")
        },
    }
    evidence = assess(roots, summary["datasets"])
    for dataset in summary["datasets"]:
        for role, arm in dataset["arms"].items():
            if (
                evidence["profiles"][dataset["id"]][role]["source_seal_sha256"]
                != arm["source_stage_seal_sha256"]
            ):
                raise ValueError(
                    "Public dataset projection has a different source seal"
                )
    for role in ("B0", "B1"):
        if (
            evidence["continuous"][role]["source_seal_sha256"]
            != summary["continuous"][role]["source_seal_sha256"]
        ):
            raise ValueError("Public C8 projection has a different source seal")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = recompute(args.evidence_pack, json.loads(args.summary.read_bytes()))
    if evidence != json.loads(args.expected.read_bytes()):
        raise ValueError(
            "Recomputed stability evidence differs from the published audit"
        )
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print("PASS: 16 original child seals; stability evidence exactly reproduced")


if __name__ == "__main__":
    main()
