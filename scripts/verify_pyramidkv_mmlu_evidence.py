"""Verify published PyramidKV MMLU-Pro evidence without a model or NPU.

This validates artifact identity and re-scores recorded outputs. It is not a
fresh inference run or a substitute for verifying the source dataset/model.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

from vllm_hust_benchmark import mmlu_pro_pyramidkv as adapter


def reassemble(root: Path, record: dict) -> bytes:
    data = b"".join(
        adapter.checked(root / part["path"], part["sha256"]) for part in record["parts"]
    )
    if len(data) != record["bytes"] or adapter.sha(data) != record["sha256"]:
        raise ValueError("Reassembled artifact identity mismatch")
    return data


def verify(root: Path) -> dict:
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Checksum path escapes evidence root")
        adapter.checked(path, digest)
    contract = json.loads((root / "plan/contract.json").read_text())
    adapter.checked(Path(adapter.__file__), contract["scorer_source_sha256"])
    adapter.checked(root / "plan/tasks.json", contract["files_sha256"]["tasks.json"])
    tasks = json.loads((root / "plan/tasks.json").read_text())
    inventory_bytes = reassemble(
        root / "plan", json.loads((root / "plan/inventory-parts.json").read_text())
    )
    if adapter.sha(inventory_bytes) != contract["files_sha256"]["inventory.jsonl.gz"]:
        raise ValueError("Inventory differs from pre-output contract")
    inventory = [
        json.loads(line) for line in gzip.decompress(inventory_bytes).splitlines()
    ]
    if len(inventory) != contract["full_test_count"]:
        raise ValueError("Full task inventory count differs")
    if {x["question_id"] for x in tasks} != adapter.choose_tasks(inventory, 5):
        raise ValueError("Selection differs from frozen first-five rule")
    if (
        sum(x["prompt_tokens"] > adapter.THRESHOLD for x in inventory)
        != contract["full_eligible_tasks"]
    ):
        raise ValueError("Compression eligibility audit differs")
    raw = {}
    for record in json.loads((root / "raw/archives.json").read_text()):
        data = reassemble(root / "raw", record)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or member.name in raw:
                    raise ValueError("Unexpected or duplicate archived member")
                raw[member.name] = archive.extractfile(member).read()
    comparison = json.loads((root / "comparison.json").read_text())
    if (
        adapter.sha((root / "plan/contract.json").read_bytes())
        != comparison["contract_sha256"]
    ):
        raise ValueError("Comparison uses a different contract")
    for name, digest in comparison["artifact_sha256"].items():
        if adapter.sha(raw[name]) != digest:
            raise ValueError(f"Raw artifact changed: {name}")
    scores = {}
    for arm in ("B0", "B1"):
        if json.loads(raw[f"{arm}/contract.json"]) != contract:
            raise ValueError("Arm contract differs")
        adapter.validate_activation(
            json.loads(raw[f"{arm}/launch.json"]), contract, arm
        )
        results = json.loads(raw[f"{arm}/results.json"])
        score = adapter.score_results(results, tasks)
        if (
            score != json.loads(raw[f"{arm}/score.json"])
            or score != comparison["arms"][arm]["quality"]
        ):
            raise ValueError("Published score differs from raw outputs")
        for result in results:
            case = result["case_id"]
            if json.loads(raw[f"{arm}/requests/{case}.json"]) != result:
                raise ValueError("Per-request record differs from aggregate")
            if not result["success"]:
                # Failed records stay in scoring; their partial/malformed streams
                # are preserved by the artifact hashes, not treated as completions.
                continue
            text = ""
            usage = None
            for line in raw[f"{arm}/requests/{case}.sse"].splitlines():
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                text += "".join(x.get("text", "") for x in chunk.get("choices", []))
            if text != result.get("output", "") or (
                result["success"] and usage != result["usage"]
            ):
                raise ValueError("Recorded output differs from raw SSE")
        scores[arm] = {
            k: score[k] for k in ("count", "correct", "accuracy", "transport_failures")
        }
    return {"verified": True, "inventory_tasks": len(inventory), "scores": scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
