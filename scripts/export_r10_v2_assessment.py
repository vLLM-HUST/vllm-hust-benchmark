"""Publish the two independently replayed r10 V2 assessments."""

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "website-publication/data/engine-release-history-v1.json"
PATH = "data/r10-v2-assessment-20261122.json"
KINDS = ("native-rms", "contiguous-restore")
GOALS = ("4.1", "4.2", "4.3", "4.4")
PROFILES = ("mmlu", "hle", "frontier", "swe", "terminal")
BASELINE = "original018-cann8.5.1"
ENGINE_URL = "https://github.com/DeepLinkNext/DeepLinkEngine/blob/"


def pin(raw):
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def read_candidate(report_root, kind, engine_root, revision, entry):
    version = f"r10-{kind}"
    directory = report_root / version
    raw = (directory / "summary.json").read_bytes()
    receipt_raw = (directory / "receipt.json").read_bytes()
    summary, receipt = json.loads(raw), json.loads(receipt_raw)
    selection = summary["selection"]
    source = next(
        node
        for node in entry["sources"]
        if f"/r10-{kind}-fixed-five-" in node["url"]
        and node["url"].endswith("/summary.json")
    )
    if (
        receipt["status"] != "completed"
        or receipt["mode"] != "cpu-replay"
        or receipt["model_requests"] != 0
        or receipt["npu_tasks_started"] != 0
        or receipt["selection"] != selection
        or selection["version"] != version
        or selection["status"] != "candidate"
        or selection["baseline_id"] != BASELINE
        or selection["model"] != "Qwen3.5-35B-A3B"
        or selection["accelerator"] != "Ascend 910B2"
        or selection["accelerator_count"] != 2
        or receipt["outputs"]["summary.json"] != pin(raw)
        or receipt["source"] != summary["source"]
        or any(receipt["source"][key] != source[key] for key in ("sha256", "bytes"))
        or receipt["assessment_sources"] != summary["assessment_sources"]
    ):
        raise ValueError(f"Unverified CPU replay or fixed-five source for {version}")
    targets = summary["submission_v2"]
    if set(targets) != set(GOALS) or any(
        type(targets[goal][phase]["met"]) is not bool
        for goal in GOALS
        for phase in ("milestone", "final")
    ):
        raise ValueError(f"Missing V2 verdict for {version}")
    sources = {}
    for key, path in (
        ("compute_audit", "evidence/r10-v2-compute-supplement.json"),
        ("original_evidence_audit", "evidence/r10-original-evidence-audit-v1.json"),
        ("context_audit", "evidence/r10-v2-context-supplement.json"),
    ):
        identity = pin((engine_root / path).read_bytes())
        if any(
            receipt["assessment_sources"][key][field] != value
            for field, value in identity.items()
        ):
            raise ValueError(f"Unverified {key} source for {version}")
        sources[key] = {"url": f"{ENGINE_URL}{revision}/{path}", **identity}
    rent = targets["4.4"]["published_rental"]
    rental_hash = pin((engine_root / rent["evidence_path"]).read_bytes())["sha256"]
    if (
        rental_hash != rent["evidence_sha256"]
        or rental_hash != receipt["published_rental_sha256"]
    ):
        raise ValueError(f"Unverified rental source for {version}")
    return {
        "selection": selection,
        "fixed_five": source,
        "cpu_replay": {"summary": pin(raw), "receipt": pin(receipt_raw)},
        "sources": sources,
        "datasets": {key: summary["workloads"][key]["dataset"] for key in PROFILES},
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--engine-revision", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-f0-9]{40}", args.engine_revision):
        parser.error("--engine-revision must be a full Git SHA")
    catalog = json.loads(CATALOG.read_bytes())
    entry = next(row for row in catalog["entries"] if row["id"] == "r10")
    result = {
        "schema": "vllm-hust-candidate-v2-assessment/v1",
        "release_family": entry["id"],
        "baseline_id": catalog["baseline"]["id"],
        "engine_revision": args.engine_revision,
        "candidates": {
            kind: read_candidate(
                args.report_root, kind, args.engine_root, args.engine_revision, entry
            )
            for kind in KINDS
        },
    }
    raw = json_text(result).encode()
    (ROOT / "website-publication" / PATH).write_bytes(raw)
    entry["candidate_submission_assessment"] = {"path": PATH, **pin(raw)}
    CATALOG.write_text(json_text(catalog), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
