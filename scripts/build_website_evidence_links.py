"""Build presentation-only evidence pointers without changing measured values."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from data_entrypoints import publication_bytes
from data_index import ROOT, render, tracked_paths

OUTPUT = "website-publication/data/benchmark-evidence-links.json"
FRONTIER = "website-publication/data/leaderboard_frontier.json.publication.json"
DIRECT = "archive/devhub-browser-access-20261022"
NAMES = ("summary-adjudicated.json", "summary-raw.json", "timeout-adjudication.json")


def extract_browser_copies(archive):
    """Small byte-identical access copies; original Release remains authoritative."""
    custody = json.loads((ROOT / "archive/dev-hub-20261020/manifest.json").read_text())
    raw = archive.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == custody["asset"]["sha256"]
    assert len(raw) == custody["asset"]["bytes"]
    destination = ROOT / DIRECT
    if destination.exists():
        raise ValueError("Refusing to replace an existing access copy")
    rows = []
    with tarfile.open(archive) as tf:
        for name in NAMES:
            member = f"results/szyn-swebench-qwen35-20261005/{name}"
            record = next(r for r in custody["files"] if r["path"] == member)
            payload = tf.extractfile(member).read()
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
            assert len(payload) == record["bytes"]
            destination.mkdir(exist_ok=True, parents=True)
            (destination / name).write_bytes(payload)
            rows.append(
                {
                    "path": name,
                    "member": member,
                    "sha256": record["sha256"],
                    "bytes": len(payload),
                }
            )
    (destination / "manifest.json").write_text(
        render(
            {
                "schema": "browser-access-copy/v1",
                "source_manifest": "archive/dev-hub-20261020/manifest.json",
                "asset": custody["asset"],
                "files": rows,
                "scope": "Byte-identical small access copies for direct website links, not new experiments or a replacement for the full original archive.",
            }
        )
    )


def identities(value):
    if isinstance(value, dict):
        yield from value.keys()
        for child in value.values():
            yield from identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from identities(child)
    elif isinstance(value, str):
        yield value


def build():
    paths = set(tracked_paths(ROOT))
    frontier = json.loads(publication_bytes(ROOT, FRONTIER, paths)[0])
    wanted = {
        rid
        for point in frontier["points"]
        for rid in point["evidence"].get("run_ids", [])
    }
    wanted.update(point["id"] for point in frontier["points"])
    extracts = {}
    for path in sorted(paths):
        if (
            not path.startswith("website-publication/data/")
            or "evidence" not in path
            or ".publication.part-" in path
            or path == OUTPUT
        ):
            continue
        if not path.endswith(".json"):
            continue
        payload = json.loads(publication_bytes(ROOT, path, paths)[0])
        for identity in set(identities(payload)) & wanted:
            extracts.setdefault(identity, []).append(path)
    raw_dir = "reports/frontier-raw-recovery-20261020/"
    raw_runs = {
        run["run_id"]: run
        for run in json.loads((ROOT / raw_dir / "manifest.json").read_text())["runs"]
    }
    points = {}
    for point in frontier["points"]:
        rids = point["evidence"].get("run_ids", [])
        links = []
        for rid in rids:
            if rid not in raw_runs:
                continue
            run = raw_runs[rid]
            archive_path = raw_dir + run["archive"]
            if archive_path in paths:
                links.append(
                    {
                        "path": archive_path,
                        "kind": "blob",
                        "role": "raw-archive",
                        "run_id": rid,
                    }
                )
            else:
                links.append(
                    {
                        "path": raw_dir + "raw",
                        "kind": "tree",
                        "role": "raw-parts",
                        "run_id": rid,
                    }
                )
                links.append(
                    {
                        "path": raw_dir + "manifest.json",
                        "kind": "blob",
                        "role": "raw-manifest",
                        "run_id": rid,
                    }
                )
        records = sorted(
            {p for identity in [point["id"], *rids] for p in extracts.get(identity, [])}
        )
        links += [
            {"path": p, "kind": "blob", "role": "public-measurement-extract"}
            for p in records
        ]
        if not links:
            links = [{"path": FRONTIER, "kind": "blob", "role": "publication-record"}]
            if (
                point["id"]
                in (
                    "qwen25-gsm8k-b128-target-only-20260913",
                    "qwen25-gsm8k-b128-vspec-draft-adaptive-20260913",  # pragma: allowlist secret
                )
            ):
                links = [
                    {
                        "path": "archive/vspec-browser-raw-20261022/snapshot/benchmark_results/regression_20260913/"  # pragma: allowlist secret
                        + rids[0],
                        "kind": "blob",
                        "role": "original-public-record",
                    }
                ]
        points[point["id"]] = {"run_ids": rids, "links": links}
    rewrites = {}
    for capsule in (
        "archive/website-links-20261022",
        "archive/website-links-accounting-20261022",
    ):
        receipt = json.loads((ROOT / capsule / "manifest.json").read_text())
        for row in receipt["files"]:
            if row["path"] == "LICENSE":
                continue
            rewrites[row["path"]] = {
                "path": capsule + "/snapshot/" + row["path"],
                "kind": "blob",
                "role": "original-public-record",
            }
    for name in NAMES:
        old = (
            "https://github.com/vLLM-HUST/vllm-hust-dev-hub/blob/542e6297f9aa93e81471fc40ed00acde729192bc/results/szyn-swebench-qwen35-20261005/"
            + name
        )
        rewrites[old] = {
            "path": DIRECT + "/" + name,
            "kind": "blob",
            "role": "verified-release-member-copy",
        }
    plans = {}
    for plan in json.loads(
        (ROOT / "website-publication/data/serving-plans.json").read_text()
    )["plans"]:
        plans[plan["id"]] = rewrites[plan["accountingEvidence"]]
    return {
        "schema": "benchmark-evidence-links/v1",
        "scope": "Navigation only. Original public records, raw archives and public extracts remain distinct. Publication-record fallback does not claim full request logs are public.",
        "rewrites": rewrites,
        "frontier_points": points,
        "serving_plans": plans,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-devhub", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.extract_devhub:
        extract_browser_copies(args.extract_devhub)
        return
    expected = render(build())
    if args.check:
        assert (ROOT / OUTPUT).read_text() == expected, (
            "Regenerate website evidence links"
        )
    else:
        (ROOT / OUTPUT).write_text(expected)


if __name__ == "__main__":
    main()
