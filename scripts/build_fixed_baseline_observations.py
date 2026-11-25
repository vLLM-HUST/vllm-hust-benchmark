"""Publish existing fixed-B0 observations independently of candidate releases."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "website-publication/data/fixed-b0-original018-v1.json"
CATALOG = "website-publication/data/engine-release-history-v1.json"
SOURCES = {
    "profiles": "reports/r8-fixed-five-20261118/summary-public-v2.json",
    "context": "reports/r8-context-tenant-20261119/summary-public-v1.json",
    "compute": "website-publication/data/r8-compute-reference-20261119-v2/summary-public-v1.json",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def render(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build(root=ROOT):
    catalog = json.loads((root / CATALOG).read_bytes())
    baseline = {k: v for k, v in catalog["baseline"].items() if k != "observations"}
    loaded = {k: json.loads((root / p).read_bytes()) for k, p in SOURCES.items()}
    profiles = loaded["profiles"]
    assert profiles["selected_execution_complete"] is True
    rows = []
    for row, identifier, count in zip(
        profiles["measurements"],
        ("mmlu", "hle", "frontier", "swe", "terminal"),
        (200, 200, 160, 64, 48),
        strict=True,
    ):
        assert row["id"] == identifier and row["baseline_id"] == baseline["id"]
        arm = row["arms"][0]
        assert row["control"] == arm["role"] == "B0"
        assert arm["requests"] == count and arm["failed"] == 0
        rows.append(dict(id=identifier, label=row["label"], arm=arm))
    context, compute = loaded["context"], loaded["compute"]
    assert context["baseline_id"] == compute["baseline_id"] == baseline["id"]
    return dict(
        schema="vllm-hust-fixed-baseline-observations/v1",
        baseline=baseline,
        model=context["model"],
        precision=context["precision"],
        tensor_parallel_size=context["tensor_parallel_size"],
        chip_model=context["chip_model"],
        scope="Existing dated B0 observations; publishing a candidate never creates a new B0 measurement or an automatic pair.",
        sources={
            name: dict(
                repository_path=path,
                sha256=sha((root / path).read_bytes()),
                bytes=(root / path).stat().st_size,
            )
            for name, path in SOURCES.items()
        },
        source_report_completed_utc=profiles["campaign_completed_utc"],
        profiles=rows,
        context=dict(
            report_date=context["report_date"],
            protocol=context["protocol"],
            seal_sha256=context["arm_seals"]["B0"],
            raw_asset=context["raw_asset"],
            workload_label_zh=context["workload_label_zh"],
            workload_label_en=context["workload_label_en"],
            observations=[
                dict(phase=row["phase"], B0=row["B0"]) for row in context["comparisons"]
            ],
        ),
        compute=dict(
            model=compute["model"],
            protocol=compute["protocol"],
            accepted_scope=compute["accepted_scope"],
            arm=compute["arms"]["B0"],
            certificate=compute["certificate"],
            reference_statistics=compute["reference_statistics"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = render(build())
    catalog = json.loads((ROOT / CATALOG).read_bytes())
    descriptor = dict(
        schema="vllm-hust-fixed-baseline-descriptor/v1",
        path=OUTPUT.removeprefix("website-publication/"),
        sha256=sha(raw),
        bytes=len(raw),
    )
    if args.check:
        assert (ROOT / OUTPUT).read_bytes() == raw, "Fixed B0 originals differ"
        assert catalog["baseline"]["observations"] == descriptor
    else:
        (ROOT / OUTPUT).write_bytes(raw)
        catalog["baseline"]["observations"] = descriptor
        (ROOT / CATALOG).write_bytes(render(catalog))
    print(
        "Fixed B0: five profiles, four context phases and original compute statistics verified"
    )


if __name__ == "__main__":
    main()
