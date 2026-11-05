"""Build the public, bounded MOD test report from the pinned GenNova evidence import."""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/gennova-mod-evidence-20261105"
SNAPSHOTS = REPORT / "snapshots"
OUTPUT = ROOT / "website-publication/data/mod-test-evidence.json"
BENCHMARK_REPORT_COMMIT = (
    "d04a5ccbbc1b1f3d2b25e363b289b2f3d13a99a4"  # pragma: allowlist secret
)
SOURCE_COMMIT = "c8416aad2c9e161abe2dde5e01690edc48e756e3"  # pragma: allowlist secret


def load(name):
    return json.loads((SNAPSHOTS / name).read_text(encoding="utf-8"))


def compact_integration(entry):
    integration = entry.get("integration") or {}
    return {
        "id": entry["id"],
        "name": entry.get("name") or entry["id"],
        "status": integration.get("status"),
        "default_enabled": integration.get("default_enabled", False),
        "priority": integration.get("priority"),
        "positive": entry.get("positive", []),
        "negative": entry.get("negative", []),
        "boundary": entry.get("boundary", ""),
        "evidence_count": len(entry.get("evidence", [])),
    }


def compact_combination(entry):
    return {
        key: entry.get(key)
        for key in (
            "id",
            "combination",
            "baseline",
            "effect",
            "quality",
            "qualification",
        )
    } | {"evidence_count": len(entry.get("evidence", []))}


def compact_negative(entry):
    return {
        key: entry.get(key)
        for key in (
            "id",
            "name",
            "status",
            "scope",
            "result",
            "do_not_repeat",
            "reopen_requires",
            "outcome_kind",
            "quality_scope",
        )
    }


def build():
    catalog = load("results-catalog.json")
    integrations = load("gennova-positive-integrations.json")
    combinations = load("combination-plan.json")
    negatives = load("negative-results.json")
    categories = Counter(entry["category"] for entry in catalog["entries"])
    tasks = Counter(task for entry in catalog["entries"] for task in entry["tasks"])

    return {
        "schema_version": "mod-test-evidence/v1",
        "updated_date": "2026-11-05",
        "title_en": "MOD test evidence report",
        "title_zh": "MOD 测试证据报告",
        "source": {
            "benchmark_repository": "vLLM-HUST/vllm-hust-benchmark",
            "benchmark_report_commit": BENCHMARK_REPORT_COMMIT,
            "report_url": (
                "https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/"
                f"{BENCHMARK_REPORT_COMMIT}/reports/gennova-mod-evidence-20261105"
            ),
            "original_repository": "GenNova-HUST/inference-engine-project",
            "original_commit": SOURCE_COMMIT,
        },
        "summary": {
            "catalog_entries": len(catalog["entries"]),
            "integration_records": len(integrations["entries"]),
            "combination_effects": len(integrations["combination_effects"]),
            "bounded_negative_results": len(negatives["entries"]),
            "measured_combinations": len(combinations["measured_combinations"]),
            "pending_register": len(combinations["pending_register"]),
            "category_counts": dict(sorted(categories.items())),
            "task_counts": dict(sorted(tasks.items())),
        },
        "scope": {
            "new_measurements": False,
            "official_leaderboard_submission": False,
            "raw_logs_published": False,
            "website_report": True,
            "note_en": (
                "Historical single-MOD, combination, and negative-result evidence. "
                "Component gains, bounded pairs, and small samples are not stable whole-model claims."
            ),
            "note_zh": (
                "历史单项 MOD、组合与负结果证据。组件加速、有限配对和小样本不能外推为稳定整机收益。"
            ),
        },
        "integration_records": [
            compact_integration(entry) for entry in integrations["entries"]
        ],
        "combination_effects": [
            compact_combination(entry) for entry in integrations["combination_effects"]
        ],
        "negative_results": [compact_negative(entry) for entry in negatives["entries"]],
    }


def render():
    return json.dumps(build(), ensure_ascii=False, indent=2) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = render()
    if args.write:
        OUTPUT.write_text(expected, encoding="utf-8")
        print(f"Updated {OUTPUT.relative_to(ROOT)}")
        return 0
    if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
        print(f"Stale publication: {OUTPUT.relative_to(ROOT)}")
        return 1
    print("MOD website publication is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
