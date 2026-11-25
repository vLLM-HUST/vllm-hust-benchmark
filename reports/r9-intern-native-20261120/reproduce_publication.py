"""Reproduce public model-integration numbers from the retained original summary."""

from pathlib import Path
from hashlib import sha256
import argparse
import gzip
import json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parents[2]
    / "website-publication/data/r9-intern-native-20261120/summary.json",
)
p.add_argument("--check", action="store_true")
args = p.parse_args()
original = Path(__file__).with_name("native-source-summary.json.gz").read_bytes()
source = gzip.decompress(original)
assert (
    sha256(source).hexdigest()
    == "0a394374301810d19fad4c9740c55e394baa3712d650963db503eb5824d60774"  # pragma: allowlist secret -- native summary digest
)
raw = json.loads(source)
rows = []
reference = raw["published_r8_reference"]
for key, kind in [
    ("MMLU200", "mmlu200"),
    ("4096-formal", "continuous-4k-c8"),
    ("16384-formal", "continuous-16k-c8"),
]:
    old = reference["mmlu200"] if key == "MMLU200" else reference["continuous_c8"][key]
    new = raw["current_mmlu200"] if key == "MMLU200" else raw["current_continuous"][key]
    changes = {
        x["metric"]: x["change_percent"]
        for x in raw["comparisons"]
        if x["workload"] == key
    }
    rows.append(
        dict(
            id=kind,
            label_zh={
                "MMLU200": "MMLU200",
                "4096-formal": "4K / C8",
                "16384-formal": "16K / C8",
            }[key],
            label_en={
                "MMLU200": "MMLU200",
                "4096-formal": "4K / C8",
                "16384-formal": "16K / C8",
            }[key],
            reference=old,
            candidate=new,
            change_percent=changes,
        )
    )
summary = dict(
    schema="vllm-hust-model-integration/v1",
    candidate=dict(
        release="full-answer-b1-20261119-r9",
        model="Intern-S2",
        dtype="BF16",
        hardware="Ascend910B2",
        devices=2,
        sdk_sha256=raw["sdk_sha256"],
        source_commit="84cc5dcd02a6b0ac7b346ff3cd96fbb2765d301f",  # pragma: allowlist secret -- frozen source commit
    ),
    reference=dict(
        release="full-answer-b1-20261117-r8",
        model="Qwen3.5-35B-A3B",
        kind="previously-published-website-reference",
        url="https://vllm-hust.sage.org.ai/b0-b1.html?release=r8",
        reference_sha256=raw["published_r8_reference_sha256"],
    ),
    scope=dict(
        original018_pair=False,
        project_full_five=False,
        qwen_performance_repeated=False,
        native_intern_run_complete=True,
    ),
    workloads=rows,
    quality={
        k: raw["current_mmlu200"][k]
        for k in ("requests", "correct", "truncated", "failed", "unparsed")
    },
    source=dict(
        benchmark_path="reports/r9-intern-native-20261120/native-source-summary.json.gz",
        sha256=sha256(original).hexdigest(),
        bytes=len(original),
        encoding="gzip",
        original_sha256=sha256(source).hexdigest(),
        original_bytes=len(source),
        private_raw_release="https://github.com/vLLM-HUST/vllm-hust-engine/releases/tag/full-answer-b1-20261119-r9",
    ),
    tradeoffs_zh="4K平均TPOT增加7.53%，4K首token延迟P99增加27.21%，16K首token延迟P95增加8.85%；保留全部尾部统计。",
    tradeoffs_en="4K mean TPOT increases7.53%,4K TTFT P99 increases27.21%, and16K TTFT P95 increases8.85%; all tail statistics are retained.",
)
output = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode()
if args.check:
    assert args.out.read_bytes() == output, (
        "Public summary differs from retained source arithmetic"
    )
else:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(output)
print("R9_PUBLICATION_REPRODUCED", sha256(output).hexdigest())
