"""Plan complete original first3 groups per official mode, never run a model.

An explicit fresh external output writes unchanged original subsets. The extra
proactive_50 release is not a separate mode in the pinned author's eval.sh.
"""

import argparse
import json
from pathlib import Path

from vllm_hust_workload_data.assets import asset_path, verify_file
from vllm_hust_workload_data.registry import source_spec
from vllm_hust_workload_data.streamingbench import adapt_case, restore_original_case

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "real": "Streaming",
    "omni": "Streaming",
    "sqa": "StreamingSQA",
    "proactive": "StreamingProactive",
    "real_stream": "StreamingOpenStreamText",
    "omni_stream": "StreamingOpenStreamText",
    "sqa_stream": "StreamingOpenStreamText",
}


def select_groups(rows):
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError(
            "original array with at least three complete outer groups required"
        )
    selected = rows[:3]
    assets, groups, questions = set(), 0, 0
    for index, row in enumerate(selected):
        record = adapt_case(
            {"original_group": row}, source={"source_id": "STREAMINGBENCH"}, index=index
        )
        if restore_original_case(record) != row:
            raise ValueError("original group changed")
        for group in row if isinstance(row, list) else [row]:
            groups += 1
            assets.add(("video", group["video_path"]))
            for question in group["questions"]:
                questions += 1
                if "audio_path" in question:
                    assets.add(("audio", question["audio_path"]))
    return selected, sorted(assets), groups, questions


def prepare(data_root, output=None):
    source = source_spec("STREAMINGBENCH")
    original_files = {item["path"]: item for item in source["files"]}
    jobs, media, subsets = [], set(), {}
    for name, benchmark in MODES.items():
        relative = f"src/data/questions_{name}.json"
        item = original_files[relative]
        verify_file(data_root, item)
        rows = json.loads(asset_path(data_root, relative).read_text("utf-8"))
        if len(rows) != item["rows"]:
            raise ValueError("original file row count differs")
        selected, assets, groups, questions = select_groups(rows)
        media.update(assets)
        subsets[Path(relative).name] = selected
        jobs.append(
            {
                "source_file": relative,
                "source_sha256": item["sha256"],
                "benchmark_name": benchmark,
                "selected_outer_rows": 3,
                "selected_nested_groups": groups,
                "selected_questions": questions,
                "original_row_indices": [0, 1, 2],
                "subset_file": Path(relative).name,
            }
        )
    report = {
        "source_id": "STREAMINGBENCH",
        "status": "FIXED_ORIGINAL_GROUP_PLAN_NOT_NATIVE_EXECUTION",
        "selection": "First3 complete original outer rows per seven official eval modes; preserve all nested questions",
        "auxiliary_proactive_50_not_counted_as_separate_native_mode": True,
        "jobs": jobs,
        "required_media": [
            {"kind": kind, "original_path": path} for kind, path in sorted(media)
        ],
        "all_original_fields_and_nested_order_preserved": True,
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if output is not None:
        output = Path(output).resolve()
        data_root = Path(data_root).resolve()
        if (
            output.is_relative_to(ROOT)
            or output == data_root
            or output.is_relative_to(data_root)
            or data_root.is_relative_to(output)
        ):
            raise ValueError(
                "fresh output must be external and separate from original data"
            )
        output.mkdir(parents=True, exist_ok=False)
        for name, rows in subsets.items():
            with (output / name).open("x", encoding="utf-8") as stream:
                json.dump(rows, stream, ensure_ascii=False)
        with (output / "preparation-public.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Fresh external subset directory; default plan only"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output)))
