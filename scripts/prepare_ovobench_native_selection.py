"""Plan fixed original cases and every native clip; no inference or download.

An explicit fresh external output directory writes an unchanged-field annotation
subset accepted by the original --anno_path interface. No task stage is removed.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from vllm_hust_workload_data.assets import asset_path, verify_file
from vllm_hust_workload_data.ovobench import FORWARD, TASKS
from vllm_hust_workload_data.registry import source_spec

ROOT = Path(__file__).resolve().parents[1]


def fixed_selection(rows):
    counts = Counter()
    selected, clips, seen = [], [], set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or row.get("task") not in TASKS
            or type(row.get("id")) is not int
            or row["id"] < 0
            or row["id"] in seen
        ):
            raise ValueError("unique original case identities and known tasks required")
        seen.add(row["id"])
        task = row["task"]
        if counts[task] >= 3:
            continue
        counts[task] += 1
        selected.append(row)
        if task in FORWARD:
            stages = row.get("test_info")
            if not isinstance(stages, list) or not stages:
                raise ValueError("complete original forward sequence required")
            for index in range(len(stages)):
                clips.append(
                    {
                        "case_id": row["id"],
                        "task": task,
                        "stage_index": index,
                        "path": f"{row['id']}_{index}.mp4",
                    }
                )
        else:
            clips.append(
                {
                    "case_id": row["id"],
                    "task": task,
                    "stage_index": None,
                    "path": f"{row['id']}.mp4",
                }
            )
    if set(counts) != TASKS or any(count != 3 for count in counts.values()):
        raise ValueError("all twelve original task groups need their first three cases")
    return selected, clips


def prepare(data_root, output=None):
    source = source_spec("OVOBENCH")
    original = source["files"][0]
    verify_file(data_root, original)
    rows = json.loads(asset_path(data_root, original["path"]).read_text("utf-8"))
    selected, clips = fixed_selection(rows)
    report = {
        "source_id": "OVOBENCH",
        "status": "FIXED_ORIGINAL_CASES_AND_FULL_CLIP_PLAN_NOT_EXECUTION",
        "original_annotation_sha256": original["sha256"],
        "original_case_count": len(rows),
        "selected_original_cases": len(selected),
        "selection": "First3 original cases per all12 tasks, in unchanged file order",
        "selected_original_ids": [row["id"] for row in selected],
        "required_native_clips": clips,
        "required_native_clip_count": len(clips),
        "all_forward_stages_preserved": True,
        "media_acquired": False,
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }
    if output is not None:
        output = Path(output).resolve()
        if output.is_relative_to(ROOT):
            raise ValueError("raw annotation subset must stay outside the repository")
        output.mkdir(parents=True, exist_ok=False)
        # The official runner consumes this original list of dicts without an adapter.
        with (output / "original-first3-each-task.json").open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(selected, stream, ensure_ascii=False)
        assert (
            json.loads((output / "original-first3-each-task.json").read_text("utf-8"))
            == selected
        )
        with (output / "preparation-public.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, help="Fresh external directory; omitted means plan only"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output)))
