"""Selection/frame-boundary fixtures; no raw videos, toolkit or model calls."""

import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "videomme2_prompts", SCRIPTS / "prepare_videomme_v2_original_prompts.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)
VIDEOS = ["001", "002", "003", "009", "013", "016"]


def annotations():
    return [
        {
            "video_id": f"{video:03d}",
            "group_type": "relevance" if video in (9, 13, 16) else "logic",
            "group_structure": "4",
            "question_id": f"{video:03d}-{question}",
        }
        for video in range(1, 801)
        for question in range(1, 5)
    ]


def test_frozen_selection_retains_all_four_original_questions():
    selected = entry.fixed_positions(annotations(), VIDEOS)
    assert selected == list(range(12)) + list(range(32, 36)) + list(
        range(48, 52)
    ) + list(range(60, 64))


def test_no_shortened_annotation_table_or_replaced_group():
    with pytest.raises(ValueError, match="all3200"):
        entry.fixed_positions(annotations()[:-1], VIDEOS)
    with pytest.raises(ValueError, match="never replace"):
        entry.fixed_positions(annotations(), ["001", "002", "003", "009", "013", "017"])


def test_group_question_order_not_repaired():
    rows = annotations()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="order"):
        entry.fixed_positions(rows, VIDEOS)


@pytest.mark.parametrize("count", [32, 63, 65])
def test_original64_frames_cannot_be_reduced_to_service_limit(count):
    prompt = [{"type": "image", "value": f"frame{i}"} for i in range(count)] + [
        {"type": "text", "value": "original question and options"}
    ]
    with pytest.raises(ValueError, match="original64"):
        entry.check_prompt(
            prompt, {"question": "original question", "options": "options"}
        )


def test_plan_mode_does_not_import_native_toolkit_or_load_assets():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_videomme_v2_original_prompts.py")],
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert value["status"] == "PLAN_ONLY" and value["model_calls"] == 0
    assert value["frames_each"] == 64 and value["selected_questions"] == 24


def test_original_prompt_preserves64_distinct_frames_and_question_options():
    images = [{"type": "image", "value": f"frame{i}"} for i in range(64)]
    original = {"question": "What happens?", "options": "A. First\nB. Second"}
    prompt = images + [
        {"type": "text", "value": "Question: What happens?\nA. First\nB. Second"}
    ]
    assert entry.check_prompt(prompt, original) == [f"frame{i}" for i in range(64)]
    with pytest.raises(ValueError, match="original64"):
        entry.check_prompt([images[0]] * 64 + prompt[-1:], original)
    with pytest.raises(ValueError, match="original64"):
        entry.check_prompt(
            images + [{"type": "text", "value": "What happens?"}], original
        )


def test_generated_tsv_requires_every_original_field_and_row(tmp_path):
    rows = annotations()
    fields = [
        "url",
        "question",
        "options",
        "answer",
        "level",
        "second_head",
        "third_head",
    ]
    for row in rows:
        row.update({key: f"original {key}" for key in fields})
    columns = [
        "index",
        "video",
        "video_path",
        "subtitle_path",
        "url",
        "group_type",
        "group_structure",
        "question_id",
        "question",
        "options",
        "answer",
        "level",
        "second_head",
        "third_head",
    ]
    generated = []
    for i, row in enumerate(rows):
        video = row["video_id"]
        record = {key: row[key] for key in columns if key in row}
        record.update(
            index=str(i),
            video=video,
            video_path=f"./video/{video}.mp4",
            subtitle_path=f"./subtitle/{video}.jsonl",
        )
        generated.append(record)
    path = tmp_path / "native.tsv"

    def save():
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(generated)

    save()
    assert entry.verify_original_tsv(path, rows) == 3200
    generated[-1]["answer"] = "changed"
    save()
    with pytest.raises(ValueError, match="row3199"):
        entry.verify_original_tsv(path, rows)
