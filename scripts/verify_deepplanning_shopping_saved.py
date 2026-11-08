"""Read-only original shopping input/phase/tool/scoring audit; never call a model."""

import argparse
import json
from pathlib import Path
import sys
import tarfile

from prepare_deepplanning_original_assets import CONFIG, prepare, verify
from run_deepplanning_original_shopping import select_tasks

TERMINAL = "ORIGINAL_SHOPPING_LOOPS_AND_SCORING_RETURNED_SAVED_AUDIT_PENDING"


def verify_history(payload, query, system_prompt, phase_prompt):
    messages = payload.get("messages")
    if not isinstance(messages, list) or messages[:2] != [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]:
        raise ValueError("original prompt/query mismatch")
    users = [row for row in messages if row.get("role") == "user"]
    if users != [
        {"role": "user", "content": query},
        {"role": "user", "content": phase_prompt},
    ]:
        raise ValueError("original two-stage workflow missing or replaced")
    pending = []
    calls = 0
    responses = 0
    for row in messages[2:]:
        role = row.get("role")
        if role == "assistant":
            if pending:
                raise ValueError("saved tool responses missing before next model turn")
            for call in row.get("tool_calls", []):
                pending.append(call["id"])
                calls += 1
        elif role == "tool":
            if not pending or row.get("tool_call_id") != pending.pop(0):
                raise ValueError("original tool call/result relation lost")
            responses += 1
        elif role == "user":
            if pending:
                raise ValueError("phase transition before tool results saved")
        else:
            raise ValueError("unexpected saved message role")
    if pending:
        raise ValueError("saved tool results truncated")
    return {
        "model_messages": sum(row.get("role") == "assistant" for row in messages),
        "tool_calls": calls,
        "tool_results": responses,
    }


def score_projection(score):
    return {
        key: score.get(key, default)
        for key, default in (
            ("case_name", None),
            ("case_score", 0.0),
            ("matched_count", 0),
            ("expected_count", 0),
            ("extra_products_count", 0),
            ("is_completed", True),
        )
    }


def verify_case_assets(archive_path, case, directory, level):
    with tarfile.open(archive_path, "r:gz") as archive:
        for name in ("products.jsonl", "validation_cases.json"):
            member = archive.getmember(f"database_level{level}/case_{case}/{name}")
            if (
                not member.isfile()
                or (directory / name).read_bytes() != archive.extractfile(member).read()
            ):
                raise ValueError("original product/reference bytes changed")


def audit(assets, native_root):
    assets = Path(assets).resolve()
    native_root = Path(native_root).resolve()
    prepare(assets, verify_only=True)
    result = json.loads((native_root / "native-public.json").read_text("utf-8"))
    if result.get("status") != TERMINAL:
        raise ValueError(
            "native task run is not terminal; do not audit it as completed"
        )
    if result.get("native_max_llm_calls") != 400:
        raise ValueError("original call parameter changed")
    manifest = json.loads(CONFIG.read_text("utf-8"))
    code = native_root / "author"
    prefix = "source/benchmark/deepplanning/"
    for entry in manifest["files"]:
        if entry["path"].startswith(prefix):
            name = entry["path"][len(prefix) :]
            if name != "models_config.json":
                verify(code / name, entry)
    domain = code / "shoppingplanning"
    sys.path.insert(0, str(domain))
    from agent.prompts import prompt_lib
    from agent.shopping_agent import ShoppingFnAgent
    from evaluation.evaluation_pipeline import evaluate_single_case

    phase_prompt = ShoppingFnAgent._add_to_cart(None, [])[0]["content"]
    checked = []
    completed = result.get("completed_levels", [])
    if [row["level"] for row in completed] != result.get("levels"):
        raise ValueError("selected original levels not all returned")
    for level_result in completed:
        level = level_result["level"]
        rows = json.loads(
            (domain / f"data/level_{level}_query_meta.json").read_text("utf-8")
        )
        full = result.get("selection") == "all-original-tasks"
        ids = select_tasks(rows, full)
        if ids != level_result["ids"] or level_result["native_failed"]:
            raise ValueError("original sample selection or native success mismatch")
        database = domain / f"database_infered/database_native-local_level{level}_fixed"
        summary = json.loads(
            (
                domain / "result_report" / database.name / "summary_report.json"
            ).read_text("utf-8")
        )
        saved = {row["case_name"]: row for row in summary["case_results"]}
        if len(saved) != len(ids) or len(summary["case_results"]) != len(ids):
            raise ValueError("original saved case scoring set mismatch")
        total_score = 0.0
        total_incomplete = 0
        for case in ids:
            directory = database / f"case_{case}"
            verify_case_assets(
                assets / f"databases/database_level{level}.tar.gz",
                case,
                directory,
                level,
            )
            original = next(row for row in rows if row["id"] == case)
            history = json.loads((directory / "messages.json").read_text("utf-8"))
            counts = verify_history(
                history,
                original["query"],
                getattr(prompt_lib, f"SYSTEM_PROMPT_level{level}"),
                phase_prompt,
            )
            recomputed = evaluate_single_case(directory)
            if score_projection(recomputed) != score_projection(saved[f"case_{case}"]):
                raise ValueError("original recomputed score differs from saved score")
            total_score += recomputed.get("case_score", 0.0)
            total_incomplete += not recomputed.get("is_completed", True)
            checked.append(
                {"level": level, "id": case, **counts, **score_projection(recomputed)}
            )
        statistics = summary["overall_statistics"]
        if (
            statistics["total_cases"] != len(ids)
            or statistics["incomplete_cases"] != total_incomplete
            or statistics["average_case_score"] != total_score / len(ids)
        ):
            raise ValueError("saved author score aggregation inconsistent")
    return {
        "source_id": "DEEPPLANNING",
        "domain": "shopping",
        "status": "ORIGINAL_SAVED_INPUT_PHASE_TOOL_RELATIONS_AND_AUTHOR_SCORE_CONSISTENCY_VERIFIED",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "checked_cases": checked,
        "boundary": "CPU consistency audit of saved original outputs, not model rerun, travel validation, environment replay, performance or complete-source qualification.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("fresh external receipt required")
    result = audit(args.assets, args.native_root)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "checked_cases"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
