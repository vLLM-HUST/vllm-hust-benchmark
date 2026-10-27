"""Audit saved original DeepPlanning travel tasks and rescore without a model.

Private original histories/reports/plans remain external. Recomputed author
scores go to a fresh separate directory, never overwrite the native evidence.
This is protocol/score consistency, not quality, performance or causal analysis.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
from threading import Lock
import zipfile

from prepare_deepplanning_original_assets import CONFIG, prepare, verify
from run_deepplanning_original_shopping import select_tasks

TERMINAL = "ORIGINAL_TRAVEL_LOOPS_CONVERSION_AND_SCORING_RETURNED_SAVED_AUDIT_PENDING"


def verify_history(payload, query, system_prompt):
    messages = payload.get("messages")
    if not isinstance(messages, list) or messages[:2] != [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]:
        raise ValueError("original travel system prompt/query mismatch")
    pending = []
    calls = assistants = 0
    for row in messages[2:]:
        if row.get("role") == "assistant":
            if pending:
                raise ValueError("missing original tool results")
            assistants += 1
            for call in row.get("tool_calls") or []:
                pending.append(call["id"])
                calls += 1
        elif row.get("role") == "tool":
            if not pending or row.get("tool_call_id") != pending.pop(0):
                raise ValueError("original tool-call/result relationship lost")
        else:
            raise ValueError("unexpected user/system message injected into travel task")
    if pending or not assistants:
        raise ValueError("truncated or empty original travel history")
    return {"model_messages": assistants, "tool_pairs": calls}


def verify_database(archive_path, database, language, case):
    count = 0
    prefix = f"database_{language}/id_{case}/"
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            relative = member.filename[len(prefix) :]
            path = (database / f"id_{case}" / relative).resolve()
            if not path.is_relative_to((database / f"id_{case}").resolve()):
                raise ValueError("invalid original database member path")
            if path.read_bytes() != archive.read(member):
                raise ValueError("original travel database/reference bytes changed")
            count += 1
    if not count:
        raise ValueError("original selected database absent from publisher archive")
    return count


def verify_aggregate(saved, recomputed):
    keys = ("total", "success", "failed", "average_score", "pass_rate", "metrics")
    if any(saved.get(key) != recomputed.get(key) for key in keys):
        raise ValueError("original full-test-denominator score aggregation changed")
    return {key: recomputed.get(key) for key in keys}


def audit(assets, native_root, output):
    assets, native_root, output = map(
        lambda value: Path(value).resolve(), (assets, native_root, output)
    )
    repository = Path(__file__).resolve().parents[1]
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (assets, native_root, repository)
    ):
        raise ValueError("fresh independent external audit directory required")
    prepare(assets, verify_only=True)
    result = json.loads((native_root / "native-public.json").read_text("utf-8"))
    if result.get("status") != TERMINAL or result.get("native_max_llm_calls") != 400:
        raise ValueError("original native run not terminal or call budget changed")
    language = result.get("language")
    if language not in ("en", "zh"):
        raise ValueError("original language missing")
    code = native_root / "author"
    prefix = "source/benchmark/deepplanning/"
    manifest = json.loads(CONFIG.read_text("utf-8"))
    for entry in manifest["files"]:
        if entry["path"].startswith(prefix):
            name = entry["path"][len(prefix) :]
            if name != "models_config.json":
                verify(code / name, entry)
    domain = code / "travelplanning"
    rows = json.loads(
        (domain / f"data/travelplanning_query_{language}.json").read_text("utf-8")
    )
    ids = select_tasks(rows, result.get("full", False))
    if ids != result.get("ids"):
        raise ValueError("fixed original sample selection changed")
    for phase in ("inference", "conversion", "scoring"):
        stats = result.get(phase, {})
        if (
            stats.get("total") != len(ids)
            or stats.get("failed") != 0
            or stats.get("success") != len(ids)
        ):
            raise ValueError("selected tasks did not complete all original phases")
    sys.path.insert(0, str(domain))
    from agent.prompts import get_system_prompt
    from evaluation.eval_converted import evaluate_plans, process_single_evaluation

    output.mkdir(mode=0o700)
    native = native_root / "native-results"
    database = domain / f"database/database_{language}"
    checked = []
    for case in ids:
        original = next(row for row in rows if str(row["id"]) == str(case))
        history = json.loads(
            (native / f"trajectories/id_{case}.json").read_text("utf-8")
        )
        if (
            history.get("query") != original["query"]
            or history.get("language") != language
            or history.get("success") is not True
        ):
            raise ValueError("original task identity/input/success missing")
        counts = verify_history(history, original["query"], get_system_prompt(language))
        report = native / f"reports/id_{case}.txt"
        if report.read_text("utf-8") != history.get("final_plan"):
            raise ValueError("saved original final report changed")
        database_files = verify_database(
            assets / f"databases/database_{language}.zip", database, language, case
        )
        saved = json.loads(
            (native / f"evaluation/id_{case}_score.json").read_text("utf-8")
        )
        recomputed = process_single_evaluation(
            native / f"converted_plans/id_{case}_converted.json",
            rows,
            output,
            database,
            Lock(),
        )
        if recomputed.get("success") is not True:
            raise ValueError("author scorer failed on unchanged saved plan")
        fresh = json.loads((output / f"id_{case}_score.json").read_text("utf-8"))
        if fresh != saved:
            raise ValueError("recomputed original score differs from saved score")
        checked.append(
            {
                "id": case,
                "database_files": database_files,
                **counts,
                "scores": recomputed["scores"],
            }
        )
    aggregate_root = output / "aggregate"
    plans = aggregate_root / "converted_plans"
    plans.mkdir(parents=True)
    for case in ids:
        name = f"id_{case}_converted.json"
        shutil.copyfile(native / "converted_plans" / name, plans / name)
    recomputed_summary = evaluate_plans(
        result_dir=aggregate_root,
        test_data_path=domain / f"data/travelplanning_query_{language}.json",
        database_dir=database,
        workers=1,
        verbose=False,
    )
    aggregate = verify_aggregate(result["scoring"], recomputed_summary)
    return {
        "source_id": "DEEPPLANNING",
        "domain": "travel",
        "language": language,
        "status": "ORIGINAL_SAVED_TRAVEL_INPUT_TOOL_REPORT_DATABASE_AND_AUTHOR_SCORE_VERIFIED",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "checked_cases": checked,
        "official_aggregate_recomputed": aggregate,
        "official_test_denominator": len(rows),
        "validated_subset_count": len(ids),
        "denominator_boundary": "Publisher aggregate includes the complete original language test set, not only the sampled tasks; do not present it as sampled mean or full-run quality.",
        "boundary": "Saved original workflow consistency only. Converter request/response bodies are not retained by publisher; no claim of independently replayed conversion, author-model reproduction, complete source or performance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "native-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = audit(args.assets, args.native_root, args.output)
    (Path(args.output) / "saved-audit-public.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
