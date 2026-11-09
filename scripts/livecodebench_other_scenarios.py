"""Pinned original execution/test-output scenarios; no research feature analysis."""

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

ASSETS = {
    "codeexecution": {
        "path": "execution-v2/data/test-00000-of-00001.parquet",
        "bytes": 71483,
        "rows": 479,
        "sha256": "c54e4377642f689758c2ee8664457af1b95ba95813a18208003c7a5a1efa410f",  # pragma: allowlist secret - publisher file identity
    },
    "testoutputprediction": {
        "path": "test_generation/test.parquet",
        "bytes": 73331,
        "rows": 442,
        "sha256": "4826aa00c059d6d47a099606ceed2d0e51d3aeeb1868f1bcf349a038bb64b4b1",  # pragma: allowlist secret - publisher file identity
    },
}


def case_identity(task, scenario):
    if scenario == "codeexecution":
        return task.id
    if scenario == "testoutputprediction":
        return json.dumps(
            [task.question_id, task.test_id], ensure_ascii=False, separators=(",", ":")
        )
    raise ValueError("explicit supported native scenario required")


def prepare_asset(data_root, output, *, scenario, sample_count=3, full=False):
    asset = ASSETS[scenario]
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from original source required")
    count = asset["rows"] if full else sample_count
    if type(count) is not int or not 1 <= count <= asset["rows"]:
        raise ValueError("positive fixed sample within original source required")
    source = data_root / asset["path"]
    if (
        source.stat().st_size != asset["bytes"]
        or hashlib.sha256(source.read_bytes()).hexdigest() != asset["sha256"]
    ):
        raise ValueError("fixed original Parquet identity mismatch")
    output.mkdir(mode=0o700, parents=True)
    filename = "original-tasks.private.parquet"
    shutil.copyfile(source, output / filename)
    receipt = {
        "source_id": "LIVECODEBENCH",
        "scenario": scenario,
        "status": "ORIGINAL_PARQUET_PREPARED_NOT_NATIVE_DELIVERY",
        "task_file_name": filename,
        "task_file_sha256": asset["sha256"],
        "source_rows": asset["rows"],
        "selected_case_count": count,
        "selected_question_ids": None,
        "selection_rule": "first_n_original_parquet_rows_no_replacement",
        "all_original_columns_preserved": True,
        "original_binary_unchanged": True,
        "private_tests_decoded": False,
        "model_calls": 0,
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def run_other(args, require_isolated_worker, author_checkout, outputs_from_saved):
    require_isolated_worker()
    asset = ASSETS[args.scenario]
    path, output = Path(args.tasks), Path(args.output).resolve()
    if (
        path.stat().st_size != asset["bytes"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]
    ):
        raise ValueError("original scenario Parquet identity mismatch")
    if output.exists():
        raise ValueError("fresh isolated output required")
    if not 1 <= args.sample_count <= asset["rows"]:
        raise ValueError("fixed original sample count required")
    output.mkdir(mode=0o700, parents=True)
    author_checkout(args.author_archive, Path("/tmp/verified-lcb-author"))
    import pyarrow.parquet as pq
    from lcb_runner.benchmarks.code_execution import CodeExecutionProblem
    from lcb_runner.benchmarks.test_output_prediction import TestOutputPredictionProblem
    from lcb_runner.evaluation import extract_instance_results
    from lcb_runner.lm_styles import LanguageModel, LMStyle
    from lcb_runner.prompts.code_execution import format_prompt_execution
    from lcb_runner.prompts.test_output_prediction import format_prompt_test_output
    from lcb_runner.runner.oai_runner import OpenAIRunner
    from lcb_runner.runner.scenario_router import combine_results, get_metrics
    from lcb_runner.utils.scenarios import Scenario

    table = pq.read_table(path)
    if table.num_rows != asset["rows"]:
        raise ValueError("complete original source row count mismatch")
    rows = table.slice(0, args.sample_count).to_pylist()
    execution = args.scenario == "codeexecution"
    tasks = [
        (CodeExecutionProblem if execution else TestOutputPredictionProblem)(**row)
        for row in rows
    ]
    model = LanguageModel(args.model, args.model, LMStyle.OpenAIChat, None)
    scenario = Scenario.codeexecution if execution else Scenario.testoutputprediction
    formatter = format_prompt_execution if execution else format_prompt_test_output
    defaults = SimpleNamespace(
        model=args.model,
        use_cache=False,
        n=10,
        temperature=0.2,
        top_p=0.95,
        max_tokens=2000,
        openai_timeout=90,
        scenario=scenario,
        cot_code_execution=False,
        num_process_evaluate=12,
        timeout=6,
    )
    runner = OpenAIRunner(defaults, model)
    requests = []
    for task in tasks:
        body = {
            key: value
            for key, value in runner.client_kwargs.items()
            if key != "timeout"
        }
        body["messages"] = formatter(task, model.model_style)
        row = {
            "question_id": task.question_id,
            "request_id": case_identity(task, args.scenario),
            "request": body,
        }
        row.update({"id": task.id} if execution else {"test_id": task.test_id})
        requests.append(row)
    if len({row["request_id"] for row in requests}) != len(requests):
        raise ValueError("original native case identities must be distinct")
    receipt = {
        "source_id": "LIVECODEBENCH",
        "scenario": args.scenario,
        "selected_question_ids": [task.question_id for task in tasks],
        "selected_case_ids": [row["request_id"] for row in requests],
        "source_rows": table.num_rows,
        "source_columns": table.column_names,
        "selection_rule": "first_n_original_parquet_rows_no_replacement",
        "cot_code_execution": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "all_four_scenarios_complete": False,
    }
    if args.prepare_prompts:
        (output / "requests-private.json").write_text(
            json.dumps(requests, ensure_ascii=False), encoding="utf-8"
        )
        receipt["status"] = "ORIGINAL_API_PROMPTS_PREPARED_NO_MODEL_CALLS"
    else:
        results = []
        for index, request in enumerate(requests):
            saved = json.loads(
                (
                    Path(args.generations) / f"{index}.request-response.private.json"
                ).read_text("utf-8")
            )
            if saved.get("request_id") != request["request_id"]:
                raise ValueError("original native case relationship differs")
            results.append(
                outputs_from_saved(
                    saved,
                    question_id=request["question_id"],
                    messages=request["request"]["messages"],
                    model=args.model,
                )
            )
        combined = combine_results(scenario, results, model, False)
        metrics = get_metrics(scenario, defaults, tasks, combined)
        graded = extract_instance_results(metrics[1])
        if not len(tasks) == len(combined) == len(graded):
            raise ValueError("complete original scoring relationships required")
        evaluation = [
            task.insert_output_evaluation(outputs, predicted, grades)
            for task, (outputs, predicted), grades in zip(tasks, combined, graded)
        ]
        (output / "original-metrics-private.json").write_text(
            json.dumps(metrics), encoding="utf-8"
        )
        (output / "original-eval-all-private.json").write_text(
            json.dumps(evaluation), encoding="utf-8"
        )
        receipt.update(
            status="ORIGINAL_SCENARIO_SAMPLE_SCORED_SOURCE_CLOSURE_PENDING",
            response_candidate_counts=[len(row) for row in results],
            graded_candidate_counts=[len(row) for row in graded],
            metric_unit="author percentage" if execution else "author fraction",
        )
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt))
