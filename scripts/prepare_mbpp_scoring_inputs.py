"""Bind saved original MBPP generations to unchanged tasks; never executes code."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_mbpp_original_assets import PINS
from prepare_xsum_original_assets import bound_path, verify
from run_mbpp_original_generation import completion_request, selected_tasks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_saved_generation(generation, *, rows, examples, source_identity):
    selected_tasks(rows, full=True)
    generation = Path(generation).resolve()
    receipt_file = generation / "receipt-private.json"
    receipt_bytes = receipt_file.read_bytes()
    receipt = json.loads(receipt_bytes)
    if (
        receipt.get("source_id") != "MBPP"
        or receipt.get("variant") != "full"
        or receipt.get("status") != "ORIGINAL_TEST_GENERATION_COMPLETE_SCORING_PENDING"
        or receipt.get("source_identity") != source_identity
        or receipt.get("demonstration_task_ids") != [2, 3, 4]
        or receipt.get("functional_tests_executed") is not False
        or type(receipt.get("max_tokens")) is not int
        or receipt["max_tokens"] <= 0
        or type(receipt.get("context_capacity")) is not int
        or receipt["context_capacity"] <= receipt["max_tokens"]
    ):
        raise ValueError(
            "terminal original full generation and asset identity required"
        )
    samples = receipt.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("at least three original generated tasks required")
    indices = [sample.get("original_row_index") for sample in samples]
    if (
        any(type(index) is not int or not 0 <= index < len(rows) for index in indices)
        or indices != sorted(set(indices))
        or indices != receipt.get("selected_original_rows")
        or [rows[index]["task_id"] for index in indices]
        != receipt.get("selected_task_ids")
    ):
        raise ValueError("fixed original selection identity mismatch")
    tasks, identities = [], []
    for sample, index in zip(samples, indices):
        task = rows[index]
        expected_name = f"task-{task['task_id']}.private.json"
        if (
            sample.get("task_id") != task["task_id"]
            or sample.get("private_file") != expected_name
        ):
            raise ValueError("original task/file identity mismatch")
        path = generation / expected_name
        if path.is_symlink() or not path.resolve().is_relative_to(generation):
            raise ValueError(
                "original generation file must be contained and nonsymlink"
            )
        raw = path.read_bytes()
        saved = json.loads(raw)
        if (
            saved.get("status") != "GENERATED_NOT_FUNCTIONALLY_SCORED"
            or saved.get("original_row_index") != index
            or saved.get("original_task") != task
            or saved.get("request")
            != completion_request(
                task, examples, receipt["model"], receipt["max_tokens"]
            )
        ):
            raise ValueError(
                "original input, prompt, or generation configuration changed"
            )
        choices = saved.get("response", {}).get("choices")
        usage = saved.get("response", {}).get("usage", {})
        token_count = saved.get("tokenize", {}).get("count")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0].get("text"), str)
            or choices[0].get("finish_reason") not in {"stop", "length"}
            or choices[0]["finish_reason"] != sample.get("finish_reason")
            or saved["response"].get("usage") != sample.get("usage")
            or type(token_count) is not int
            or token_count <= 0
            or token_count + receipt["max_tokens"] > receipt["context_capacity"]
            or usage.get("prompt_tokens") != token_count
            or type(usage.get("completion_tokens")) is not int
            or not 0 <= usage["completion_tokens"] <= receipt["max_tokens"]
        ):
            raise ValueError("unchanged terminal raw completion required")
        setup = task.get("test_setup_code")
        if not isinstance(setup, str) or not isinstance(task.get("code"), str):
            raise ValueError(
                "original setup and reference code required; no inferred fallback"
            )
        tasks.append(
            {
                "original_row_index": index,
                "task_id": task["task_id"],
                "completion": choices[0]["text"],
                "finish_reason": choices[0]["finish_reason"],
                "original_task": task,
            }
        )
        identities.append(
            {"path": expected_name, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    if receipt_file.read_bytes() != receipt_bytes or any(
        digest(generation / item["path"]) != item["sha256"] for item in identities
    ):
        raise ValueError("saved generation changed during preparation")
    return tasks, {
        "generation_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "generation_files": identities,
    }


def prepare(generation, *, rows, examples, source_identity, output, final_verify=None):
    generation, output = Path(generation).resolve(), Path(output).resolve()
    if output.is_relative_to(generation) or generation.is_relative_to(output):
        raise ValueError("fresh output must be separate from frozen generation")
    tasks, provenance = validate_saved_generation(
        generation, rows=rows, examples=examples, source_identity=source_identity
    )
    if final_verify is not None:
        final_verify()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    task_file = output / "tasks-private.json"
    task_file.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
    task_file.chmod(0o600)
    result = {
        "source_id": "MBPP",
        "status": "ORIGINAL_TASK_SCORING_INPUTS_PREPARED_NOT_EXECUTED",
        **provenance,
        "source_identity": source_identity,
        "tasks_sha256": digest(task_file),
        "task_ids": [item["task_id"] for item in tasks],
        "completion_postprocessing": "NONE_EXACT_GENERATED_TEXT",
        "empty_and_length_outputs_retained": True,
        "functional_tests_executed": False,
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "receipt-private.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    root, out = args.assets.resolve(), args.output.resolve()
    if root.is_relative_to(out) or out.is_relative_to(root):
        parser.error("private scoring inputs must be separate from original assets")

    def verify_all():
        for item in pins["dataset"]["files"]:
            verify(bound_path(root, "data", item), item)

    verify_all()
    import pyarrow.parquet as pq

    rows = pq.read_table(root / "data/full/test-00000-of-00001.parquet").to_pylist()
    demos = pq.read_table(root / "data/full/prompt-00000-of-00001.parquet").to_pylist()
    if len(demos) != 10 or {item["task_id"] for item in demos} != set(range(1, 11)):
        raise ValueError("complete original prompt split required")
    lookup = {item["task_id"]: item for item in demos}
    result = prepare(
        args.generation,
        rows=rows,
        examples=[lookup[i] for i in (2, 3, 4)],
        source_identity={
            "repo_id": pins["dataset"]["repo_id"],
            "revision": pins["dataset"]["revision"],
            "asset_manifest_sha256": digest(PINS),
            "files": pins["dataset"]["files"],
        },
        output=out,
        final_verify=verify_all,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "tasks": len(result["task_ids"]),
                "new_model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
