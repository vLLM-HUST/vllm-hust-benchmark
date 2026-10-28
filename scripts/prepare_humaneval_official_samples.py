"""Reuse original HumanEval generations as official scorer input; execute no code.

Run the author functional evaluator only in a separately verified sandbox. This
preparation is not functional evaluation or a source-completion claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_humaneval_original_generation import DATA_SHA, REVISION, completion_payload


def prepare(generation_dir, *, records, output):
    generation_dir, output = Path(generation_dir).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(generation_dir)
        or generation_dir.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from historical generation required")
    expected_ids = ["HumanEval/0", "HumanEval/1", "HumanEval/2"]
    if [record.task["task_id"] for record in records] != expected_ids:
        raise ValueError("exact original preselected tasks required")
    receipt_path = generation_dir / "receipt.json"
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    if (
        receipt.get("schema_version") != "humaneval-generation/v1"
        or receipt.get("source_id") != "HUMANEVAL"
        or receipt.get("revision") != REVISION
        or receipt.get("data_sha256") != DATA_SHA
        or receipt.get("status") != "THREE_ORIGINAL_PROMPTS_GENERATED_NOT_SCORED"
        or receipt.get("generation_config")
        != {"temperature": 0, "max_tokens": 2048, "n": 1}
    ):
        raise ValueError("audited original generation receipt required")
    samples = receipt.get("samples", [])
    if [item.get("task_id") for item in samples] != expected_ids:
        raise ValueError("generation task selection mismatch")
    completions, identities = [], []
    for record, item in zip(records, samples):
        relative = item["task_id"].replace("/", "-") + ".private.json"
        path = (generation_dir / relative).resolve()
        if not path.is_relative_to(generation_dir):
            raise ValueError("escaped historical generation path")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if (
            digest != item.get("private_result_sha256")
            or item.get("status") != "GENERATION_RECEIVED_NOT_SCORED"
        ):
            raise ValueError("historical completion identity or status mismatch")
        saved = json.loads(raw)
        request = completion_payload(record, receipt["model"])
        if (
            saved.get("request") != request
            or item.get("prompt_sha256")
            != hashlib.sha256(request["prompt"].encode()).hexdigest()
        ):
            raise ValueError("original prompt or generation configuration changed")
        choices = saved.get("response", {}).get("choices", [])
        if (
            len(choices) != 1
            or not isinstance(choices[0].get("text"), str)
            or not choices[0]["text"]
            or choices[0].get("finish_reason") not in {"stop", "length"}
        ):
            raise ValueError("original terminal completion required")
        completions.append(
            {"task_id": item["task_id"], "completion": choices[0]["text"]}
        )
        identities.append(
            {"task_id": item["task_id"], "historical_result_sha256": digest}
        )
    if receipt_path.read_bytes() != receipt_raw:
        raise ValueError("historical generation receipt changed during preparation")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    samples_path = output / "samples-private.jsonl"
    with samples_path.open("x", encoding="utf-8") as stream:
        for item in completions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = {
        "status": "OFFICIAL_SCORER_INPUT_PREPARED_NOT_EXECUTED",
        "source_id": "HUMANEVAL",
        "source_revision": REVISION,
        "original_generation_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "historical_samples": identities,
        "new_model_calls": 0,
        "completion_postprocessing": "NONE_EXACT_ORIGINAL_TEXT",
        "functional_tests_executed": False,
        "source_checkbox_eligible": False,
        "next_gate": "Author evaluate_functional_correctness with original problem file in separately verified isolation; never on model server host process",
    }
    (output / "receipt-private.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.registry import source_spec

    source = source_spec("HUMANEVAL")
    if source["revision"] != REVISION or source["files"][0]["sha256"] != DATA_SHA:
        raise ValueError("exact reviewed original source required")
    records = list(iter_workload("HUMANEVAL", data_root=args.data_root, limit=3))
    prepare(args.generation_dir, records=records, output=args.output)


if __name__ == "__main__":
    main()
