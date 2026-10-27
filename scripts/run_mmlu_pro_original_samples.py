"""Run pinned original MMLU-Pro rows with lm-eval's unchanged prompt and scorer.

Install separately: lm-eval[api]==0.4.11 datasets==3.6.0 pyarrow==19.0.1
huggingface-hub==1.31.0. No model weights or datasets are distributed here.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time


FILES = {
    "validation": (
        "validation-00000-of-00001.parquet",
        "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",  # pragma: allowlist secret - original data checksum
    ),
    "test": (
        "test-00000-of-00001.parquet",
        "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",  # pragma: allowlist secret - original data checksum
    ),
}
SUBJECTS = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subject", choices=SUBJECTS, default="biology")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if metadata.version("lm-eval") != "0.4.11":
        raise RuntimeError("requires reviewed lm-eval==0.4.11")
    paths = {}
    for split, (name, expected) in FILES.items():
        path = args.data_root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("original dataset file identity mismatch")
        paths[split] = str(path.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.utils import load_yaml_config

    task_name = f"mmlu_pro_{args.subject}"
    yaml_path = Path(lm_eval.__file__).parent / "tasks/mmlu_pro" / f"{task_name}.yaml"
    config = load_yaml_config(str(yaml_path))
    config.update(
        dataset_path="parquet", dataset_name=None, dataset_kwargs={"data_files": paths}
    )
    import pyarrow.parquet as pq

    original = pq.read_table(paths["test"]).to_pylist()
    selected_rows = [
        (i, row.get("question_id"))
        for i, row in enumerate(original)
        if row["category"] == args.subject
    ][:3]
    if len(selected_rows) != 3:
        raise RuntimeError(f"three original {args.subject} rows required")
    receipt = {
        "original_test_rows_and_question_ids": selected_rows,
        "source_id": "MMLU_PRO",
        "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",  # pragma: allowlist secret - public dataset revision
        "subject": args.subject,
        "selected_samples": [f"{args.subject}:{i}" for i in range(3)],
        "selection": f"first three {args.subject}-filtered original test rows, fixed before model execution",
        "model": args.model,
        "endpoint": args.endpoint,
        "started_at_unix": time.time(),
        "runner": "lm-eval==0.4.11",
        "task_config_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        "changes_from_upstream_task": [
            "local byte-identical parquet paths",
            "three test rows only",
            "chat API transport",
            "max_gen_toks=2048",
            "enable_thinking=false",
        ],
        "prompt_filters_metric_fewshot": f"unchanged upstream {args.subject} recipe; five original validation examples",
        "claim_boundary": "integration sample, not full benchmark score or performance evaluation",
        "status": "STARTED",
        "versions": {
            name: metadata.version(name)
            for name in ("lm-eval", "datasets", "pyarrow", "huggingface-hub")
        },
    }
    target = args.output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        result = evaluator.simple_evaluate(
            model="local-chat-completions",
            model_args={
                "model": args.model,
                "base_url": args.endpoint,
                "tokenizer_backend": None,
                "tokenized_requests": False,
                "num_concurrent": 1,
                "max_retries": 0,
                "max_gen_toks": 2048,
                "timeout": 300,
            },
            tasks=[config],
            limit=3,
            batch_size=1,
            bootstrap_iters=0,
            apply_chat_template=True,
            log_samples=True,
            gen_kwargs={
                "max_gen_toks": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if result is None or result["n-samples"][task_name]["effective"] != 3:
            raise RuntimeError("expected exactly three evaluated test tasks")
        # The harness logs each task once per filter; these are not extra tasks.
        if {sample["doc_id"] for sample in result["samples"][task_name]} != {
            0,
            1,
            2,
        }:
            raise RuntimeError("evaluated sample IDs differ from preselected rows")
        # Private output contains source questions, answers and model responses.
        (args.output / "private-results.json").write_text(
            json.dumps(result, default=str, indent=2), encoding="utf-8"
        )
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            metrics=result["results"],
            sample_counts=result["n-samples"],
        )
    except Exception as error:
        receipt.update(status="HARNESS_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, default=str, indent=2), encoding="utf-8")
    print(json.dumps(receipt, default=str))


if __name__ == "__main__":
    main()
