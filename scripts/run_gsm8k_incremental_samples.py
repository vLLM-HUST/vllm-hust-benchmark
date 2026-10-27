"""Run preselected original GSM8K ranges with the pinned prompt and scorer.

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
    "train": (
        "train-00000-of-00001.parquet",
        "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",  # pragma: allowlist secret - original dataset SHA
    ),
    "test": (
        "test-00000-of-00001.parquet",
        "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",  # pragma: allowlist secret - original dataset SHA
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 1319:
        parser.error("sample range must be within the 1319 original test rows")
    selected = list(range(args.start, args.start + args.count))
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

    yaml_path = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    config = load_yaml_config(str(yaml_path))
    config.update(
        dataset_path="parquet", dataset_name=None, dataset_kwargs={"data_files": paths}
    )
    receipt = {
        "source_id": "GSM8K",
        "revision": "740312add88f781978c0658806c59bc2815b9866",  # pragma: allowlist secret - public dataset revision
        "selected_samples": [f"test:{index}" for index in selected],
        "selection": "contiguous original test rows, fixed before model execution",
        "model": args.model,
        "endpoint": args.endpoint,
        "started_at_unix": time.time(),
        "runner": "lm-eval==0.4.11",
        "task_config_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        "changes_from_upstream_task": [
            "local byte-identical parquet paths",
            f"{args.count} preselected test rows only",
            "chat API transport",
            "max_gen_toks=2048",
            "enable_thinking=false",
        ],
        "prompt_filters_metric_fewshot": "unchanged upstream gsm8k recipe; five training examples",
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
            samples={"gsm8k": selected},
            batch_size=1,
            bootstrap_iters=0,
            apply_chat_template=True,
            log_samples=True,
            gen_kwargs={
                "max_gen_toks": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if result is None:
            raise RuntimeError("evaluation returned no result")
        # lm-eval's n-samples effective count ignores explicit sample IDs in
        # this version; validate actual logged document IDs instead.
        # The harness logs each task once per filter; these are not extra tasks.
        if {sample["doc_id"] for sample in result["samples"]["gsm8k"]} != set(selected):
            raise RuntimeError("evaluated sample IDs differ from preselected rows")
        # Private output contains source questions, answers and model responses.
        (args.output / "private-results.json").write_text(
            json.dumps(result, default=str, indent=2), encoding="utf-8"
        )
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            metrics=result["results"],
            sample_counts=result["n-samples"],
            verified_unique_tasks=len(selected),
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
