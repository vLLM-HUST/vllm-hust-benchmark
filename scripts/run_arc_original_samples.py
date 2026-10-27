"""Run fixed original AI2-ARC ranges or the full split with upstream likelihood scoring.

Separate environment: lm-eval[api]==0.4.11 datasets==3.6.0 pyarrow==19.0.1
huggingface-hub==1.31.0. Output includes private tasks/answers: never publish it.
"""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

REVISION = "210d026faf9955653af8916fad021475a3f00453"  # pragma: allowlist secret - public dataset revision
TEST_COUNTS = {"ARC-Challenge": 1172, "ARC-Easy": 2376}


def selected_indices(suite, start=0, count=3, full=False):
    total = TEST_COUNTS[suite]
    if full:
        if start != 0 or count != 3:
            raise ValueError("--full cannot be combined with an explicit range")
        return list(range(total))
    if start < 0 or count < 1 or start + count > total:
        raise ValueError("sample range must be within the original test split")
    return list(range(start, start + count))


SUITE_HASHES = {
    "ARC-Challenge": {
        "train": "e488c1587ffdcfc8443f916c53488a95cd471c5790e0746c6bfe4cecf20962cb",  # pragma: allowlist secret - original parquet SHA
        "validation": "395a5c88d1580d69855fbaee9450270578df1ad5af6259771cd0a42c20e99f05",  # pragma: allowlist secret - original parquet SHA
        "test": "62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9",  # pragma: allowlist secret - original parquet SHA
    },
    "ARC-Easy": {
        "train": "b315db8a4be597dc7daa50a4e70d48dd7c990c32085629e6ccd8c926beaa80b5",  # pragma: allowlist secret - original parquet SHA
        "validation": "ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74",  # pragma: allowlist secret - original parquet SHA
        "test": "4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177",  # pragma: allowlist secret - original parquet SHA
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--suite", choices=tuple(SUITE_HASHES), default="ARC-Challenge")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Full /v1/completions URL, not chat/completions",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Local tokenizer matching the served model",
    )
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    try:
        selected = selected_indices(args.suite, args.start, args.count, args.full)
    except ValueError as error:
        parser.error(str(error))
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run

    if metadata.version("lm-eval") != "0.4.11":
        raise RuntimeError("requires reviewed lm-eval==0.4.11")
    if not args.endpoint.endswith("/v1/completions"):
        raise ValueError("preserve original likelihood protocol via completions")
    if args.max_length <= 0 or not args.tokenizer.is_dir():
        raise ValueError(
            "positive context length and explicit local tokenizer required"
        )
    hashes = SUITE_HASHES[args.suite]
    files = {
        s: args.data_root / f"{args.suite}/{s}-00000-of-00001.parquet" for s in hashes
    }
    for split, path in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[split]:
            raise ValueError("original dataset file identity mismatch")
    binding = bind_source(
        "AI2_ARC",
        data_root=args.data_root,
        files=[f"{args.suite}/test-00000-of-00001.parquet"],
        revision=REVISION,
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = args.output / "binding.json"
    manifest.write_text(json.dumps(binding), encoding="utf-8")
    preparation = run(
        {
            "sources": [
                {
                    "source_id": "AI2_ARC",
                    "data_root": str(args.data_root.resolve()),
                    "manifest": str(manifest.resolve()),
                    "limit": selected[-1] + 1,
                }
            ]
        },
        output=args.output / "prepared",
    )
    database = args.output / "prepared/workload.sqlite"
    all_rows = list(prepared_rows(database))
    rows = [all_rows[index] for index in selected]
    if len(rows) != len(selected):
        raise ValueError("selected original inputs are missing")
    for row in rows:
        record = resolve_input(database, row["record_id"], data_root=args.data_root)
        if "answerKey" in record.input_record()["task"]["input"]:
            raise ValueError("reference answer leaked into model input")
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.utils import load_yaml_config

    task_root = Path(lm_eval.__file__).parent / "tasks/arc"
    task_name = "arc_easy" if args.suite == "ARC-Easy" else "arc_challenge"
    task_config = task_root / f"{task_name}.yaml"
    config = load_yaml_config(str(task_config))
    config.update(
        dataset_path="parquet",
        dataset_name=None,
        dataset_kwargs={"data_files": {s: str(p.resolve()) for s, p in files.items()}},
    )
    receipt = {
        "schema_version": "workload-server-validation/v1",
        "source_id": "AI2_ARC",
        "subsuite": args.suite,
        "revision": REVISION,
        "selected_samples": [f"test:{index}" for index in selected],
        "selection": f"original {args.suite} test indices, fixed before execution",
        "runner": "lm-eval==0.4.11",
        "served_model": args.model,
        "protocol": "original multiple-choice loglikelihood; no chat conversion",
        "changes_from_upstream_task": [
            "local byte-identical parquet paths",
            f"{len(selected)} preselected original test rows",
        ],
        "claim_boundary": f"Only the requested original {args.suite} indices under the declared model; no performance claim",
        "full_test_split_selected": args.full,
        "status": "STARTED",
        "max_length": args.max_length,
        "task_config_sha256": hashlib.sha256(task_config.read_bytes()).hexdigest(),
        "tokenizer_sha256": {
            name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        },
        "reader_validation": {
            "prepared_records": len(all_rows),
            "selected_records": len(rows),
            "resolved_records": len(rows),
            "code_sha256": preparation["code_sha256"],
        },
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    target = args.output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    try:
        result = evaluator.simple_evaluate(
            model="local-completions",
            model_args={
                "model": args.model,
                "base_url": args.endpoint,
                "tokenizer": str(args.tokenizer),
                "tokenizer_backend": "huggingface",
                "tokenized_requests": True,
                "num_concurrent": 1,
                "max_retries": 0,
                "max_length": args.max_length,
                "truncate": False,
                "timeout": 120,
            },
            tasks=[config],
            samples={task_name: selected},
            batch_size=1,
            bootstrap_iters=0,
            log_samples=True,
        )
        if result is None:
            raise RuntimeError("evaluation returned no result")
        if {s["doc_id"] for s in result["samples"][task_name]} != set(selected):
            raise RuntimeError("sample IDs differ from preselection")
        private = args.output / "private-results.json"
        private.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            metrics=result["results"],
            sample_counts=result["n-samples"],
            private_results_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
        )
    except Exception as error:
        receipt.update(
            status="HARNESS_OR_BACKEND_ERROR", error_type=type(error).__name__
        )
        raise
    finally:
        target.write_text(json.dumps(receipt, default=str, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: receipt[k] for k in ("source_id", "status", "metrics", "sample_counts")}
        )
    )


if __name__ == "__main__":
    main()
