"""Score saved original GAIA validation predictions with the pinned author function.

Default plan does not read assets. This does not run an Agent or establish how
predictions were generated. Test references are withheld and cannot be scored.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
from pathlib import Path

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.assets import sha256 as digest
from vllm_hust_workload_data.registry import source_spec

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_REVISION = "9f133d71362e77b3539f1514f31b9c101a545fec"  # pragma: allowlist secret - official public leaderboard software revision
SCORER_SHA = "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2"  # pragma: allowlist secret - official public scorer.py digest
VALIDATION_FILE = "2023/validation/metadata.parquet"


def load_author(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SCORER_SHA:
        raise ValueError("unchanged pinned official GAIA scorer.py required")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("gaia_original_leaderboard_scorer", loader=None)
    )
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def score(*, data_root, scorer_file, predictions_file, output):
    data_root = Path(data_root).resolve(strict=True)
    scorer_file = Path(scorer_file).resolve(strict=True)
    predictions_file = Path(predictions_file).resolve(strict=True)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("never overwrite previous scoring output")
    output = output.resolve()
    if (
        output.is_relative_to(ROOT)
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError(
            "fresh private scoring output outside checkout and original assets required"
        )
    initial_predictions_sha = digest(predictions_file)
    records = list(iter_workload("GAIA", data_root=data_root, files=[VALIDATION_FILE]))
    originals = {r.task["task_id"]: r for r in records}
    if not originals or len(originals) != len(records):
        raise ValueError("original validation task identities must be unique")
    predictions = []
    seen = set()
    with predictions_file.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {"task_id", "prediction"}:
                raise ValueError(
                    "predictions require only original task_id and text/null prediction"
                )
            identity = item["task_id"]
            if (
                not isinstance(identity, str)
                or identity not in originals
                or identity in seen
            ):
                raise ValueError(
                    "distinct original validation IDs required; no test or replacement task"
                )
            if item["prediction"] is not None and not isinstance(
                item["prediction"], str
            ):
                raise ValueError(
                    "original scorer accepts a saved text or explicit null failure"
                )
            seen.add(identity)
            predictions.append(item)
    if not predictions:
        raise ValueError("nonempty saved original predictions required")
    author = load_author(scorer_file)
    receipt = {
        "source_id": "GAIA",
        "status": "SCORING_SAVED_ORIGINAL_VALIDATION_OUTPUTS",
        "data_revision": source_spec("GAIA")["revision"],
        "author_space": "gaia-benchmark/leaderboard",
        "author_revision": AUTHOR_REVISION,
        "scorer_sha256": SCORER_SHA,
        "predictions_sha256": initial_predictions_sha,
        "original_validation_tasks": len(originals),
        "selected_original_validation_tasks": len(predictions),
        "unscored_original_validation_tasks": len(originals) - len(predictions),
        "model_calls": 0,
        "native_agent_tasks_executed": 0,
        "generation_provenance_verified": False,
        "source_checkbox_eligible": False,
    }
    output.mkdir(mode=0o700, parents=True)
    marker = output / "receipt-private.json"
    try:
        results = []
        with (
            (output / "author-stdout-private.txt").open(
                "x", encoding="utf-8"
            ) as stdout,
            (output / "author-stderr-private.txt").open(
                "x", encoding="utf-8"
            ) as stderr,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            for item in predictions:
                record = originals[item["task_id"]]
                correct = author.question_scorer(
                    item["prediction"], record.outcome["reference_answer"]
                )
                if type(correct) is not bool:
                    raise ValueError("original scorer returned a non-boolean result")
                results.append(
                    {
                        "task_id": item["task_id"],
                        "original_row_index": record.replay["source_row_index"],
                        "level": record.replay["annotations"]["level"],
                        "prediction_is_null": item["prediction"] is None,
                        "correct": correct,
                    }
                )
        if (
            digest(predictions_file) != initial_predictions_sha
            or digest(scorer_file) != SCORER_SHA
        ):
            raise ValueError("saved scoring inputs changed during evaluation")
        receipt.update(
            status="SAVED_ORIGINAL_GAIA_VALIDATION_OUTPUTS_AUTHOR_SCORED_NOT_AGENT_EXECUTION",
            correct=sum(x["correct"] for x in results),
            accuracy=sum(x["correct"] for x in results) / len(results),
            metric_scale="0_to_1",
            tasks=results,
        )
    except Exception as error:
        receipt.update(
            status="FAILED_ORIGINAL_GAIA_SCORING_RETAINED",
            error_type=type(error).__name__,
        )
        marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        raise
    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("data-root", "scorer-file", "predictions-file", "output"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_id": "GAIA",
                    "status": "PLAN_ONLY",
                    "original_split": "validation",
                    "author_revision": AUTHOR_REVISION,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    values = vars(args)
    values.pop("execute")
    if any(value is None for value in values.values()):
        parser.error(
            "execution requires data root, original scorer, saved predictions and fresh private output"
        )
    try:
        result = score(**values)
    except Exception as error:
        parser.exit(
            1,
            f"GAIA saved-output scoring failed ({type(error).__name__}); payload suppressed\n",
        )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "source_id",
                    "status",
                    "selected_original_validation_tasks",
                    "correct",
                    "accuracy",
                    "model_calls",
                    "source_checkbox_eligible",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
