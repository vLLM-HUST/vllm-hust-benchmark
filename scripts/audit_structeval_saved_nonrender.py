"""Reuse terminal StructEval nonrender outputs and official CPU scoring; no inference.

Original task files remain external. Public summary contains counts and file
hashes only. Rendered/VQA tasks require their own workflow and are rejected.
"""

import argparse
import contextlib
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import zipfile

DATA_SHA = "66704021b99dd10b242e7a74c44f6a222c2fa6779edbb0476ec8cf5e18132fd8"  # pragma: allowlist secret - original public asset checksum
ARCHIVE_SHA = "43593f97711c77c15c4e3711fe94bea188c58e718535893c0640fa17095360bb"  # pragma: allowlist secret - fixed official source archive checksum
SCORE_KEYS = ("render_score", "key_validation_score", "final_eval_score")


def read_json(path):
    return json.loads(Path(path).read_text("utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_author(archive_path, package_root):
    if digest(archive_path) != ARCHIVE_SHA:
        raise ValueError("fixed original source archive required")
    package_root = Path(package_root).resolve(strict=True)
    expected = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if "/structeval/" in name and name.endswith(".py"):
                relative = name.split("/structeval/", 1)[1]
                path = (package_root / relative).resolve(strict=True)
                if not path.is_relative_to(
                    package_root
                ) or path.read_bytes() != archive.read(name):
                    raise ValueError("installed official source identity mismatch")
                expected[relative] = digest(path)
    actual = {
        p.relative_to(package_root).as_posix() for p in package_root.rglob("*.py")
    }
    if len(expected) != 27 or set(expected) != actual:
        raise ValueError("complete fixed 27-module official source required")
    return expected


def audit_runs(
    rows, run_dirs, work_root, evaluate_nonrenderable, calculate_final_score
):
    work_root = Path(work_root).resolve()
    run_dirs = [Path(p).resolve(strict=True) for p in run_dirs]
    if not run_dirs or len(run_dirs) != len(set(run_dirs)):
        raise ValueError("distinct saved run directories required")
    if any(
        work_root.is_relative_to(p) or p.is_relative_to(work_root) for p in run_dirs
    ):
        raise ValueError("fresh audit output must not overlap saved runs")
    work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    seen, protocols, files = set(), {}, []
    for ordinal, folder in enumerate(run_dirs):
        receipt = read_json(folder / "receipt.json")
        if (
            receipt.get("source_id") != "STRUCTEVAL"
            or receipt.get("status") != "END_TO_END_EXECUTED_AND_SCORED"
            or receipt.get("max_tokens") is not None
            or "max_tokens" not in receipt
            or receipt.get("source_revision")
            != "936be10548688c6d417cd435dd906158f19ba14b"  # pragma: allowlist secret - original dataset revision
            or receipt.get("runner_revision")
            != "7781339a5b033965ac664133543ebd10773643dd"  # pragma: allowlist secret - official runner revision
        ):
            raise ValueError("terminal original workflow receipt required")
        indices = receipt.get("original_row_indices", [])
        if (
            not indices
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("valid original row positions required")
        if seen.intersection(indices):
            raise ValueError("overlapping original tasks; do not double count reruns")
        originals = [rows[i] for i in indices]
        if any(r["rendering"] for r in originals):
            raise ValueError("rendered/VQA workflow cannot use nonrender scoring")
        selected = read_json(folder / "original-selected.private.json")
        inference_path = folder / "official-workflow/inference.json"
        evaluation_path = folder / "official-workflow/evaluation.json"
        inference, evaluation = read_json(inference_path), read_json(evaluation_path)
        ids = [r["task_id"] for r in originals]
        if (
            selected != originals
            or receipt.get("selected_samples") != ids
            or receipt.get("unique_tasks") != len(indices)
            or len(inference) != len(evaluation)
            or len(inference) != len(indices)
        ):
            raise ValueError("original task coverage/fields changed")
        for original, generated, scored in zip(originals, inference, evaluation):
            if (
                any(
                    generated.get(k) != v or scored.get(k) != v
                    for k, v in original.items()
                )
                or "generation" not in generated
                or generated["generation"] != scored.get("generation")
            ):
                raise ValueError("saved original fields or generation changed")
        output = work_root / f"run-{ordinal:04d}"
        output.mkdir(mode=0o700)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            rescored = evaluate_nonrenderable(
                deepcopy(inference), str(output / "formats")
            )
            for actual in rescored:
                calculate_final_score(actual)
        (output / "rescore-private.log").write_text(stream.getvalue(), encoding="utf-8")
        if len(rescored) != len(evaluation) or any(
            actual.get("task_id") != saved["task_id"]
            or any(
                k not in actual or k not in saved or actual[k] != saved[k]
                for k in SCORE_KEYS
            )
            for actual, saved in zip(rescored, evaluation)
        ):
            raise ValueError("official saved-output rescore mismatch")
        seen.update(indices)
        for original in originals:
            key = original["input_type"] + "-to-" + original["output_type"]
            protocols[key] = protocols.get(key, 0) + 1
        files.append(
            {
                "receipt_sha256": digest(folder / "receipt.json"),
                "inference_sha256": digest(inference_path),
                "evaluation_sha256": digest(evaluation_path),
            }
        )
    return {
        "source_id": "STRUCTEVAL",
        "status": "SAVED_NONRENDER_ORIGINALS_OFFICIALLY_RESCORED",
        "unique_original_tasks": len(seen),
        "protocol_task_counts": protocols,
        "full_original_source_rows": len(rows),
        "original_nonrender_rows": sum(not r["rendering"] for r in rows),
        "new_model_calls": 0,
        "failed_tasks_replaced": False,
        "original_inputs_or_generations_modified": False,
        "source_checkbox_eligible": False,
        "boundary": "Saved-output CPU scoring reuse only; rendered/VQA protocols and final consumer delivery remain separate.",
        "observed_saved_file_hashes": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.data_file) != DATA_SHA:
        raise ValueError("original full dataset identity required")
    spec = importlib.util.find_spec("structeval")
    if spec is None or len(list(spec.submodule_search_locations or [])) != 1:
        raise RuntimeError("fixed official StructEval installation required")
    verify_author(args.official_archive, next(iter(spec.submodule_search_locations)))
    os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP="True", LITELLM_TELEMETRY="False")
    import pyarrow.parquet as pq
    from structeval.eval_engine.eval_nonrenderable import evaluate_nonrenderable
    from structeval.eval_engine.main import calculate_final_score

    rows = pq.read_table(args.data_file).to_pylist()
    if len(rows) != 2035 or digest(args.data_file) != DATA_SHA:
        raise ValueError("original full2035 dataset changed")
    for source in (args.data_file.parent, args.official_archive.parent):
        work = args.work_root.resolve()
        if work.is_relative_to(source.resolve()) or source.resolve().is_relative_to(
            work
        ):
            raise ValueError("audit output must not overlap original asset roots")
    result = audit_runs(
        rows,
        args.run_dir,
        args.work_root,
        evaluate_nonrenderable,
        calculate_final_score,
    )
    (args.work_root / "receipt-public.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
