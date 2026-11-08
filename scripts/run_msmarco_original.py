"""Run original complete MS MARCO candidate pools on NPU, or rescore saved ranks."""

import argparse
from importlib import metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_xsum_original_assets import sha, verify


ROOT = Path(__file__).resolve().parents[1]
SCORER = {
    "bytes": 7465,
    "sha256": "3fc557384a697b59d533e43a521cf75974239d12855999fd095904476bc37ab6",
}  # pragma: allowlist secret - original public Microsoft evaluator checksum


def rank_candidates(candidates, scores):
    if len(candidates) != len(scores) or not candidates:
        raise ValueError("one native score per original candidate required")
    if len({item["passage_id"] for item in candidates}) != len(candidates):
        raise ValueError(
            "duplicate original passage IDs need explicit native scorer review"
        )
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        for score in scores
    ):
        raise ValueError("finite native cross-encoder scores required")
    return [
        candidates[index]["passage_id"]
        for index in sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )
    ]


def read_rankings(path, records):
    result = {record.task["task_id"]: [] for record in records}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) != 3
                or fields[0] not in result
                or fields[2] != str(len(result[fields[0]]) + 1)
            ):
                raise ValueError("complete sequential original query rankings required")
            result[fields[0]].append(fields[1])
    for record in records:
        ids = [item["passage_id"] for item in record.task["input"]["candidates"]]
        ranked = result[record.task["task_id"]]
        if (
            len(ranked) != len(ids)
            or len(set(ranked)) != len(ranked)
            or set(ranked) != set(ids)
        ):
            raise ValueError(
                "saved ranking must contain the complete original candidate pool"
            )
    return result


def score_original(author_root, output, records, rankings):
    scorer = author_root / "ms_marco_eval.py"
    verify(scorer, SCORER)
    references = output / "selected-original-qrels.private.tsv"
    with references.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            for row in record.outcome["references"]["original_qrel_rows"]:
                stream.write("\t".join(row) + "\n")
    # Subset the original qrel rows to selected queries, not their candidate pool.
    # Preserve relevant IDs even when they are absent from the original candidates.
    result = subprocess.run(
        [
            sys.executable,
            str(scorer),
            str(references),
            str(output / "rankings.private.tsv"),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    (output / "official-score.private.log").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    official = float(
        next(
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.startswith("MRR @10:")
        )
    )
    reciprocal = []
    for record in records:
        relevant = {
            row[2] for row in record.outcome["references"]["original_qrel_rows"]
        }
        if relevant:
            reciprocal.append(
                next(
                    (
                        1 / rank
                        for rank, pid in enumerate(
                            rankings[record.task["task_id"]][:10], 1
                        )
                        if pid in relevant
                    ),
                    0.0,
                )
            )
    if (
        not reciprocal
        or not math.isfinite(official)
        or not math.isclose(
            official, sum(reciprocal) / len(reciprocal), rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "original Microsoft MRR and independent saved-rank audit differ"
        )
    verify(scorer, SCORER)
    return {
        "official_mrr_at10": official,
        "independently_recomputed_reciprocal_ranks": reciprocal,
        "queries_with_original_qrels": len(reciprocal),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "score"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=3)
    selection.add_argument("--all-queries", action="store_true")
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args(argv)
    if args.limit <= 0 or args.output.exists():
        parser.error("positive query limit and fresh output directory required")
    if args.mode == "run" and (
        args.model_root is None
        or args.rankings is not None
        or not args.device.startswith("npu:")
        or not args.device[4:].isdigit()
    ):
        parser.error(
            "run requires model-root and explicit npu:<index>, without saved rankings"
        )
    if args.mode == "score" and (args.rankings is None or args.model_root is not None):
        parser.error("score requires saved rankings, without loading a model")
    from vllm_hust_workload_data import iter_workload

    verify(args.author_root / "ms_marco_eval.py", SCORER)
    records = list(
        iter_workload(
            "MSMARCO_PASSAGE",
            data_root=args.data_root,
            limit=None if args.all_queries else args.limit,
        )
    )
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": "MSMARCO_PASSAGE",
        "status": "RUNNING",
        "mode": args.mode,
        "original_query_ids": [r.task["task_id"] for r in records],
        "complete_candidate_counts": [
            len(r.task["input"]["candidates"]) for r in records
        ],
        "source_identity": records[0].source,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        if args.mode == "score":
            rankings = read_rankings(args.rankings, records)
            (args.output / "rankings.private.tsv").write_bytes(
                args.rankings.read_bytes()
            )
        else:
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            pins = json.loads(
                (ROOT / "configs/workload-native/msmarco-reranker-pins.json").read_text(
                    "utf-8"
                )
            )["model"]
            for item in pins["files"]:
                verify(args.model_root / item["path"], item)
            if (
                metadata.version("sentence-transformers") != "5.1.2"
                or metadata.version("transformers") != "4.57.3"
            ):
                raise ValueError(
                    "reuse the validated sentence-transformers5.1.2/transformers4.57.3 NPU client"
                )
            import torch
            import torch_npu  # noqa: F401 - register the authorized device backend
            from sentence_transformers import CrossEncoder

            target = torch.device(args.device)
            torch.npu.set_device(target)
            if torch.npu.mem_get_info()[0] < 2 * 1024**3:
                raise ValueError(
                    "insufficient free NPU memory; leave other services untouched"
                )
            model = CrossEncoder(
                str(args.model_root),
                device=args.device,
                max_length=512,
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
            )
            if any(p.device != target for p in model.model.parameters()):
                raise ValueError(
                    "cross-encoder parameters must remain on the selected NPU"
                )
            rankings = {}
            with (args.output / "scores.private.jsonl").open(
                "x", encoding="utf-8"
            ) as saved:
                for record in records:
                    inputs = record.task["input"]
                    scores = model.predict(
                        [(inputs["query"], c["passage"]) for c in inputs["candidates"]],
                        batch_size=128,
                        show_progress_bar=False,
                        convert_to_numpy=False,
                        convert_to_tensor=True,
                    )
                    if (
                        scores.device != target
                        or not torch.isfinite(scores).all().item()
                    ):
                        raise ValueError(
                            "finite native NPU cross-encoder output required"
                        )
                    values = scores.flatten().tolist()
                    rankings[record.task["task_id"]] = rank_candidates(
                        inputs["candidates"], values
                    )
                    saved.write(
                        json.dumps(
                            {
                                "query_id": record.task["task_id"],
                                "candidate_ids": [
                                    c["passage_id"] for c in inputs["candidates"]
                                ],
                                "scores": values,
                            }
                        )
                        + "\n"
                    )
                    saved.flush()
                    receipt["model_calls"] += 1
                    save()
            receipt.update(
                model=pins["repo_id"],
                model_revision=pins["revision"],
                device=args.device,
                dtype="float32",
                native_model_max_length=512,
                batch_size=128,
            )
            with (args.output / "rankings.private.tsv").open(
                "x", encoding="utf-8", newline="\n"
            ) as stream:
                for qid, ranked in rankings.items():
                    for rank, pid in enumerate(ranked, 1):
                        stream.write(f"{qid}\t{pid}\t{rank}\n")
        validated = read_rankings(args.output / "rankings.private.tsv", records)
        receipt.update(
            score_original(args.author_root, args.output, records, validated)
        )
        receipt.update(
            status="ORIGINAL_COMPLETE_POOLS_RANKED_AND_MICROSOFT_MRR_VERIFIED"
            if args.mode == "run"
            else "SAVED_ORIGINAL_RANKINGS_RESCORED_NO_MODEL_CALLS",
            rankings_sha256=sha(args.output / "rankings.private.tsv"),
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
