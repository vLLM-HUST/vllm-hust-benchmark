"""Original BEIR full-test retrieval and top-100 reranking, or offline rescoring.

No chat prompt conversion, gold-derived candidate pool, CPU model fallback or
research observation. Offline scoring does not establish model execution.
"""

import argparse
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import time


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/beir-author-pins.json"
)
PROTOCOL_NAMES = ("dense_retrieval", "dense_top100_cross_encoder_reranking")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root, files):
    root = Path(root).resolve()
    for spec in files:
        path = (root / spec["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("pinned file escapes explicit root")
        if path.stat().st_size != spec["bytes"] or sha(path) != spec["sha256"]:
            raise ValueError("original pinned file identity mismatch")


def validate_rankings(results, *, queries, corpus, dense=None, top_k=100):
    if not isinstance(results, dict) or set(results) != set(queries):
        raise ValueError("every original test query must have results")
    for query_id, scores in results.items():
        if not isinstance(scores, dict) or not scores:
            raise ValueError("nonempty original retrieval ranking required")
        if any(document_id not in corpus for document_id in scores):
            raise ValueError("ranking references unknown original document")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            for value in scores.values()
        ):
            raise ValueError("finite original retrieval scores required")
        if dense is None and (
            query_id in scores
            or len(scores) != min(1000, len(corpus) - int(query_id in corpus))
        ):
            raise ValueError("complete original dense top-1000 depth required")
        if dense is not None:
            # Match the original Rerank implementation's stable descending sort.
            expected = sorted(dense[query_id], key=dense[query_id].get, reverse=True)[
                :top_k
            ]
            if set(scores) != set(expected):
                raise ValueError("reranking must use exactly the original dense top-k")


def read_rankings(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate original ranking key")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs
    )


def score_original(evaluator, qrels, rankings):
    # BEIR's evaluator removes self-matches in place. Preserve saved rankings.
    values = evaluator.evaluate(deepcopy(qrels), deepcopy(rankings), evaluator.k_values)
    return {
        "metrics": dict(
            zip(("ndcg", "map", "recall", "precision"), values, strict=True)
        ),
        "mrr": evaluator.evaluate_custom(
            deepcopy(qrels), deepcopy(rankings), evaluator.k_values, metric="mrr"
        ),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-id", required=True, choices=["BEIR_SCIFACT", "BEIR_NFCORPUS"]
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["run", "score"], default="run")
    parser.add_argument("--encoder-root", type=Path)
    parser.add_argument("--cross-encoder-root", type=Path)
    parser.add_argument("--dense-results", type=Path)
    parser.add_argument("--rerank-results", type=Path)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args(argv)
    if args.mode == "run":
        if not args.encoder_root or not args.cross_encoder_root:
            parser.error("run requires both pinned original model directories")
        if not args.device.startswith("npu:") or not args.device[4:].isdigit():
            parser.error("model execution requires an explicit npu:<index>")
        if args.dense_results or args.rerank_results:
            parser.error(
                "existing results belong to score mode, never silently skip generation"
            )
    elif not args.dense_results or not args.rerank_results:
        parser.error("score requires both complete original ranking files")
    return args


def main(argv=None):
    args = parse_args(argv)
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    protocol = pins["protocol"]
    if args.output.exists():
        raise ValueError("fresh private output directory required")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from vllm_hust_workload_data.beir import validate_original_relations

    relations = validate_original_relations(
        source_id=args.source_id, data_root=args.data_root
    )
    if metadata.version("beir") != pins["framework_version"]:
        raise ValueError("pinned original BEIR version required")
    distribution = metadata.distribution("beir")
    core_files = [f for f in pins["source_files"] if f["path"].startswith("beir/")]
    verify_files(distribution.locate_file(""), core_files)
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval

    corpus, queries, qrels = GenericDataLoader(data_folder=str(args.data_root)).load(
        split="test"
    )
    if not corpus or set(queries) != set(qrels):
        raise ValueError("original full test loader relation mismatch")
    raw_queries = {
        row["_id"]: row
        for row in map(
            json.loads,
            (args.data_root / "queries.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    raw_corpus = {
        row["_id"]: row
        for row in map(
            json.loads,
            (args.data_root / "corpus.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    if (
        set(corpus) != set(raw_corpus)
        or any(
            corpus[key]["text"] != row["text"] or corpus[key]["title"] != row["title"]
            for key, row in raw_corpus.items()
        )
        or any(text != raw_queries[key]["text"] for key, text in queries.items())
    ):
        raise ValueError("original loader changed native query or corpus text")
    args.output.mkdir(parents=True, mode=0o700)
    receipt = {
        "source_id": args.source_id,
        "mode": args.mode,
        "status": "RUNNING",
        "protocol": protocol,
        "framework_revision": pins["framework_revision"],
        "model_bindings": [
            {"repo_id": m["repo_id"], "revision": m["revision"]} for m in pins["models"]
        ],
        "original_assets": relations,
        "original_test_queries": len(queries),
        "original_corpus_documents": len(corpus),
        "generation_performed_this_run": args.mode == "run",
        "device": args.device if args.mode == "run" else None,
        "protocols": [],
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "started_at_unix": time.time(),
    }

    def save():
        (args.output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        evaluator = EvaluateRetrieval()
        if args.mode == "score":
            dense, reranked = (
                read_rankings(args.dense_results),
                read_rankings(args.rerank_results),
            )
            receipt["input_ranking_sha256"] = {
                "dense": sha(args.dense_results),
                "rerank": sha(args.rerank_results),
            }
        else:
            import torch
            import torch_npu  # noqa: F401 - registers the authorized NPU backend
            from beir.retrieval import models
            from beir.retrieval.search.dense import DenseRetrievalExactSearch
            from beir.reranking import Rerank
            from beir.reranking.models import CrossEncoder

            for name, version in {
                "sentence-transformers": "5.1.2",
                "transformers": "4.57.3",
            }.items():
                if metadata.version(name) != version:
                    raise ValueError("validated model-client version required")
            verify_files(args.encoder_root, pins["models"][0]["files"])
            verify_files(args.cross_encoder_root, pins["models"][1]["files"])
            target = torch.device(args.device)
            torch.npu.set_device(target)
            if torch.npu.mem_get_info()[0] < 6 * 1024**3:
                raise ValueError(
                    "insufficient free NPU memory; do not stop others' services"
                )

            class NativeEncoder(models.SentenceBERT):
                def encode_queries(self, queries, **kwargs):
                    result = super().encode_queries(queries, **kwargs)
                    check_tensor(result)
                    return result

                def encode_corpus(self, corpus, **kwargs):
                    result = super().encode_corpus(corpus, **kwargs)
                    check_tensor(result)
                    return result

            def check_tensor(tensor):
                if tensor.device != target or not torch.isfinite(tensor).all().item():
                    raise ValueError("finite native NPU embedding required")

            def check_parameters(model):
                if not all(
                    parameter.device == target for parameter in model.parameters()
                ):
                    raise ValueError(
                        "all model parameters must remain on the selected NPU"
                    )

            model_options = {
                "device": args.device,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_kwargs": {
                    "dtype": torch.float32,
                    "attn_implementation": "eager",
                },
            }
            encoder = NativeEncoder(str(args.encoder_root), **model_options)
            check_parameters(encoder.q_model)
            if encoder.q_model.max_seq_length != protocol["encoder_native_max_length"]:
                raise ValueError("original encoder-native context policy changed")
            search = DenseRetrievalExactSearch(
                encoder,
                batch_size=protocol["dense_batch_size"],
                corpus_chunk_size=protocol["dense_corpus_chunk_size"],
                show_progress_bar=False,
                convert_to_tensor=True,
            )
            retriever = EvaluateRetrieval(
                search, score_function=protocol["dense_score_function"]
            )
            dense = retriever.retrieve(corpus, queries)
            cross = CrossEncoder(str(args.cross_encoder_root), **model_options)
            check_parameters(cross.model.model)
            reranked = Rerank(
                cross, batch_size=protocol["cross_encoder_batch_size"]
            ).rerank(corpus, queries, deepcopy(dense), top_k=protocol["rerank_top_k"])
            for name, results in (("dense", dense), ("rerank", reranked)):
                path = args.output / f"{name}-results.private.json"
                path.write_text(json.dumps(results) + "\n", encoding="utf-8")
            receipt["model_parameters_and_embeddings_verified_on_npu"] = True
        validate_rankings(dense, queries=queries, corpus=corpus)
        validate_rankings(
            reranked,
            queries=queries,
            corpus=corpus,
            dense=dense,
            top_k=protocol["rerank_top_k"],
        )
        for name, rankings in zip(PROTOCOL_NAMES, (dense, reranked), strict=True):
            receipt["protocols"].append(
                {"protocol": name, **score_original(evaluator, qrels, rankings)}
            )
        receipt["status"] = (
            "ORIGINAL_FULL_TEST_EXECUTED_AND_SCORED"
            if args.mode == "run"
            else "ORIGINAL_FULL_TEST_OFFLINE_RESCORED_NOT_NEW_GENERATION"
        )
    except Exception as exc:
        receipt.update(
            status="FAILED_NO_TASK_OR_MODEL_REPLACEMENT", error_type=type(exc).__name__
        )
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
