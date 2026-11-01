"""Run pinned official scoring with an explicitly named local NPU judge.

Model-provider transport and deterministic output-field binding are adaptations;
publisher prompts, token budgets, application retries and scoring stay unchanged.
"""

import argparse
from functools import partial
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from appworld_model_transport import model_proxy
from discoverybench_isolated_python import check_container
from run_discoverybench_original_actor import original_queries


SCORER_FILES = {
    "eval/new_eval.py": "83406ddbe6894429cc5012b17cb82a209d5397aa7863b0aba1bb51edf1dd6c65",  # pragma: allowlist secret - original official scorer
    "eval/lm_utils.py": "6f96bd2b5b44f84640c0b2d1314a0ed5012f7e59d09611a4eb946ac78c046706",  # pragma: allowlist secret - original official scorer
    "utils/openai_helpers.py": "7993d6d28e3734d6c9d5dbbdebed139a17c9d6451cefe0cd3f19dfc0630c8f20",  # pragma: allowlist secret - original official scorer
    "utils/openai_semantic_gen_prompts.py": "c5a6bea42ed8fbda329f36aa7de5ed6cd0a0035052c3e97e2cbda1086b689900",  # pragma: allowlist secret - original official scorer
    "utils/helpers.py": "4e50bd0cc5ab594c61796bb0833ae034260d6c2d09c4f015202ade5994483d41",  # pragma: allowlist secret - original official scorer
    "utils/arguments.py": "889a0cdcd58c67e40669cdef8e89bd66c6cefea3d99215a2b6bebb5323d5fe1d",  # pragma: allowlist secret - original official scorer
}


def verify_scorer(source):
    source = Path(source).resolve(strict=True)
    for name, expected in SCORER_FILES.items():
        path = (source / name).resolve(strict=True)
        if (
            not path.is_relative_to(source)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"unchanged original scorer required: {name}")
    return source


def prediction_fields(raw):
    if not isinstance(raw, str) or raw == "":
        raise ValueError(
            "nonempty original Agent output required; never invent an answer"
        )
    marker = "WORKFLOW SUMMARY:"
    if raw.count(marker) == 1:
        hypo, workflow = raw.split(marker, 1)
        if hypo.strip():
            return hypo, workflow, "exact_original_workflow_marker"
    return raw, None, "raw_original_output_no_unambiguous_workflow_marker"


def verified_json(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("original input file identity mismatch")
    return json.loads(raw), raw


def execute(args):
    check_container()
    source = verify_scorer(args.author_root)
    metadata, metadata_raw = verified_json(args.metadata, args.metadata_sha256)
    reference, reference_raw = verified_json(args.reference, args.reference_sha256)
    prediction, prediction_raw = verified_json(args.prediction, args.prediction_sha256)
    if (
        set(reference) != {"dataset", "metadataid", "query_id", "gold_hypo"}
        or not reference["gold_hypo"]
    ):
        raise ValueError("unchanged original real answer-key row required")
    if args.metadata.stem != "metadata_" + str(int(reference["metadataid"])):
        raise ValueError("original reference/metadata identity mismatch")
    matches = [
        q
        for q in original_queries(metadata["queries"])
        if int(q["qid"]) == int(reference["query_id"])
    ]
    if len(matches) != 1 or not isinstance(matches[0]["question"], str):
        raise ValueError("unique original question/reference identity required")
    if not isinstance(prediction, list) or len(prediction) != 1:
        raise ValueError("one saved original Agent result required")
    hypo, workflow, binding = prediction_fields(prediction[0]["output"])
    if not args.model_queue.is_dir() or any(args.model_queue.iterdir()):
        raise ValueError(
            "fresh private judge queue required; never restart existing inference"
        )
    args.output.mkdir(mode=0o700, exist_ok=False)
    sys.path.insert(0, str(source))
    os.environ["OPENAI_API_KEY"] = (
        "local-queue-no-secret"  # pragma: allowlist secret - noncredential placeholder for owned local queue
    )
    scorer = importlib.import_module("eval.new_eval")
    lm = importlib.import_module("eval.lm_utils")
    # Disable SDK transport retries only; publisher application retries remain.
    scorer.OpenAI = partial(
        scorer.OpenAI, max_retries=0, timeout=args.transport_timeout
    )
    lm.OpenAI = partial(lm.OpenAI, max_retries=0, timeout=args.transport_timeout)
    with model_proxy(args.model_queue, port=0, timeout=args.transport_timeout) as port:
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        result = scorer.run_eval_gold_vs_gen_NL_hypo_workflow(
            query=matches[0]["question"],
            gold_hypo=reference["gold_hypo"],
            gold_workflow=None,
            gen_hypo=hypo,
            gen_workflow=workflow,
            dataset_meta=metadata,
            llm_used=args.model,
            dataset_type="real",
            use_column_metadata=True,
        )
    for path, raw in (
        (args.metadata, metadata_raw),
        (args.reference, reference_raw),
        (args.prediction, prediction_raw),
    ):
        if path.read_bytes() != raw:
            raise ValueError("original scoring input changed")
    verify_scorer(source)
    (args.output / "official-evaluation-private.json").write_text(
        json.dumps(result, indent=2)
    )
    receipt = {
        "source_id": "DISCOVERYBENCH",
        "status": "ORIGINAL_SCORER_RETURNED_ALTERNATE_JUDGE_AUDIT_PENDING",
        "requested_judge": args.model,
        "publisher_cli_default_judge": "gpt-4-1106-preview",
        "official_gpt4_score": False,
        "prediction_field_binding": binding,
        "original_prompt_and_scoring_source_unchanged": True,
        "original_gold_workflow_default": None,
        "original_metadata_type": "real",
        "source_checkbox_eligible": False,
    }
    (args.output / "judge-receipt-public.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(receipt))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "author-root",
        "metadata",
        "reference",
        "prediction",
        "model-queue",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    for name in ("metadata", "reference", "prediction"):
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--transport-timeout", type=float, default=3600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"status": "PLAN_ONLY", "model_calls": 0, "official_gpt4_score": False}
            )
        )
        return 0
    required = (
        "author_root",
        "metadata",
        "reference",
        "prediction",
        "metadata_sha256",
        "reference_sha256",
        "prediction_sha256",
        "model_queue",
        "output",
    )
    if (
        any(getattr(args, key) is None for key in required)
        or args.transport_timeout <= 0
    ):
        parser.error(
            "explicit original input identities, private queue/output and positive timeout required"
        )
    os.umask(0o077)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
