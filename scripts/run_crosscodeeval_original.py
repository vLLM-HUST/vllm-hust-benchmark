"""Pinned author CrossCodeEval generation on one NPU, plan-only unless --action run.

Uses unmodified author cceval_generate, original tasks and original token budgets.
No service restart, dependency installation, CPU fallback or automatic retry.
"""

import argparse
from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

from prepare_crosscodeeval_original_assets import PINS, verify_git_blob
from prepare_xsum_original_assets import verify
from score_crosscodeeval_original import sha256, validate_predictions

MODEL_PINS = PINS.with_name("crosscodeeval-starcoder2-model.json")


def select_rows(archive, language, profile, start, count):
    """Retain raw selected lines for scoring, with no filesystem extraction."""
    with tarfile.open(archive, "r:xz") as bundle:
        matches = [
            member
            for member in bundle.getmembers()
            if member.name == f"{language}/{profile}.jsonl"
        ]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError("one original regular task member required")
        lines = []
        with bundle.extractfile(matches[0]) as stream:
            for index, line in enumerate(stream):
                if index < start:
                    continue
                lines.append(line)
                if count is not None and len(lines) == count:
                    break
    if not lines or (count is not None and len(lines) != count):
        raise ValueError("fixed sample extends beyond original member")
    rows = [json.loads(line) for line in lines]
    if any("pred" in row or "task_id" in row for row in rows):
        raise ValueError("prediction fields collide with original data")
    return lines, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action", choices=["plan", "run"], default="plan")
    parser.add_argument("--start", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=3)
    selection.add_argument("--full", action="store_true")
    parser.add_argument("--memory-utilization", type=float, default=0.16)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or (args.full and args.start != 0):
        parser.error("nonnegative start, positive count; full starts at zero")
    if not 0 < args.memory_utilization < 1:
        parser.error("memory utilization must be between zero and one")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    model_pins = json.loads(MODEL_PINS.read_text(encoding="utf-8"))
    defaults = pins["native_profile"]["author_api_defaults"]
    plan = {
        "source_id": "CROSSCODEEVAL",
        "status": "PLAN_ONLY_NO_ASSETS_ACQUIRED_OR_MODEL_CALLS",
        "model_repo": model_pins["repo_id"],
        "model_revision": model_pins["revision"],
        "languages": pins["native_profile"]["languages"],
        "profile": pins["native_profile"]["task"],
        "start": args.start,
        "count": None if args.full else args.count,
        "full": args.full,
        **defaults,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.memory_utilization,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "enforce_eager": True,
        "device": "one visible NPU, no CPU inference fallback",
    }
    if args.action == "plan":
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        raise ValueError("fresh output required; inspect existing run, never overwrite")
    for item in pins["author_files"]:
        verify_git_blob(args.author_root / item["path"], item)
    verify_git_blob(args.data_archive, pins["dataset"])
    for item in model_pins["files"]:
        verify(args.model_root / item["path"], item)
    config = json.loads((args.model_root / "config.json").read_text(encoding="utf-8"))
    if config.get("sliding_window") != 4096:
        raise ValueError("original model sliding window required")
    selected = {}
    for language in plan["languages"]:
        selected[language] = select_rows(
            args.data_archive, language, plan["profile"], args.start, plan["count"]
        )
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu" or torch.npu.device_count() != 1:
        raise ValueError(
            "exactly one visible NPU required; do not alter shared services"
        )
    sys.path.insert(0, str(args.author_root.resolve() / "scripts"))
    import vllm_inference as native

    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        **plan,
        "languages": [],
        "publisher_revision": pins["revision"],
        "archive_sha256": pins["dataset"]["sha256"],
        "status": "NATIVE_INITIALIZATION_PENDING",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "vllm": metadata.version("vllm"),
        "torch": metadata.version("torch"),
        "torch_npu": metadata.version("torch-npu"),
    }

    def save():
        (args.output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    engine = None
    try:
        engine = LLM(
            model=str(args.model_root.resolve()),
            tensor_parallel_size=1,
            max_model_len=defaults["model_max_tokens"],
            dtype="bfloat16",
            gpu_memory_utilization=args.memory_utilization,
            max_num_seqs=1,
            max_num_batched_tokens=2048,
            enforce_eager=True,
        )
        if engine.llm_engine.model_config.dtype != torch.bfloat16:
            raise ValueError("BF16 native model configuration required")
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_root.resolve()),
            local_files_only=True,
            trust_remote_code=False,
        )
        sampling = SamplingParams(
            temperature=defaults["temperature"],
            top_p=defaults["top_p"],
            max_tokens=defaults["generation_max_tokens"],
        )

        class CapturedEngine:
            def __init__(self, root):
                self.root = root

            def generate(self, prompts, params):
                (self.root / "actual-prompts.private.json").write_text(
                    json.dumps(prompts, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                responses = engine.generate(prompts, params)
                if len(responses) != len(prompts):
                    raise ValueError("native responses dropped original cases")
                terminal = []
                for response in responses:
                    if (
                        not response.finished
                        or len(response.outputs) != 1
                        or response.outputs[0].finish_reason not in ("stop", "length")
                    ):
                        raise ValueError(
                            "native request did not reach a valid terminal"
                        )
                    output = response.outputs[0]
                    terminal.append(
                        {
                            "finish_reason": output.finish_reason,
                            "prompt_tokens": len(response.prompt_token_ids),
                            "completion_tokens": len(output.token_ids),
                            "empty_output_retained": not output.text,
                        }
                    )
                (self.root / "terminal-private.json").write_text(
                    json.dumps(terminal, indent=2) + "\n", encoding="utf-8"
                )
                return responses

        for language, (lines, rows) in selected.items():
            root = args.output / language
            root.mkdir(mode=0o700)
            subset = root / "original.private.jsonl"
            subset.write_bytes(b"".join(lines))
            original = deepcopy(rows)
            native_args = SimpleNamespace(
                use_crossfile_context=True,
                **{
                    key: defaults[key]
                    for key in (
                        "crossfile_max_tokens",
                        "model_max_tokens",
                        "generation_max_tokens",
                    )
                },
                output_dir=str(root.resolve()),
            )
            receipt.update(status="NATIVE_AUTHOR_GENERATION_RUNNING")
            receipt["model_calls"] += len(rows)
            save()
            native.cceval_generate(
                native_args, rows, tokenizer, sampling, CapturedEngine(root)
            )
            prediction = root / "prediction.jsonl"
            validate_predictions(
                original,
                [json.loads(line) for line in prediction.read_bytes().splitlines()],
            )
            receipt["languages"].append(
                {
                    "language": language,
                    "original_positions": list(
                        range(args.start, args.start + len(rows))
                    ),
                    "completed": len(rows),
                    "predictions_sha256": sha256(prediction),
                    "original_subset_sha256": sha256(subset),
                    "subset_file": "original.private.jsonl",
                }
            )
            save()
        receipt["status"] = (
            "AUTHOR_NATIVE_FOUR_LANGUAGE_NPU_GENERATION_COMPLETE_SCORING_PENDING"
        )
        save()
    except Exception as error:
        receipt.update(
            status="NATIVE_FAILURE_NOT_REPLACED_OR_RETRIED",
            error_type=type(error).__name__,
            error=str(error),
        )
        save()
        raise
    finally:
        if engine is not None:
            del engine
    print(
        json.dumps(
            {key: receipt[key] for key in ("source_id", "status", "model_calls")}
        )
    )


if __name__ == "__main__":
    main()
