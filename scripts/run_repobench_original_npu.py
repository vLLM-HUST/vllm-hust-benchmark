"""Original RepoBench next-line completion with an explicit local NPU port.

Uses unchanged author filters, prompt construction, first-code-line extraction
and eval.py scoring. Model assets/data stay external. No chat conversion, tool
emulation or execution of generated code. Default prints a plan only.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

from prepare_repobench_original_assets import REVISION, verify_original

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
MODEL_REVISION = "c919139c3a9b4070729c8b2cca4847ab29ca8d94"  # pragma: allowlist secret - public model revision
MODEL_FILES = {
    "LICENSE": (
        13759,
        "ccfee4895df06bcab524151c278e8dde88bbe76165a24ecbcbcf9fafd71fd2b3",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "config.json": (
        631,
        "aae6fb3378d48735b5f1f7f0046b57e7fd7735ca049ac3fad29ce2c2f37ed1e8",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "generation_config.json": (
        119,
        "aee1459302885cf8614f024bd29adae66b518baf640dea6ed8abb2a39aece832",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "pytorch_model.bin": (
        2693014838,
        "a400155e495b4d6b1cd5b05f04c56032b7dbc6bdff940e046a89ffc1cc2fbdcd",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "special_tokens_map.json": (
        482,
        "acb949a6d246354249847f5e1f0dabe04c258271eab55a6d6fac251d7af41075",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "tokenizer.json": (
        1367962,
        "ef48ebdc8546c2d8092349d321f1d162de804a1c8900df2b615c7dc8b02ce141",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
    "tokenizer_config.json": (
        793,
        "c185a68412e595b34f9a32a2ffd68b726a1286b0936a2b616e7d1a236197fc48",  # pragma: allowlist secret - public file checksum
    ),  # pragma: allowlist secret - original public checksum
}
LEVELS = ["2k", "4k", "8k", "12k", "16k"]
SETTINGS = ("cross_file_first", "cross_file_random", "in_file")
INPUT_FIELDS = ("repo_name", "file_path", "context", "import_statement", "cropped_code")
MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "workload-preparation/data/catalogs/repobench_source_files_v1.json"
)


def verify_files(root, specs):
    root = Path(root).resolve()
    for name, (size, checksum) in specs.items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or path.stat().st_size != size
        ):
            raise ValueError("original asset path or size mismatch")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                raise ValueError("original asset checksum mismatch")


def select_rows(rows, per_stratum):
    if per_stratum is not None and (type(per_stratum) is not int or per_stratum <= 0):
        raise ValueError("positive fixed sample count required")
    counts = defaultdict(int)
    selected = []
    for index, row in enumerate(rows):
        key = row["level"]
        if per_stratum is None or counts[key] < per_stratum:
            selected.append((index, row))
            counts[key] += 1
    return selected


def original_module(source, name):
    spec = importlib.util.spec_from_file_location(
        f"repobench_original_{name}", source / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_original(source, results, language):
    """Author scorer unchanged; freeze Python hash ordering in a fresh process.

    CodeBLEU0.7 dataflow matching was empirically hash-seed-sensitive. Seed0 is
    a declared reproducibility configuration, not a search for a better score.
    Setting os.environ inside the existing interpreter would not fix its hash.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        [
            sys.executable,
            str(source / "eval.py"),
            "--path",
            str(results),
            "--language",
            language,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        completed.check_returncode()
    return completed.stdout


def prepare(args, author, tokenizer):
    from datasets import DatasetDict, load_dataset

    manifest = json.loads(MANIFEST.read_text())
    if manifest["source_id"] != "REPOBENCH" or manifest["revision"] != REVISION:
        raise ValueError("pinned original dataset manifest required")
    specs = manifest["files"]
    verify_files(
        args.data_root,
        {item["path"]: (item["bytes"], item["sha256"]) for item in specs},
    )
    selected = []
    for language in args.languages:
        files = {}
        for setting in SETTINGS:
            files[setting] = [
                str(args.data_root / item["path"])
                for item in specs
                if item["language"] == language
                and Path(item["path"]).name.startswith(setting + "-")
            ]
        original = load_dataset(
            "parquet",
            data_files=files,
            cache_dir=str(args.output / "local-arrow-cache"),
        )
        filtered = author.filter_dataset_by_date_range(
            original, "2023-12-01", "2023-12-31"
        )
        filtered = author.filter_dataset_by_levels(DatasetDict(filtered), LEVELS)
        for setting in SETTINGS:
            rows = select_rows(filtered[setting], args.per_stratum)
            if not rows:
                raise ValueError("empty original subset; do not silently replace tasks")
            for index, row in rows:
                # Ground-truth/reference fields never reach the prompt builder.
                prompt = author.construct_prompt(
                    {key: row[key] for key in INPUT_FIELDS},
                    tokenizer=tokenizer,
                    max_token_nums=15800,
                    language=language,
                )
                selected.append(
                    {
                        "language": language,
                        "setting": setting,
                        "filtered_index": index,
                        "original": row,
                        "prompt": prompt,
                        "prompt_tokens": len(tokenizer.encode(prompt)),
                    }
                )
    return selected


def generate_one(torch, model, tokenizer, prompt, device):
    inputs = tokenizer([prompt], return_tensors="pt", padding=True).to(device)
    if inputs["input_ids"].device.type != "npu":
        raise ValueError("input tensor is not on NPU; no silent fallback")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.2, top_p=0.95, do_sample=True
        )
    if outputs.device.type != "npu":
        raise ValueError("generated tensor is not on NPU")
    tokens = outputs[0][inputs["input_ids"][0].shape[-1] :].tolist()
    return tokens, tokenizer.decode(tokens, skip_special_tokens=True)


def execute(args):
    if args.execute and not args.resource_window_confirmed:
        raise ValueError("explicit idle research/NPU window confirmation required")
    verify_original(args.source_root, args.archive)
    verify_files(args.model_root, MODEL_FILES)
    repository = Path(__file__).resolve().parents[1]
    protected = [args.source_root, args.data_root, args.model_root]
    if (repository / ".git").exists():
        protected.append(repository)
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in protected):
        raise ValueError(
            "fresh output must remain outside source/data/model/repository"
        )
    if args.output.exists():
        raise ValueError(
            "fresh output required; preserve existing attempts and do not rerun blindly"
        )
    args.output.mkdir(parents=True, mode=0o700)
    sys.path.insert(0, str(args.source_root.resolve()))
    author = original_module(args.source_root, "run")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_root), local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tasks = prepare(args, author, tokenizer)
    (args.output / "selection.private.json").write_text(json.dumps(tasks, default=str))
    receipt = {
        "source_id": "REPOBENCH",
        "code_revision": REVISION,
        "dataset_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "status": "ORIGINAL_INPUTS_PREPARED_NOT_NATIVE_EXECUTION",
        "selected_tasks": len(tasks),
        "selection": "FirstN per original language/setting/length stratum after unchanged author default date and level filters",
        "per_stratum": args.per_stratum,
        "levels": LEVELS,
        "date_range": ["2023-12-01", "2023-12-31"],
        "model_calls": 0,
        "dtype": args.dtype,
        "device": args.npu_device,
        "prompt_budget": 15800,
        "max_new_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.95,
        "do_sample": True,
        "attention_implementation": "eager",
        "source_checkbox_eligible": False,
        "author_model_quality_reproduced": False,
        "scoring_python_hash_seed": 0,
    }
    path = args.output / "receipt-public.json"
    path.write_text(json.dumps(receipt, indent=2))
    if not args.execute:
        return receipt
    try:
        import torch
        import torch_npu  # noqa: F401 - explicit NPU registration, never CPU/CUDA fallback
        from transformers import AutoModelForCausalLM

        if not torch.npu.is_available():
            raise ValueError("NPU required; no CPU/CUDA fallback")
        torch.npu.set_device(args.npu_device)
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(args.model_root),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=False,
                dtype=getattr(torch, args.dtype),
                attn_implementation="eager",
            )
            .to(args.npu_device)
            .eval()
        )
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if next(model.parameters()).device.type != "npu":
            raise ValueError("model parameters are not on NPU")
        receipt["verified_parameter_device_type"] = "npu"
        receipt["npu_device_name"] = torch.npu.get_device_name(args.npu_device)
        receipt["status"] = "NPU_ORIGINAL_GENERATION_IN_PROGRESS_NOT_DELIVERED"
        path.write_text(json.dumps(receipt, indent=2))
        for index, task in enumerate(tasks):
            attempt = {
                "task_index": index,
                "language": task["language"],
                "setting": task["setting"],
                "filtered_index": task["filtered_index"],
                "status": "STARTED_UNKNOWN_OUTCOME",
            }
            saved = args.output / f"attempt-{index:05d}.private.json"
            saved.write_text(json.dumps(attempt))
            receipt["model_calls"] += 1
            path.write_text(json.dumps(receipt, indent=2))
            tokens, raw = generate_one(
                torch, model, tokenizer, task["prompt"], args.npu_device
            )
            pred = author.get_first_line_not_comment(raw, task["language"])
            attempt.update(
                status="GENERATED_ORIGINAL_NEXT_LINE",
                token_ids=tokens,
                raw_output=raw,
                pred=pred,
            )
            saved.write_text(json.dumps(attempt))
            result_dir = args.output / task["language"]
            result_dir.mkdir(exist_ok=True)
            with (result_dir / (task["setting"] + ".jsonl")).open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "idx": task["filtered_index"],
                            "level": task["original"]["level"],
                            "pred": pred,
                            "gt": task["original"]["next_line"],
                        }
                    )
                    + "\n"
                )
            path.write_text(json.dumps(receipt, indent=2))
        for language in args.languages:
            (args.output / (language + "-official-scoring.txt")).write_text(
                score_original(args.source_root, args.output / language, language)
            )
        verify_original(args.source_root, args.archive)
        receipt["status"] = (
            "ORIGINAL_GENERATION_AND_AUTHOR_SCORING_COMPLETED_PENDING_SAVED_AUDIT"
        )
    except Exception as exc:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__)
        raise
    finally:
        path.write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "archive", "data-root", "model-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--languages", nargs="+", choices=["python", "java"], default=["python", "java"]
    )
    parser.add_argument("--npu-device", default="npu:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--resource-window-confirmed", action="store_true")
    sample = parser.add_mutually_exclusive_group(required=True)
    sample.add_argument("--per-stratum", type=int)
    sample.add_argument("--full", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.npu_device.startswith("npu:") or not args.npu_device[4:].isdigit():
        parser.error("explicit npu:<ordinal> required")
    if len(set(args.languages)) != len(args.languages) or (
        args.per_stratum is not None and args.per_stratum <= 0
    ):
        parser.error("unique languages and positive sample count required")
    if not args.execute and not args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "model": MODEL_ID,
                    "levels": LEVELS,
                    "native_generation_and_author_scoring": True,
                    "model_calls": 0,
                }
            )
        )
    else:
        print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
