"""Original GPT2-large base-probe perplexity, explicit NPU compatibility.

Default plans only. --execute uses original model/Dataset/evaluate_ppl bodies;
--full evaluates all eight original probe CSVs instead of fixed first3 per file.
Training checkpoints, corpora, T5 and temporal-update claims are separate modes.
"""

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

from prepare_jsonschemabench_author_source import verify_source
from prepare_temporalwiki_original_assets import verify_bytes
from temporalwiki_original_binding import baseline_factory_tree, npu_evaluator_tree

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_PINS = ROOT / "configs/workload-native/temporalwiki-author-source.json"
MODEL_PINS = ROOT / "configs/workload-native/temporalwiki-gpt2-large-model-pins.json"
DATA_PINS = (
    ROOT
    / "workload-preparation/data/catalogs/temporalwiki_original_probe_files_v1.json"
)


def verify_model(root, pins):
    if pins["model"] != "openai-community/gpt2-large":
        raise ValueError("original GPT2-large base model required")
    for item in pins["files"]:
        if Path(item["path"]).name != item["path"]:
            raise ValueError("contained original model file required")
        path = root / item["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
        ):
            raise ValueError("original model file identity or size differs")
        digest = (
            hashlib.sha256()
            if item.get("lfs_sha256")
            else hashlib.sha1(b"blob " + str(item["bytes"]).encode() + b"\0")
        )
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("lfs_sha256", item.get("git_blob")):
            raise ValueError("original model Git/LFS identity differs")


def run(args):
    import re

    if not re.fullmatch(r"npu:[0-9]+", args.device):
        raise ValueError("explicit NPU required; no CPU/CUDA model fallback")
    author = verify_source(args.author_root, AUTHOR_PINS)
    profile = json.loads(DATA_PINS.read_text("utf-8"))
    files = [item for item in profile["files"] if item["role"] == "data"]
    if len(files) != 8:
        raise ValueError("all eight original probe files required")
    for item in files:
        path = args.data_root / item["path"]
        if not path.resolve().is_relative_to(args.data_root.resolve()):
            raise ValueError("contained original probe file required")
        verify_bytes(path.read_bytes(), item)
    factory_tree, removed = baseline_factory_tree(
        (author / "models/GPT2_Model.py").read_text("utf-8")
    )
    evaluator_tree, placements = npu_evaluator_tree(
        (author / "evaluation_ppl.py").read_text("utf-8"), args.device
    )
    report = {
        "source_id": "TEMPORALWIKI",
        "status": "PLAN_ONLY",
        "protocol": "GPT2-large baseline evaluate_ppl with empty checkpoint_path",
        "original_probe_files": 8,
        "original_rows": sum(item["rows"] for item in files),
        "selection": "all original rows" if args.full else "first3 each original file",
        "selected_rows": sum(
            item["rows"] if args.full else min(3, item["rows"]) for item in files
        ),
        "original_input_length": 25,
        "original_output_length": 25,
        "original_batch_size": 32,
        "excluded_training_only_imports": removed,
        "device_placements": placements,
        "reproducibility_seed": args.seed,
        "allocator_budget_gib": args.allocator_budget_gib,
        "seed_note": "Explicit compatibility-profile seed; original validation branch does not call set_seed",
        "model_calls": 0,
        "completed_model_calls": 0,
        "native_execution_complete": False,
        "source_checkbox_eligible": False,
        "paper_checkpoint_or_quality_reproduction": False,
    }
    if not args.execute:
        return report
    verify_model(
        args.model_root.resolve(strict=True), json.loads(MODEL_PINS.read_text("utf-8"))
    )
    output = args.output.resolve()
    for protected in [
        author,
        args.data_root.resolve(),
        args.model_root.resolve(),
        ROOT,
    ]:
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                "fresh external output separate from original inputs required"
            )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    previous_cwd, previous_path = Path.cwd(), list(sys.path)
    try:
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("available NPU required")
        torch.npu.set_device(args.device)
        if args.allocator_budget_gib is not None:
            budget = args.allocator_budget_gib * 1024**3
            total = torch.npu.get_device_properties(args.device).total_memory
            if not math.isfinite(budget) or not 0 < budget <= total:
                raise ValueError(
                    "positive allocator budget within device capacity required"
                )
            torch.npu.set_per_process_memory_fraction(budget / total, args.device)
        torch.manual_seed(args.seed)
        torch.npu.manual_seed(args.seed)
        if "Datasets" in sys.modules:
            raise RuntimeError("fresh isolated original Dataset import required")
        sys.path.insert(0, str(author))
        original_dataset = importlib.import_module("Datasets").CustomDataset
        model_module = ModuleType("temporalwiki_original_baseline")
        model_module.__file__ = str(author / "models/GPT2_Model.py")
        sys.modules[model_module.__name__] = model_module
        exec(
            compile(factory_tree, model_module.__file__, "exec"), model_module.__dict__
        )
        evaluator = ModuleType("temporalwiki_original_evaluation")
        exec(
            compile(evaluator_tree, str(author / "evaluation_ppl.py"), "exec"),
            evaluator.__dict__,
        )

        class SelectedDataset(original_dataset):
            def __init__(self, *values, **options):
                super().__init__(*values, **options)
                if not args.full:
                    self.dataset = self.dataset.iloc[:3]

        evaluator.CustomDataset = SelectedDataset
        (output / "data").symlink_to(args.data_root.resolve(), target_is_directory=True)
        (output / "models").mkdir()
        local_model = output / "models/gpt2-large"
        local_model.symlink_to(args.model_root.resolve(), target_is_directory=True)
        (output / "scores").mkdir()
        os.chdir(output)
        results = []
        for item in files:
            name = Path(item["path"]).stem
            settings = argparse.Namespace(
                output_dir=str(output),
                dataset=name,
                dataset_version="full",
                method="baseline",
                mode="evaluate_ppl",
                model_name_or_path=str(local_model),
                max_input_length=25,
                max_output_length=25,
                train_batch_size=32,
                check_validation_only=True,
                checkpoint_path="",
                seed=args.seed,
                output_log=str(output / "scores" / (name + ".csv")),
            )
            losses, models, handles = [], [], []

            def factory(values):
                model = model_module.GPT2(values)
                models.append(model)

                def before_forward(module, inputs, keywords):
                    if any(p.device.type != "npu" for p in module.parameters()):
                        raise RuntimeError(
                            "all neural weights must be on NPU before forward"
                        )
                    tensors = list(inputs) + list(keywords.values())
                    if any(
                        isinstance(x, torch.Tensor) and x.device.type != "npu"
                        for x in tensors
                    ):
                        raise RuntimeError("all neural inputs must be on NPU")
                    report["model_calls"] += 1

                def after_forward(module, inputs, outputs):
                    loss = outputs[0]
                    if loss.device.type != "npu" or not bool(torch.isfinite(loss)):
                        raise RuntimeError("finite original NPU loss required")
                    losses.append(float(loss.detach().item()))
                    report["completed_model_calls"] += 1

                handles.extend(
                    [
                        model.model.register_forward_pre_hook(
                            before_forward, with_kwargs=True
                        ),
                        model.model.register_forward_hook(after_forward),
                    ]
                )
                return model

            evaluator.evaluate_ppl(settings, factory)
            (output / "scores" / (name + ".batch-losses.json")).write_text(
                json.dumps(losses), "utf-8"
            )
            score_lines = Path(settings.output_log).read_text("utf-8").splitlines()
            if len(score_lines) != 1 or not losses:
                raise ValueError("one original perplexity CSV result required")
            perplexity = float(score_lines[0])
            independent = math.exp(sum(losses) / len(losses))
            if not math.isclose(perplexity, independent, rel_tol=1e-5):
                raise ValueError("original mean-batch-loss perplexity audit differs")
            results.append(
                {
                    "file": item["path"],
                    "rows": item["rows"] if args.full else 3,
                    "batches": len(losses),
                    "original_perplexity": perplexity,
                    "observed_model_dtype": str(next(models[0].parameters()).dtype),
                    "attention_implementation": models[
                        0
                    ].model.config._attn_implementation,
                    "original_batch_loss_mean_audit_passed": True,
                }
            )
            for handle in handles:
                handle.remove()
            models.clear()
            gc.collect()
            torch.npu.empty_cache()
            report["results"] = results
            (output / "receipt-public.json").write_text(
                json.dumps(report, indent=2), "utf-8"
            )
        report.update(
            status="ORIGINAL_BASE_MODEL_PROBE_PERPLEXITY_EXECUTED",
            native_execution_complete=True,
        )
    except BaseException as error:
        report.update(
            status="FAILED_RETAINED_NO_REPLACEMENT", error_type=type(error).__name__
        )
        raise
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        (output / "receipt-public.json").write_text(
            json.dumps(report, indent=2), "utf-8"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["author-root", "data-root", "model-root", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allocator-budget-gib", type=float)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))
