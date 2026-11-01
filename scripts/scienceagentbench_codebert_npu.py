"""Original SAB CodeBERTScore, explicitly bound to pinned local NPU weights.

This auxiliary metric is not the scientific execution grader or source delivery.
Coordinate a resource window before running; this entry does not stop services.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


SOURCE_SHA256 = "5cb7e93cc0403e0aac99e14c9feb2cf45973351f7bd47861c5c82d374bbf2eb0"  # pragma: allowlist secret - pinned publisher source
WEIGHTS_SHA256 = "b96022334a02675d3cce85bc1ef159802c3e6724669393dc05e4eb53aa4a750c"  # pragma: allowlist secret - public model LFS identity
PACKAGE_HASHES = {
    "__init__.py": "290a18bfff42cf87e4904e75ecce859854684e990f24b280f7381ee458ecd321",  # pragma: allowlist secret - original package identity
    "score.py": "94c5db15afc6fd14313fd425fafff24f90d665341e0b57c327101e38f6a0ab34",  # pragma: allowlist secret - original package identity
    "scorer.py": "c210480a0c2eb1fe8fc47ab94b1e1d7f3d999018dc3635d67089ee3beaeb7c19",  # pragma: allowlist secret - original package identity
    "utils.py": "7d57d0683b44442bb5b5a827d55a9858cb7b713a38908079bfa72c70711009e2",  # pragma: allowlist secret - original package identity
}
MODEL_BLOBS = {
    "README.md": "63d013702f50210a28d992b66ce31e9f591176fd",  # pragma: allowlist secret - publisher Git identity
    "config.json": "c00e7773aa213ae4bb2cbf3d34207999020def5c",  # pragma: allowlist secret - publisher Git identity
    "merges.txt": "6636bda4a1fd7a63653dffb22683b8162c8de956",  # pragma: allowlist secret - publisher Git identity
    "special_tokens_map.json": "d5698132694f4f1bcff08fa7d937b1701812598e",  # pragma: allowlist secret - publisher Git identity
    "tokenizer.json": "93951c5b9caf8f0315a9c4078eda2f5976f27085",  # pragma: allowlist secret - publisher Git identity
    "tokenizer_config.json": "939b13a66a1dd980040c83f6bd35d0f73f564540",  # pragma: allowlist secret - publisher Git identity
    "vocab.json": "4ebe4bb3f3114daf2e4cc349f24873a1175a35d7",  # pragma: allowlist secret - publisher Git identity
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model(model_dir):
    model_dir = Path(model_dir).resolve(strict=True)
    for name, identity in MODEL_BLOBS.items():
        raw = (model_dir / name).read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest()
        if actual != identity:
            raise ValueError(f"original model Git identity mismatch: {name}")
    if file_sha256(model_dir / "model.safetensors") != WEIGHTS_SHA256:
        raise ValueError("original Python CodeBERT weights mismatch")
    return model_dir


def original_metric(source_file, score_function):
    """Load only the pinned original text-reading/metric function, not execution."""
    raw = Path(source_file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("pinned original compute_scores.py required")
    functions = [
        node
        for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name == "codebert_score"
    ]
    if len(functions) != 1:
        raise ValueError("one original codebert_score function required")
    namespace = {"code_bert_score": SimpleNamespace(score=score_function)}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source_file), "exec"),
        namespace,
    )
    return namespace["codebert_score"]


def bound_score(score_function, model_dir, device):
    """Keep original metric args; only bind original model identity and NPU."""
    if device not in {f"npu:{i}" for i in range(64)}:
        raise ValueError("explicit logical NPU device required")

    def score(**kwargs):
        if set(kwargs) != {"cands", "refs", "lang", "sources"}:
            raise ValueError("unchanged original SAB metric arguments required")
        if kwargs["lang"] != "python":
            raise ValueError("original Python metric required")
        return score_function(
            **kwargs, model_type=str(model_dir), num_layers=11, device=device
        )

    return score


def run(args):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_dir = verify_model(args.model_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch_npu  # noqa: F401 - explicit NPU backend registration

    if importlib.metadata.version("code-bert-score") != "0.4.1":
        raise ValueError("original code-bert-score distribution0.4.1 required")
    spec = importlib.util.find_spec("code_bert_score")
    if spec is None or spec.origin is None:
        raise ValueError("original scoring package required")
    package = Path(spec.origin).parent
    for name, digest in PACKAGE_HASHES.items():
        if file_sha256(package / name) != digest:
            raise ValueError(f"original scoring implementation mismatch: {name}")
    import code_bert_score

    device = args.device
    # Validate before initializing a device or processing task records.
    score = bound_score(code_bert_score.score, model_dir, device)
    metric = original_metric(args.original_source, score)
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty original task/prediction/reference records required")
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "task_inst",
            "pred_fname",
            "gold_fname",
        }:
            raise ValueError("explicit original metric record fields required")
        if not all(isinstance(value, str) and value for value in record.values()):
            raise ValueError("original nonempty text and file paths required")
    torch.npu.set_device(device)
    total = torch.npu.get_device_properties(device).total_memory
    torch.npu.set_per_process_memory_fraction(3 * 1024**3 / total, device)
    forward_count = 0
    score_globals = code_bert_score.score.__globals__
    original_factory = score_globals["get_model"]

    def factory(*factory_args, **factory_kwargs):
        model = original_factory(*factory_args, **factory_kwargs)

        def before(module, inputs, kwargs):
            if any(str(p.device) != device for p in module.parameters()) or any(
                str(p.device) != device for p in module.buffers()
            ):
                raise RuntimeError("neural weights not entirely on requested NPU")
            if any(
                isinstance(x, torch.Tensor) and str(x.device) != device
                for x in (*inputs, *kwargs.values())
            ):
                raise RuntimeError("neural inputs not on requested NPU")

        def after(module, inputs, result):
            nonlocal forward_count
            hidden = getattr(result, "last_hidden_state", None)
            if not isinstance(hidden, torch.Tensor) or str(hidden.device) != device:
                raise RuntimeError("neural output not on requested NPU")
            forward_count += 1

        model.register_forward_pre_hook(before, with_kwargs=True)
        model.register_forward_hook(after)
        return model

    report = {
        "source_id": "SCIENCEAGENTBENCH",
        "status": "AUXILIARY_METRIC_RUNNING",
        "source_checkbox_eligible": False,
        "generated_code_executed": False,
        "device": device,
        "original_internal_version": code_bert_score.__version__,
        "distribution_version": "0.4.1",
        "model": "neulab/codebert-python",
        "model_layers": 11,
        "results": [],
    }
    receipt = output / "receipt-public.json"
    score_globals["get_model"] = factory
    try:
        for index, record in enumerate(records):
            value = metric(**record)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite original metric output required")
            report["results"].append(
                {
                    "position": index,
                    "task_sha256": hashlib.sha256(
                        record["task_inst"].encode()
                    ).hexdigest(),
                    "prediction_sha256": file_sha256(record["pred_fname"]),
                    "reference_sha256": file_sha256(record["gold_fname"]),
                    "codebert_score": value,
                }
            )
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if forward_count == 0:
            raise RuntimeError("no completed neural model forward")
        report["status"] = (
            "ORIGINAL_AUXILIARY_METRIC_COMPLETED_NOT_SCIENTIFIC_TASK_DELIVERY"
        )
    except BaseException as exc:
        report.update(
            status="AUXILIARY_METRIC_FAILED_RETAINED", error_type=type(exc).__name__
        )
        raise
    finally:
        score_globals["get_model"] = original_factory
        report["completed_npu_forwards"] = forward_count
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-source", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="npu:0")
    run(parser.parse_args())
