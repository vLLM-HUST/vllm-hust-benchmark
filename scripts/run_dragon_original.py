"""Original DRAGON core retrieval/generation/scoring with explicit NPU transport.

Full original corpus, original first3 (or full) questions, unchanged author core.
This is a deployment adapter, not the author's CUDA CLI or model-score reproduction.
No reference-derived retrieval pool, fake updates or research observation.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from prepare_jsonschemabench_author_source import verify_source
from probe_dragon_native_environment import verify_versions
from run_docvqa_original import local_endpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/workload-native"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_artifact(root, pins):
    root = Path(root).resolve(strict=True)
    for spec in pins["files"]:
        file = (root / spec["path"]).resolve(strict=True)
        if not file.is_relative_to(root) or file.stat().st_size != spec["bytes"]:
            raise ValueError("original embedding asset path or length differs")
        if "sha256" in spec:
            valid = sha(file) == spec["sha256"]
        else:
            raw = file.read_bytes()
            valid = (
                hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                == spec["git_blob"]
            )
        if not valid:
            raise ValueError("original embedding published identity differs")
    if list(root.rglob("*.py")):
        raise ValueError("original export contains no remote Python files")


def original_system_prompt(author):
    tree = ast.parse(
        (author / "examples/calculate_baseline_metrics.py").read_text("utf-8")
    )
    values = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "llm_system_prompt"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(values) != 1:
        raise ValueError("one unchanged author standalone system prompt required")
    return values[0]


def reference_subset(reference_rows, public_ids):
    selected = [row for row in reference_rows if row["public_id"] in public_ids]
    if len(selected) != len(public_ids) or {
        row["public_id"] for row in selected
    } != set(public_ids):
        raise ValueError("exact original selected reference relations required")
    return selected


def reader_generation_config(root, generation_class, model_config_class):
    if (Path(root) / "generation_config.json").exists():
        return generation_class.from_pretrained(
            str(root), local_files_only=True
        ), "original_generation_config.json"
    config = model_config_class.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False
    )
    return generation_class.from_model_config(
        config
    ), "transformers_from_original_model_config_no_separate_generation_file"


def generation_parameters(config, model):
    return {
        "model": model,
        "temperature": config.temperature if config.temperature is not None else 0.7,
        "top_p": config.top_p if config.top_p is not None else 0.95,
        "top_k": config.top_k if config.top_k is not None else 20,
        "max_tokens": config.max_new_tokens
        if config.max_new_tokens is not None
        else 512,
        "seed": 42,
    }


def idle(endpoint, engine_pid):
    with urlopen(endpoint.removesuffix("/v1") + "/metrics", timeout=10) as response:
        lines = response.read().decode().splitlines()
    counts = {}
    for line in lines:
        if line.startswith(
            ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
        ):
            key = line.split("{", 1)[0]
            counts[key] = counts.get(key, 0) + float(line.split()[-1])
    devices = subprocess.check_output(["npu-smi", "info"], text=True)
    owners = {
        int(pid)
        for pid in re.findall(r"^\|\s+\d+\s+\d+\s+\|\s+(\d+)\s+\|", devices, re.M)
    }
    if (
        set(counts) != {"vllm:num_requests_running", "vllm:num_requests_waiting"}
        or any(counts.values())
        or engine_pid not in owners
        or not owners.issubset({engine_pid, os.getpid()})
    ):
        raise RuntimeError("service busy or other NPU owner; no mixed requests")


def complete(endpoint, body, output, index):
    request_file = output / f"reader-{index:04d}-request-private.json"
    response_file = output / f"reader-{index:04d}-response-private.json"
    with request_file.open("x", encoding="utf-8") as handle:
        json.dump(body, handle, ensure_ascii=False)
    request = Request(
        endpoint + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=1200) as response:
        result = json.load(response)
    with response_file.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False)
    if len(result.get("choices", [])) != 1 or not isinstance(
        result["choices"][0].get("text"), str
    ):
        raise ValueError("one original reader completion required; response retained")
    if result["choices"][0].get("finish_reason") not in {"stop", "length"}:
        raise ValueError("reader not terminal; response retained, no automatic retry")
    return result["choices"][0]["text"]


def execute(args):
    endpoint = local_endpoint(args.endpoint)
    author = verify_source(args.author_root, CONFIG / "dragon-author-source.json")
    data = args.data_root.resolve(strict=True)
    model_root = args.embedding_root.resolve(strict=True)
    reader_root = args.reader_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (ROOT, author, data, model_root, reader_root)
    ):
        raise ValueError(
            "fresh private output separate from source/data/models required"
        )
    if not re.fullmatch(r"npu:\d+", args.device):
        raise ValueError("explicit NPU device required; no CPU model fallback")
    if not args.engine_pid or args.context_window <= 0:
        raise ValueError("actual owned engine PID and context window required")
    pins = json.loads((CONFIG / "dragon-e5-original-artifact.json").read_bytes())
    verify_artifact(model_root, pins)
    verify_versions(
        json.loads(
            (CONFIG / "dragon-npu-inherited-runtime-py312-observed.json").read_bytes()
        )
    )
    idle(endpoint, args.engine_pid)
    sys.path.insert(0, str(author / "lib/src"))
    from datasets import Dataset, DatasetDict
    from vllm_hust_workload_data.dragon import original_rows
    from rag_bench import baseline, evaluator
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.language_models.llms import LLM
    import torch
    import torch_npu  # noqa: F401 - explicit NPU registration, no CUDA shim

    if any(
        Path(module.__file__).resolve()
        != author / "lib/src/rag_bench" / (module.__name__.split(".")[-1] + ".py")
        for module in (baseline, evaluator)
    ):
        raise RuntimeError("imported original core must match the verified author root")

    original = list(
        original_rows(data / "public_questions/data/train-00000-of-00001.parquet")
    )
    questions = Dataset.from_parquet(
        str(data / "public_questions/data/train-00000-of-00001.parquet")
    )
    texts = Dataset.from_parquet(
        str(data / "public_texts/data/train-00000-of-00001.parquet")
    )
    references = Dataset.from_parquet(
        str(data / "reference_qa/data/train-00000-of-00001.parquet")
    )
    mappings = Dataset.from_parquet(
        str(data / "reference_texts/data/train-00000-of-00001.parquet")
    )
    count = len(questions) if args.full else 3
    if len(original) != len(questions) or len(questions) < count:
        raise ValueError(
            "complete original release and fixed original question order required"
        )
    ids = [questions[index]["id"] for index in range(count)]
    target = torch.device(args.device)
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "DRAGON_PUBLIC",
        "status": "PREPARING_ORIGINAL_NATIVE_PIPELINE",
        "original_corpus_documents": len(texts),
        "original_selected_question_ids": ids,
        "original_question_count": len(questions),
        "top_k": 5,
        "chunk_size_characters": 1000,
        "chunk_overlap_characters": 100,
        "search_type": "mmr",
        "reader_model": args.model,
        "embedding_model": pins["model"],
        "embedding_revision": pins["revision"],
        "embedding_device": args.device,
        "npu_embedding_encode_calls": 0,
        "reader_calls": 0,
        "source_checkbox_eligible": False,
        "started": time.time(),
    }

    def save():
        (output / "native-public.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    previous_cwd = Path.cwd()
    # The unchanged author helper writes LOG_FILE="log.txt" relative to cwd.
    # Keep this native runtime artifact in the explicit private output, not HOME
    # or a caller's source checkout. No author constant or message is rewritten.
    os.chdir(output)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(reader_root), local_files_only=True, trust_remote_code=False
        )
        generation, config_source = reader_generation_config(
            reader_root, GenerationConfig, AutoConfig
        )
        parameters = generation_parameters(generation, args.model)
        receipt["reader_generation_parameters"] = parameters
        receipt["reader_generation_config_source"] = config_source
        receipt["reader_model_config_sha256"] = sha(reader_root / "config.json")
        save()
        torch.npu.set_device(target)
        if torch.npu.mem_get_info()[0] < 8 * 1024**3:
            raise RuntimeError(
                "less than8GiB free selected NPU; preserve other services"
            )
        torch.manual_seed(42)

        class NativeEmbeddings(HuggingFaceEmbeddings):
            def _embed(self, values, encode_kwargs):
                idle(endpoint, args.engine_pid)
                vectors = super()._embed(values, encode_kwargs)
                index = len(list(output.glob("embedding-batch-*-private.json")))
                with (output / f"embedding-batch-{index:04d}-private.json").open(
                    "x", encoding="utf-8"
                ) as handle:
                    json.dump(
                        {
                            "texts": values,
                            "encode_kwargs": encode_kwargs,
                            "embeddings": vectors,
                        },
                        handle,
                        ensure_ascii=False,
                    )
                return vectors

        embedding = NativeEmbeddings(
            model_name=str(model_root),
            model_kwargs={
                "device": args.device,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_kwargs": {
                    "dtype": torch.float32,
                    "attn_implementation": "eager",
                },
            },
        )
        if embedding._client.max_seq_length != pins["native_max_seq_length"] or any(
            p.device != target for p in embedding._client.parameters()
        ):
            raise RuntimeError(
                "original encoder window and all NPU parameters required"
            )

        original_encode = embedding._client.encode

        def checked_encode(*values, **kwargs):
            # Only change the returned representation, not encoder inputs or
            # computation. ST6 calls model.forward directly, bypassing PyTorch
            # forward hooks; validate the actual NPU tensor before CPU export.
            kwargs["convert_to_tensor"] = True
            tensor = original_encode(*values, **kwargs)
            if tensor.device != target or not torch.isfinite(tensor).all().item():
                raise RuntimeError(
                    "finite actual NPU forward required, no CPU model fallback"
                )
            receipt["npu_embedding_encode_calls"] += 1
            return tensor

        embedding._client.encode = checked_encode
        retriever = baseline.init_retriever(
            DatasetDict(train=texts),
            embedding,
            top_k=5,
            chunk_size=1000,
            chunk_overlap=100,
        )
        receipt.update(
            status="ORIGINAL_FULL_CORPUS_NPU_RETRIEVER_READY",
            indexed_chunks=retriever.vectorstore._collection.count(),
        )
        receipt["embedding_instruction_profile"] = (
            "Original standalone CLI default: no doc/query prompt; original exported Normalize module retained"
        )
        save()

        class ExistingNpuReader(LLM):
            @property
            def _llm_type(self):
                return "original-dragon-existing-npu-completion-transport"

            def _call(self, prompt, stop=None, run_manager=None, **kwargs):
                idle(endpoint, args.engine_pid)
                if (
                    len(tokenizer.encode(prompt)) + parameters["max_tokens"]
                    > args.context_window
                ):
                    raise RuntimeError("original prompt exceeds context; no truncation")
                body = dict(parameters, prompt=prompt)
                if stop is not None:
                    body["stop"] = stop
                index = receipt["reader_calls"]
                result = complete(endpoint, body, output, index)
                receipt["reader_calls"] += 1
                save()
                return result

        chain = baseline.init_generation(
            retriever,
            ExistingNpuReader(),
            tokenizer,
            system_prompt=original_system_prompt(author),
        )
        results = baseline.get_results(
            chain, DatasetDict(train=questions), take=None if args.full else 3
        )
        if set(results) != set(ids) or receipt["reader_calls"] != count:
            raise RuntimeError("all and only fixed original tasks must be retained")
        with (output / "answers-private.json").open("x", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False)
        selected = Dataset.from_list(reference_subset(list(references), ids))
        mapping = {row["public_id"]: row["id"] for row in mappings}
        scored = evaluator.evaluate_rag_results(
            results, DatasetDict(train=selected), mapping
        ).to_dict()
        with (output / "official-scores-private.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(scored, handle, ensure_ascii=False)
        if receipt["npu_embedding_encode_calls"] <= 0:
            raise RuntimeError("missing actual NPU embedding execution evidence")
        receipt.update(
            status="FIXED_ORIGINAL_RETRIEVAL_READER_AND_OFFICIAL_SCORING_COMPLETE_AUDIT_PENDING",
            scored_original_questions=len(scored["individual_results"]),
            original_source_unchanged=True,
            answers_sha256=sha(output / "answers-private.json"),
            scores_sha256=sha(output / "official-scores-private.json"),
            finished=time.time(),
        )
    except BaseException as error:
        receipt.update(
            status="RETAINED_FAILURE_NO_TASK_REPLACEMENT_OR_AUTOMATIC_RETRY",
            error_type=type(error).__name__,
        )
        raise
    finally:
        save()
        os.chdir(previous_cwd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "data-root", "embedding-root", "reader-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-pid", type=int)
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        execute(args)
    else:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_MODEL_IMPORT_OR_FILE_CREATION",
                    "selected_questions": "all" if args.full else "first3",
                    "corpus": "complete original542-document corpus",
                    "core": "unchanged author retrieval/generation/scoring",
                    "model_calls": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
