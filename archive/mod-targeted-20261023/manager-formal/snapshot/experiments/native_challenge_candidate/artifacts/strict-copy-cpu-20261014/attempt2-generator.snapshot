"""New long-prompt KV cohort; actual tokenizer freeze, finite metadata scope only."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .request_runner import (
    CLASSIFICATION,
    canonical,
    make_recipe,
    require,
    sha,
    validate_rpc_budget,
)

SCHEMA = "ecpa-native-kv-challenge-campaign/v1"
SCHEMA_V2 = "ecpa-native-kv-challenge-campaign/v2"
REQUIREMENTS = {
    "attention_group": 0,
    "semantic_anchor": 1201,
    "physical_anchor": 1024,
    "compression_threshold_tokens": 1152,
    "allocation_block_size": 128,
    "destination_block_count": 8,
}


STRICT_COPY = "strict-system-copy-v1"
STRICT_SYSTEM = (
    "You are a deterministic text-copying assistant. Copy the entire text inside "
    "<answer>...</answer> in the user message exactly. Output only that text, "
    "preserving its single space. Do not explain, summarize, add labels, or stop "
    "after its first part."
)


def strict_copy_messages(expected, padding):
    require(type(padding) is int and 0 <= padding < 2402, "strict padding invalid")
    return [
        {"role": "system", "content": STRICT_SYSTEM},
        {
            "role": "user",
            "content": (
                "Ignore this padding:\n"
                + "pad " * padding
                + "\nCopy both space-separated strings below, verbatim. Your complete "
                "reply must contain both strings and nothing else.\n<answer>"
                + expected
                + "</answer>\nReturn the entire contents of <answer> exactly, "
                "including both strings separated by one space."
            ),
        },
    ]


def actual_chat_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    require(
        isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids),
        "actual tokenizer did not return plain token IDs",
    )
    return ids


def runtime_tokenization_binding(tokenizer):
    method_path = Path(
        inspect.getsourcefile(type(tokenizer).apply_chat_template)
    ).resolve()
    implementation = {"path": str(method_path), "sha256": sha(method_path.read_bytes())}
    fixture = type(tokenizer).__name__ == "CPUFixtureTokenizer"
    if fixture:
        return {
            "classification": "CPU_TOKENIZER_FIXTURE_ONLY",
            "transformers_version": None,
            "host_version": None,
            "implementation_source": implementation,
            "host_sources": [],
        }
    host = importlib.metadata.distribution("vllm")
    host_paths = [
        "vllm/renderers/hf.py",
        "vllm/renderers/chat_utils.py",
        "vllm/renderers/base.py",
        "vllm/renderers/params.py",
        "vllm/tokenizers/hf.py",
        "vllm/entrypoints/openai/chat_completion/serving.py",
        "vllm/entrypoints/serve/engine/serving.py",
        "vllm/v1/engine/input_processor.py",
    ]
    sources = []
    for relative in host_paths:
        path = Path(host.locate_file(relative)).resolve()
        require(path.is_file(), "actual Host tokenization source is unavailable")
        sources.append({"path": str(path), "sha256": sha(path.read_bytes())})
    return {
        "classification": "ACTUAL_LOCAL_TOKENIZER_CPU_PREFLIGHT_ONLY",
        "transformers_version": importlib.metadata.version("transformers"),
        "host_version": host.version,
        "implementation_source": implementation,
        "host_sources": sources,
    }


def make_kv_recipe(
    phase_targets,
    sources,
    tokenizer,
    tokenizer_sources,
    *,
    materialized_row=False,
    rpc_budget_s=None,
    prompt_variant=None,
):
    """No guessed prompt length: fail if actual local template cannot reach 1201."""
    require(prompt_variant in (None, STRICT_COPY), "unknown KV prompt variant")
    if prompt_variant is not None:
        require(materialized_row, "strict prompt requires opt-in KV v2")
    if rpc_budget_s is not None:
        require(materialized_row, "frozen RPC budget requires opt-in KV v2")
        validate_rpc_budget(rpc_budget_s)
    recipe = make_recipe(phase_targets, sources)
    recipe["schema"] = SCHEMA_V2 if materialized_row else SCHEMA
    recipe["request_scope"] = "KV-host-metadata-to-later-upload-only"
    require(tokenizer_sources, "tokenizer source bytes required")
    runtime_binding = runtime_tokenization_binding(tokenizer)
    for phase in recipe["phases"]:
        if rpc_budget_s is not None:
            phase["rpc_budget_s"] = rpc_budget_s
        expected = ("ECPA-CHALLENGE-" + phase["challenge_nonce"] + " ") * 2
        expected = expected.strip()
        if prompt_variant == STRICT_COPY:
            expected_ids = tokenizer.encode(expected, add_special_tokens=False)
            require(
                isinstance(expected_ids, list)
                and all(type(x) is int and x >= 0 for x in expected_ids)
                and 32 <= len(expected_ids) <= 128,
                "single-sampled strict oracle cannot satisfy frozen decode bounds",
            )
            phase["prompt_variant"] = STRICT_COPY
            phase["oracle_tokenization"] = {
                "schema": "ecpa-strict-copy-oracle-tokenization/v1",
                "token_ids": expected_ids,
                "token_ids_sha256": sha(canonical(expected_ids)),
                "count": len(expected_ids),
                "add_special_tokens": False,
                "sample_policy": (
                    "one-fresh-nonce-cpu-reject-if-short-no-runtime-retuning"
                ),
            }
        selected = None
        for padding in range(2402):
            content = (
                "Read the padding silently:\n"
                + "pad " * padding
                + "\nReply exactly: "
                + expected
            )
            messages = (
                strict_copy_messages(expected, padding)
                if prompt_variant == STRICT_COPY
                else [{"role": "user", "content": content}]
            )
            ids = actual_chat_ids(tokenizer, messages)
            if len(ids) == 1201:
                selected = messages, ids
                if prompt_variant == STRICT_COPY:
                    phase["padding_count"] = padding
                break
        require(
            selected is not None, "actual tokenizer cannot freeze exact 1201 prompt"
        )
        messages, ids = selected
        phase["payload"].update(
            messages=messages,
            max_tokens=128,
            chat_template_kwargs={"enable_thinking": False},
        )
        phase["response_oracle"]["expected"] = expected
        phase["kv_requirements"] = dict(REQUIREMENTS)
        phase["minimum_completion_tokens"] = 32
        phase["tokenization"] = {
            "schema": "ecpa-actual-KV-chat-tokenization/v1",
            "tokenizer_class": type(tokenizer).__name__,
            "runtime_binding": runtime_binding,
            "assets": tokenizer_sources,
            "messages_sha256": sha(canonical(messages)),
            "template_kwargs": {"enable_thinking": False},
            "add_generation_prompt": True,
            "prompt_token_ids": ids,
            "prompt_token_ids_sha256": sha(canonical(ids)),
            "prompt_token_count": len(ids),
        }
    validate_kv(recipe)
    return recipe


def validate_kv(recipe, *, source_resolver=None):
    require(
        recipe["schema"] in {SCHEMA, SCHEMA_V2}
        and recipe["classification"] == CLASSIFICATION
        and recipe["formal_result"] is False
        and recipe["native_ids_preregistered_before_launch"] is False
        and recipe["request_scope"] == "KV-host-metadata-to-later-upload-only",
        "KV cohort scope changed",
    )
    require(
        all(
            recipe[k] is None
            for k in ["lost_requests", "duplicate_requests", "full_objective_satisfied"]
        ),
        "KV unknown objective promoted",
    )
    require(
        recipe["sources"] and 0 < len(recipe["phases"]) <= 64,
        "bounded source-pinned KV cohort required",
    )
    resolve = source_resolver or Path
    for source in recipe["sources"]:
        path = resolve(source["path"])
        require(
            path.is_absolute()
            and path.is_file()
            and sha(path.read_bytes()) == source["sha256"],
            "KV frozen source changed",
        )
    for key in [
        "phase_id",
        "client_request_id",
        "challenge_nonce",
        "expected_response_id",
    ]:
        require(
            len({p[key] for p in recipe["phases"]}) == len(recipe["phases"]),
            "KV phase/request/nonce reused",
        )
    for phase in recipe["phases"]:
        variant = phase.get("prompt_variant")
        require(variant in (None, STRICT_COPY), "unknown KV prompt variant")
        if variant is not None:
            require(
                recipe["schema"] == SCHEMA_V2, "strict prompt requires opt-in KV v2"
            )
        if "rpc_budget_s" in phase:
            require(
                recipe["schema"] == SCHEMA_V2, "frozen RPC budget requires opt-in KV v2"
            )
            validate_rpc_budget(phase["rpc_budget_s"])
        client, nonce = phase["client_request_id"], phase["challenge_nonce"]
        require(
            re.fullmatch(r"[a-zA-Z0-9._-]{1,80}", phase["phase_id"]) is not None
            and all(
                re.fullmatch(r"[0-9a-f]{32}", x) is not None for x in [client, nonce]
            ),
            "KV phase/client nonce invalid",
        )
        url = urlsplit(phase["url"])
        require(
            url.scheme == "http"
            and url.hostname == "127.0.0.1"
            and url.port is not None
            and url.path == "/v1/chat/completions"
            and not url.query
            and not url.fragment
            and url.username is None,
            "direct KV endpoint required",
        )
        require(
            phase["expected_ranks"] == [0, 1]
            and phase["external_request_id"]
            == phase["expected_response_id"]
            == "chatcmpl-" + client,
            "KV native external request binding changed",
        )
        require(
            phase["headers"]
            == {"Content-Type": "application/json", "X-Request-Id": client},
            "KV HTTP identity changed",
        )
        payload, tokenization = phase["payload"], phase["tokenization"]
        require(
            payload["request_id"] == client
            and payload["temperature"] == 0
            and payload["n"] == 1
            and payload["stream"] is False
            and payload["max_tokens"] == 128
            and payload["chat_template_kwargs"] == {"enable_thinking": False},
            "KV bounded decode policy changed",
        )
        require(
            phase["minimum_completion_tokens"] == 32
            and phase["kv_requirements"] == REQUIREMENTS,
            "KV frozen semantic/decode predicate changed",
        )
        expected = " ".join(["ECPA-CHALLENGE-" + nonce] * 2)
        require(
            phase["response_oracle"]
            == {
                "expected": expected,
                "strip_outer_whitespace": True,
                "finish_reason": "stop",
            },
            "KV nonce response oracle changed",
        )
        messages = payload["messages"]
        if variant == STRICT_COPY:
            require(
                messages == strict_copy_messages(expected, phase.get("padding_count")),
                "strict canonical two-message prompt changed",
            )
            require(
                set(payload)
                == {
                    "request_id",
                    "model",
                    "messages",
                    "temperature",
                    "max_tokens",
                    "n",
                    "stream",
                    "chat_template_kwargs",
                },
                "strict copy forbids extra decode constraints",
            )
            receipt = phase["oracle_tokenization"]
            expected_ids = receipt["token_ids"]
            require(
                set(receipt)
                == {
                    "schema",
                    "token_ids",
                    "token_ids_sha256",
                    "count",
                    "add_special_tokens",
                    "sample_policy",
                }
                and receipt["schema"] == "ecpa-strict-copy-oracle-tokenization/v1"
                and isinstance(expected_ids, list)
                and all(type(x) is int and x >= 0 for x in expected_ids)
                and 32 <= receipt["count"] == len(expected_ids) <= 128
                and receipt["token_ids_sha256"] == sha(canonical(expected_ids))
                and receipt["add_special_tokens"] is False
                and receipt["sample_policy"]
                == "one-fresh-nonce-cpu-reject-if-short-no-runtime-retuning",
                "strict single-sampled oracle tokenizer receipt changed",
            )
        else:
            require(
                isinstance(messages, list)
                and len(messages) == 1
                and set(messages[0]) == {"role", "content"}
                and messages[0]["role"] == "user"
                and messages[0]["content"].endswith("\nReply exactly: " + expected),
                "KV challenge prompt changed",
            )
        ids = tokenization["prompt_token_ids"]
        require(
            tokenization["schema"] == "ecpa-actual-KV-chat-tokenization/v1"
            and tokenization["assets"]
            and tokenization["tokenizer_class"]
            and tokenization["messages_sha256"] == sha(canonical(messages))
            and tokenization["template_kwargs"] == {"enable_thinking": False}
            and tokenization["add_generation_prompt"] is True
            and tokenization["prompt_token_count"] == len(ids) == 1201
            and all(type(x) is int and x >= 0 for x in ids)
            and tokenization["prompt_token_ids_sha256"] == sha(canonical(ids)),
            "KV actual tokenizer receipt changed",
        )
        binding = tokenization["runtime_binding"]
        require(
            binding["classification"]
            in {
                "CPU_TOKENIZER_FIXTURE_ONLY",
                "ACTUAL_LOCAL_TOKENIZER_CPU_PREFLIGHT_ONLY",
            },
            "tokenization source classification unsupported",
        )
        if binding["classification"] != "CPU_TOKENIZER_FIXTURE_ONLY":
            require(
                binding["transformers_version"]
                and binding["host_version"]
                and len(binding["host_sources"]) == 8,
                "actual Host/transformers tokenization source binding incomplete",
            )
        for binding_source in [
            binding["implementation_source"],
            *binding["host_sources"],
        ]:
            path = resolve(binding_source["path"])
            require(
                path.is_absolute()
                and path.is_file()
                and sha(path.read_bytes()) == binding_source["sha256"],
                "tokenization implementation source bytes changed",
            )
        for asset in tokenization["assets"]:
            path = resolve(asset["path"])
            require(
                path.is_absolute()
                and path.is_file()
                and sha(path.read_bytes()) == asset["sha256"],
                "KV tokenizer assets changed",
            )
        require(
            not any(
                k in phase
                for k in ["native_plan_id", "native_launch_id", "runtime_request_id"]
            ),
            "KV native identities must be minted after launch",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized-row-v2", action="store_true")
    parser.add_argument("--rpc-budget-s", type=float)
    parser.add_argument("--prompt-variant", choices=[STRICT_COPY])
    args = parser.parse_args()
    from transformers import AutoTokenizer

    config = json.loads(args.configuration.read_bytes())
    directory = args.model_directory.resolve(strict=True)
    assets = [
        {"path": str(p), "sha256": sha(p.read_bytes())}
        for p in sorted(directory.iterdir())
        if p.is_file()
        and p.suffix in {".json", ".txt", ".jinja", ".model", ".tiktoken"}
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    targets = [(p["phase_id"], p["url"], p["model"]) for p in config["phase_targets"]]
    recipe = make_kv_recipe(
        targets,
        config["sources"],
        tokenizer,
        assets,
        materialized_row=args.materialized_row_v2,
        rpc_budget_s=args.rpc_budget_s,
        prompt_variant=args.prompt_variant,
    )
    # The caller must establish explicit 0700, including cleared special bits.
    import os
    import stat

    require(
        args.output.is_absolute()
        and stat.S_IMODE(args.output.parent.stat().st_mode) == 0o700
        and args.output.parent.stat().st_uid == os.getuid(),
        "private owned KV freeze parent required",
    )
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(recipe))
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps({"recipe_sha256": sha(canonical(recipe)), "formal_result": False}))


if __name__ == "__main__":
    main()
