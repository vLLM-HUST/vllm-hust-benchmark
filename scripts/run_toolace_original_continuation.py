"""Bounded continuation of original ToolACE histories, never online tool execution.

Publisher tool descriptions/system text and from/value turns are unchanged.
The selected local model's own tokenizer renders original tool roles directly;
no tool_call_id, synthetic tools, dynamic observations or function score is made.
"""

import argparse
import hashlib
import json
from pathlib import Path

from audit_toolace_original_corpus import DATA_SHA, REVISION, sha
from run_oasst_original_continuation import post, validate_endpoint


def local_tokenizer_files(root):
    root = Path(root).resolve()
    names = {"tokenizer_config.json", "tokenizer.json", "config.json"}
    for name in (
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        if (root / name).exists():
            names.add(name)
    templates = root / "chat_templates"
    if templates.is_dir():
        names.update(
            str(path.relative_to(root)).replace("\\", "/")
            for path in templates.rglob("*.jinja")
        )
    result = {}
    for name in sorted(names):
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("contained complete local tokenizer files required")
        result[name] = sha(path)
    return result


def verify_tokenizer_files(root, expected):
    if local_tokenizer_files(root) != expected:
        raise ValueError("local tokenizer configuration/template changed")


def original_prefixes(row):
    if (
        not isinstance(row, dict)
        or set(row) != {"system", "conversations"}
        or not isinstance(row["system"], str)
        or not isinstance(row["conversations"], list)
        or not row["conversations"]
    ):
        raise ValueError("original ToolACE system and conversations required")
    turns = row["conversations"]
    roles = []
    for turn in turns:
        if (
            not isinstance(turn, dict)
            or set(turn) != {"from", "value"}
            or turn["from"] not in {"user", "assistant", "tool"}
            or not isinstance(turn["value"], str)
        ):
            raise ValueError("unsupported original ToolACE turn; no coercion")
        roles.append(turn["from"])
    if roles[0] != "user":
        raise ValueError("original first user turn required")
    positions = [i for i, role in enumerate(roles) if role == "assistant"]
    if not positions or any(roles[i - 1] not in {"user", "tool"} for i in positions):
        raise ValueError("assistant must follow original user/tool context")
    return [
        {
            "original_assistant_turn_index": i,
            "messages": [{"role": "system", "content": row["system"]}]
            + [{"role": turn["from"], "content": turn["value"]} for turn in turns[:i]],
            "historical_reference": turns[i]["value"],
            "original_tool_turn_indices": [j for j in range(i) if roles[j] == "tool"],
        }
        for i in positions
    ]


def select(rows, *, start=0, count=3, full=False):
    if not isinstance(rows, list) or len(rows) != 11300:
        raise ValueError("complete unchanged publisher11300-row corpus required")
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot mix with range selection")
        return list(enumerate(rows))
    if start < 0 or count < 3 or start + count > len(rows):
        raise ValueError("fixed range of at least three original rows required")
    return list(enumerate(rows))[start : start + count]


def run(
    selected,
    *,
    render,
    output,
    endpoint,
    model,
    max_tokens,
    context,
    transport=post,
    final_verify=None,
    tokenizer_identity=None,
):
    endpoint = validate_endpoint(endpoint)
    if (
        not model
        or type(max_tokens) is not int
        or type(context) is not int
        or not 0 < max_tokens < context
    ):
        raise ValueError("explicit model/budget/context required")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "TOOLACE",
        "source_revision": REVISION,
        "status": "STARTED",
        "model": model,
        "max_tokens": max_tokens,
        "context_capacity": context,
        "tokenizer_identity": tokenizer_identity,
        "selected_original_rows": [i for i, _ in selected],
        "conversations": 0,
        "requests": [],
        "teacher_forced_original_history": True,
        "transport": "local_model_native_chat_template_raw_completions",
        "tools_executed": 0,
        "dynamic_observations": 0,
        "invented_tool_call_ids": 0,
        "quality_scored": False,
        "online_agent_episode": False,
        "publisher_model_replication": False,
        "source_checkbox_eligible": False,
    }

    def save(name, value):
        path = output / name
        with path.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        path.chmod(0o600)

    save("receipt-private.json", receipt)
    try:
        for index, row in selected:
            save(f"original-row-{index}.private.json", row)
            views = original_prefixes(row)
            for view in views:
                name = f"request-{len(receipt['requests']):06d}.private.json"
                attempt = {
                    "original_row_index": index,
                    "view": view,
                    "status": "PREPARING",
                }
                save(name, attempt)
                try:
                    prompt, local_tokens = render(view["messages"])
                    if (
                        not isinstance(prompt, str)
                        or type(local_tokens) is not int
                        or local_tokens <= 0
                    ):
                        raise ValueError("explicit local tokenizer rendering required")
                    attempt["request"] = {
                        "model": model,
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "n": 1,
                    }
                    attempt["local_token_count"] = local_tokens
                    save(name, attempt)
                    tokenized = transport(
                        endpoint.removesuffix("/v1") + "/tokenize",
                        {"model": model, "prompt": prompt},
                    )
                    attempt["tokenize"] = tokenized
                    save(name, attempt)
                    if (
                        tokenized.get("count") != local_tokens
                        or local_tokens + max_tokens > context
                    ):
                        raise ValueError(
                            "tokenizer identity or complete-context capacity mismatch"
                        )
                    result = transport(endpoint + "/completions", attempt["request"])
                    attempt["response"] = result
                    save(name, attempt)
                    choices, usage = result.get("choices"), result.get("usage", {})
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 1
                        or not isinstance(choices[0].get("text"), str)
                        or choices[0].get("finish_reason") not in {"stop", "length"}
                        or usage.get("prompt_tokens") != local_tokens
                        or type(usage.get("completion_tokens")) is not int
                        or not 0 <= usage["completion_tokens"] <= max_tokens
                    ):
                        raise ValueError(
                            "original bounded continuation terminal integrity failed"
                        )
                    attempt["status"] = "TERMINAL_RESPONSE_VERIFIED_NO_QUALITY_SCORE"
                    receipt["requests"].append(
                        {
                            "original_row_index": index,
                            "original_assistant_turn_index": view[
                                "original_assistant_turn_index"
                            ],
                            "historical_tool_turns": len(
                                view["original_tool_turn_indices"]
                            ),
                            "finish_reason": choices[0]["finish_reason"],
                            "usage": usage,
                        }
                    )
                except Exception as error:
                    attempt.update(
                        status="FAILED_NOT_REPLACED", error_type=type(error).__name__
                    )
                    raise
                finally:
                    save(name, attempt)
                    save("receipt-private.json", receipt)
            receipt["conversations"] += 1
        if not receipt["conversations"]:
            raise ValueError("no selected original conversation")
        if final_verify:
            final_verify()
        receipt["status"] = (
            "ORIGINAL_TOOL_HISTORY_CONTINUATION_COMPLETE_NO_TOOLS_OR_QUALITY_SCORE"
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save("receipt-private.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-capacity", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    validate_endpoint(args.endpoint)
    root, output, tokenizer_path = (
        args.data_root.resolve(),
        args.output.resolve(),
        args.tokenizer.resolve(),
    )
    if (
        output.is_relative_to(root)
        or root.is_relative_to(output)
        or output.is_relative_to(tokenizer_path)
        or tokenizer_path.is_relative_to(output)
    ):
        parser.error("fresh results must not overlap original data or tokenizer")
    if not tokenizer_path.is_dir():
        parser.error("explicit existing local tokenizer directory required")
    data = root / "data.json"
    identity = None

    def verify():
        if sha(data) != DATA_SHA:
            raise ValueError("original ToolACE asset identity mismatch")
        if identity:
            verify_tokenizer_files(tokenizer_path, identity["configuration_files"])

    verify()
    rows = json.loads(data.read_text("utf-8"))
    selected = select(rows, start=args.start, count=args.count, full=args.full)
    from transformers import AutoTokenizer

    token_files = local_tokenizer_files(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    verify_tokenizer_files(tokenizer_path, token_files)
    identity = {
        "directory": str(tokenizer_path),
        "tokenizer_class": type(tokenizer).__name__,
        "configuration_files": token_files,
        "effective_chat_template_sha256": hashlib.sha256(
            json.dumps(
                tokenizer.chat_template, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
    }

    def render(messages):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    result = run(
        selected,
        render=render,
        output=output,
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        context=args.context_capacity,
        final_verify=verify,
        tokenizer_identity=identity,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "conversations": result["conversations"],
                "requests": len(result["requests"]),
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
