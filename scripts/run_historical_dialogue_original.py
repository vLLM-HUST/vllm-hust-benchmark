"""Replay complete original text histories; not a live episode or quality score."""

import argparse
import json
from pathlib import Path

from run_oasst_original_continuation import post, validate_endpoint


def run_records(
    records,
    *,
    output,
    endpoint,
    model,
    max_tokens,
    context,
    transport=post,
    final_verify=None,
    expected_count=None,
    source_identity=None,
):
    from vllm_hust_workload_data.conversations import historical_dialogue_views
    from vllm_hust_workload_data.identity import portable

    endpoint = validate_endpoint(endpoint)
    if not model or type(max_tokens) is not int or not 0 < max_tokens < context:
        raise ValueError("explicit model, positive output budget and context required")
    output = Path(output)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "status": "STARTED",
        "model": model,
        "max_tokens": max_tokens,
        "context_capacity": context,
        "conversations": 0,
        "requests": [],
        "quality_scored": False,
        "live_episode_validated": False,
        "original_api_request_capture": False,
        "performance_claim": False,
        "source_checkbox_eligible": False,
        "source_identity": source_identity,
    }

    def save(name, value):
        path = output / name
        path.write_text(
            json.dumps(portable(value), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.chmod(0o600)

    save("receipt-private.json", receipt)
    try:
        for record in records:
            # Preserve rejected original rows too, before any view validation.
            save(
                f"conversation-{receipt['conversations']:06d}.private.json",
                record.input_record(),
            )
            views = historical_dialogue_views(record)
            for view in views:
                attempt = {
                    "view": view,
                    "request": {
                        "model": model,
                        "messages": view["messages"],
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "n": 1,
                    },
                    "status": "PREPARED",
                }
                name = f"request-{len(receipt['requests']):06d}.private.json"
                save(name, attempt)
                try:
                    tokenized = transport(
                        endpoint.removesuffix("/v1") + "/tokenize",
                        {
                            "model": model,
                            "messages": view["messages"],
                            "add_generation_prompt": True,
                        },
                    )
                    attempt["tokenize"] = tokenized
                    save(name, attempt)
                    count = tokenized.get("count")
                    if (
                        type(count) is not int
                        or count <= 0
                        or count + max_tokens > context
                    ):
                        raise ValueError(
                            "complete original history exceeds declared context"
                        )
                    result = transport(
                        endpoint + "/chat/completions", attempt["request"]
                    )
                    attempt["response"] = result
                    save(name, attempt)
                    choices = result.get("choices")
                    usage = result.get("usage", {})
                    if (
                        not isinstance(choices, list)
                        or len(choices) != 1
                        or choices[0].get("finish_reason") not in {"stop", "length"}
                        or not isinstance(
                            choices[0].get("message", {}).get("content"), str
                        )
                        or not choices[0]["message"]["content"]
                        or usage.get("prompt_tokens") != count
                        or type(usage.get("completion_tokens")) is not int
                        or not 0 < usage["completion_tokens"] <= max_tokens
                    ):
                        raise ValueError("response integrity not verified")
                    attempt["status"] = (
                        "RESPONSE_INTEGRITY_VERIFIED"
                        if choices[0]["finish_reason"] == "stop"
                        else "BUDGET_EXHAUSTED_QUALITY_UNVERIFIED"
                    )
                    receipt["requests"].append(
                        {
                            "source_id": view["source_id"],
                            "source_row_index": view["source_row_index"],
                            "source_file": view["source"].get("file"),
                            "prefix_length": view[
                                "request_after_original_prefix_length"
                            ],
                            "status": attempt["status"],
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
            raise ValueError("no selected original conversations")
        if expected_count is not None and receipt["conversations"] != expected_count:
            raise ValueError("selected original conversation count mismatch")
        if final_verify is not None:
            final_verify()
        receipt["status"] = "ORIGINAL_HISTORY_GENERATED_NO_QUALITY_SCORE"
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save("receipt-private.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_id",
        choices=["WILDCHAT", "ULTRACHAT", "LMSYS_CHAT", "TULU3", "OPENTHOUGHTS3"],
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--context-capacity", type=int, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=3)
    selection.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from vllm_hust_workload_data.adapters import iter_workload
    from vllm_hust_workload_data.assets import asset_path, sha256
    from vllm_hust_workload_data.registry import source_spec

    data_root, output = args.data_root.resolve(), args.output.resolve()
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        parser.error("fresh private output must be separate from original data root")
    if not args.full and args.count < 3:
        parser.error("fixed original sample requires at least three conversations")
    manifest = json.loads(
        (args.prepared / "source-manifest.private.json").read_text("utf-8")
    )
    upstream = source_spec(args.source_id)
    known = {f["path"]: f for f in upstream["files"] if f["role"] == "raw_source"}
    files = manifest.get("files", [])
    if (
        manifest.get("source_id") != args.source_id
        or manifest.get("revision") != upstream["revision"]
        or manifest.get("upstream_revision_verified") is not True
        or not files
        or len({f["path"] for f in files}) != len(files)
        or any(
            f["path"] not in known
            or f["role"] != "data"
            or f["sha256"] != known[f["path"]]["sha256"]
            or f["bytes"] != known[f["path"]]["bytes"]
            for f in files
        )
    ):
        raise ValueError(
            "prepared manifest must bind unchanged registered original assets"
        )

    def verify():
        for item in files:
            path = asset_path(args.data_root, item["path"])
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("original asset identity changed")

    verify()
    records = iter_workload(
        args.source_id,
        data_root=args.data_root,
        manifest=manifest,
        limit=None if args.full else args.count,
    )
    receipt = run_records(
        records,
        output=args.output,
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        context=args.context_capacity,
        final_verify=verify,
        expected_count=sum(f["rows"] for f in files) if args.full else args.count,
        source_identity={
            "source_id": args.source_id,
            "revision": manifest["revision"],
            "files": files,
            "selection": "all original rows"
            if args.full
            else "first fixed original rows",
            "requested_conversations": None if args.full else args.count,
        },
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "conversations": receipt["conversations"],
                "requests": len(receipt["requests"]),
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
