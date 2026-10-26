"""Generate from explicit original OASST histories; no quality or episode score."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DATA_SHA = "24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416"  # pragma: allowlist secret - public original validation checksum
SPLITS = {
    "validation": (DATA_SHA, 4401),
    "train": (
        "bbfadf5ed1278ba2208c837fdcad865adf65f5df55d80abadab2745db13fcb5e",  # pragma: allowlist secret - original train file SHA
        84437,
    ),
}
TARGETS = [
    "c8dc7c16-e493-4078-bdc7-368b24476ca9",
    "48ac2156-f823-4e97-81ab-a66354549f59",
    "3f509e81-40ff-47dc-a677-4cb4a1623d7b",
]


def selected_targets(explicit, split="validation"):
    if split != "validation" and explicit is None:
        raise ValueError("train requires explicit original target IDs")
    targets = list(TARGETS if explicit is None else explicit)
    if not targets or any(not target.strip() for target in targets):
        raise ValueError("at least one nonempty original target ID is required")
    if len(set(targets)) != len(targets):
        raise ValueError("duplicate target IDs are not independent tasks")
    return targets


def validate_endpoint(endpoint):
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("only local /v1 model endpoints permitted")
    return endpoint


def post(url, payload, *, timeout=300):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--split", choices=tuple(SPLITS), default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument(
        "--target-message-id",
        action="append",
        help="Repeat for preselected original assistant IDs; default is historical three. No filtering or failed-target replacement.",
    )
    args = parser.parse_args()
    try:
        targets = selected_targets(args.target_message_id, args.split)
    except ValueError as error:
        parser.error(str(error))
    endpoint = validate_endpoint(args.endpoint.rstrip("/"))
    expected_sha, expected_rows = SPLITS[args.split]
    if hashlib.sha256(args.data_file.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("original OASST split file identity mismatch")
    import pyarrow.parquet as pq
    from vllm_hust_workload_data import oasst_history_prefix

    rows = pq.read_table(args.data_file).to_pylist()
    if len(rows) != expected_rows:
        raise ValueError("original OASST split row count mismatch")
    views = [oasst_history_prefix(rows, target) for target in targets]
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "OASST1",
        "selected_target_message_ids": targets,
        "selection": "explicit original target IDs, fixed before execution; complete original ancestors",
        "split": args.split,
        "source_sha256": expected_sha,
        "model": args.model,
        "status": "STARTED",
        "samples": [],
        "quality_scored": False,
        "live_episode_validated": False,
        "full_suite_validated": False,
        "performance_claim": False,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save():
        path = args.output / "receipt.json"
        path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        path.chmod(0o600)

    save()
    for view in views:
        payload = {
            "model": args.model,
            "messages": view["messages"],
            "temperature": 0,
            "max_tokens": 2048,
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tokens = post(
            endpoint.removesuffix("/v1") + "/tokenize",
            {
                k: v
                for k, v in payload.items()
                if k in {"model", "messages", "chat_template_kwargs"}
            }
            | {"add_generation_prompt": True},
        )
        count = tokens.get("count", len(tokens.get("tokens", [])))
        if not isinstance(count, int) or count <= 0 or count + 2048 > 131072:
            raise ValueError("full original input cannot fit declared context budget")
        result = post(endpoint + "/chat/completions", payload)
        choice = result["choices"][0]
        if (
            choice["finish_reason"] not in {"stop", "length"}
            or not isinstance(choice["message"].get("content"), str)
            or not choice["message"]["content"]
            or result["usage"]["prompt_tokens"] != count
        ):
            raise ValueError("response integrity check failed")
        private = args.output / (view["target_message_id"] + ".private.json")
        private.write_text(
            json.dumps({"request": payload, "response": result}, ensure_ascii=False),
            encoding="utf-8",
        )
        private.chmod(0o600)
        receipt["samples"].append(
            {
                "target_message_id": view["target_message_id"],
                "finish_reason": choice["finish_reason"],
                "usage": result["usage"],
                "messages_sha256": hashlib.sha256(
                    json.dumps(
                        view["messages"], ensure_ascii=False, sort_keys=True
                    ).encode()
                ).hexdigest(),
            }
        )
        save()
    if hashlib.sha256(args.data_file.read_bytes()).hexdigest() != expected_sha:
        raise ValueError("original file changed during execution")
    receipt["status"] = "ORIGINAL_HISTORY_GENERATED_NO_QUALITY_SCORE"
    receipt["all_normal_stop"] = all(
        s["finish_reason"] == "stop" for s in receipt["samples"]
    )
    save()
    print(json.dumps({"status": receipt["status"], "samples": len(receipt["samples"])}))


if __name__ == "__main__":
    main()
