"""Disclosed local transport for the author's Anthropic Messages protocol.

Preserve the original user prompt/messages, 2000-token/0.5 defaults, Human and
Observation stops, and explicit edit/fast overrides. No Claude model aliases,
prompt repair, retry, scientific code execution or scoring are provided here.
"""

import json
import argparse
import io
import math
import os
from pathlib import Path
import time
import uuid
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from appworld_model_transport import save_new


def queue_opener(queue):
    queue = Path(queue).resolve(strict=True)
    if not queue.is_dir():
        raise ValueError("dedicated private Messages queue required")

    def open_request(request, timeout):
        if urlsplit(request.full_url).path != "/v1/messages":
            raise ValueError("only original Messages requests accepted")
        raw = request.data
        json.loads(raw)
        identity = uuid.uuid4().hex
        save_new(queue / (identity + ".request.json"), raw)
        result = queue / (identity + ".response.json")
        deadline = time.monotonic() + timeout
        while not result.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Retained Messages job timed out; no resubmission")
            time.sleep(0.1)
        payload = json.loads(result.read_bytes())
        wire = io.BytesIO(json.dumps(payload["response"]).encode())
        if payload["http_status"] >= 400:
            raise HTTPError(
                request.full_url,
                payload["http_status"],
                "Original service error",
                {},
                wire,
            )
        return wire

    return open_request


def process_messages_one(
    queue, api_base, *, model, timeout=600, disable_thinking=False
):
    api_base = local_endpoint(api_base)
    queue = Path(queue).resolve(strict=True)
    for path in sorted(
        queue.glob("*.request.json"), key=lambda p: p.stat().st_mtime_ns
    ):
        identity = path.name.removesuffix(".request.json")
        if (
            len(identity) != 32
            or any(c not in "0123456789abcdef" for c in identity)
            or path.is_symlink()
        ):
            raise ValueError("invalid private queue identity")
        result = queue / (identity + ".response.json")
        if result.exists():
            continue
        try:
            claim = (queue / (identity + ".claim")).open("xb")
        except FileExistsError:
            continue
        with claim:
            uid = path.stat().st_uid
            if os.fstat(claim.fileno()).st_uid not in {0, uid}:
                raise ValueError("queue worker ownership differs")
            raw = path.read_bytes()
            body = json.loads(raw)
            if body.get("model") != model:
                raise ValueError("actual selected model differs")
            if disable_thinking:
                if body.get("thinking") not in (None, {"type": "disabled"}):
                    raise ValueError(
                        "do not override conflicting original thinking setting"
                    )
                options = body.setdefault("chat_template_kwargs", {})
                if not isinstance(options, dict) or options.get(
                    "enable_thinking"
                ) not in (None, False):
                    raise ValueError(
                        "do not override conflicting original template thinking setting"
                    )
                options["enable_thinking"] = False
                forwarded = json.dumps(body).encode()
            else:
                forwarded = raw
            save_new(queue / (identity + ".forwarded-request.json"), forwarded)
            request = Request(
                api_base + "/messages",
                data=forwarded,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
            try:
                response = urlopen(request, timeout=timeout)
            except HTTPError as error:
                response = error
            with response:
                status, wire = response.code, response.read()
            save_new(queue / (identity + ".response.raw"), wire)
            save_new(
                result,
                json.dumps(
                    dict(http_status=status, response=json.loads(wire))
                ).encode(),
                owner_uid=uid,
            )
            return True
    return False


def original_messages_request(
    prompt,
    model,
    *,
    max_tokens_to_sample=2000,
    temperature=0.5,
    stop_sequences=None,
    messages=None,
):
    if not isinstance(prompt, str) or not isinstance(model, str) or not model.strip():
        raise ValueError("original prompt and actual served model required")
    if type(max_tokens_to_sample) is not int or max_tokens_to_sample < 1:
        raise ValueError("positive original output budget required")
    if type(temperature) not in {int, float} or not math.isfinite(temperature):
        raise ValueError("finite original temperature required")
    stops = ["\n\nHuman:", "Observation:"] if stop_sequences is None else stop_sequences
    if not isinstance(stops, list) or any(
        not isinstance(x, str) or not x for x in stops
    ):
        raise ValueError("original stop strings required")
    if messages is not None and (not isinstance(messages, list) or not messages):
        raise ValueError("explicit original messages must be a nonempty list")
    return dict(
        model=model,
        messages=[dict(role="user", content=prompt)] if messages is None else messages,
        max_tokens=max_tokens_to_sample,
        temperature=temperature,
        stop_sequences=stops,
    )


def local_endpoint(api_base):
    endpoint = urlsplit(api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.path.rstrip("/") != "/v1"
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError("owned loopback /v1 Messages service required")
    endpoint.port
    return api_base.rstrip("/")


def complete_original_messages(
    prompt,
    model,
    *,
    api_base,
    allowed_model,
    log_file=None,
    max_tokens_to_sample=2000,
    temperature=0.5,
    stop_sequences=None,
    messages=None,
    timeout=600,
    opener=urlopen,
):
    api_base = local_endpoint(api_base)
    if model != allowed_model:
        raise ValueError("actual selected model must not be aliased")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite transport timeout required")
    body = original_messages_request(
        prompt,
        model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=temperature,
        stop_sequences=stop_sequences,
        messages=messages,
    )
    request = Request(
        api_base.rstrip("/") + "/messages",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
    )
    with opener(request, timeout=timeout) as response:
        result = json.loads(response.read())
    content = result.get("content")
    text_blocks = (
        [
            block
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if isinstance(content, list)
        else []
    )
    if not text_blocks or not isinstance(text_blocks[0].get("text"), str):
        raise ValueError("original first text completion required")
    if log_file is not None:
        with Path(str(log_file) + ".local-messages-transport.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    dict(
                        request=body,
                        response=result,
                        protocol="original_anthropic_messages_with_explicit_local_model",
                    )
                )
                + "\n"
            )
    # Modern local Messages can prepend a separate reasoning block. Select the
    # first real text block, never reasoning, repaired text or fabricated fields.
    return text_blocks[0]["text"]


def main():
    parser = argparse.ArgumentParser(
        description="Trusted Messages queue worker; never execute scientific task code"
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disclosed independent local-model nonthinking profile; original prompt/budgets unchanged",
    )
    parser.add_argument("--idle-timeout", type=int, default=21600)
    args = parser.parse_args()
    if not 0 < args.idle_timeout <= 86400:
        parser.error("bounded positive idle timeout required")
    endpoint = local_endpoint(args.api_base)
    with urlopen(endpoint + "/models", timeout=30) as response:
        models = json.load(response)["data"]
    if not any(row["id"] == args.model for row in models):
        raise ValueError("actual served model absent")
    if args.disable_thinking:
        with urlopen(
            endpoint.removesuffix("/v1") + "/openapi.json", timeout=30
        ) as response:
            schema = json.load(response)
        fields = schema["components"]["schemas"]["AnthropicMessagesRequest"][
            "properties"
        ]
        if "chat_template_kwargs" not in fields:
            raise ValueError(
                "native Messages template control absent; no silently ignored thinking option"
            )
    last = time.monotonic()
    jobs = 0
    while not (args.queue / "STOP").exists():
        if process_messages_one(
            args.queue,
            endpoint,
            model=args.model,
            disable_thinking=args.disable_thinking,
        ):
            jobs += 1
            last = time.monotonic()
        elif not args.watch or time.monotonic() - last >= args.idle_timeout:
            break
        else:
            time.sleep(1)
    print(
        json.dumps(dict(processed_messages_jobs=jobs, source_checkbox_eligible=False))
    )


if __name__ == "__main__":
    main()
