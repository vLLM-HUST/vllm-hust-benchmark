"""Explicit local model transport for the author's OpenAI-chat protocol.

This is not the original default Claude backend or an Agent runner. It preserves
the original single-user prompt, 500-token/0.2 defaults, Observation stop, and
explicit edit/fast overrides. It never aliases a local model to GPT or Claude.
"""

import json
import math
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def original_chat_request(
    prompt,
    model,
    *,
    max_tokens_to_sample=500,
    temperature=0.2,
    stop_sequences=None,
):
    if not isinstance(prompt, str) or not isinstance(model, str) or not model.strip():
        raise ValueError(
            "original text prompt and explicit actual served model required"
        )
    if type(max_tokens_to_sample) is not int or max_tokens_to_sample <= 0:
        raise ValueError("positive original generation budget required")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("finite original temperature required")
    stops = ["Observation:"] if stop_sequences is None else stop_sequences
    if not isinstance(stops, list) or any(
        not isinstance(s, str) or not s for s in stops
    ):
        raise ValueError("original nonempty stop strings required")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens_to_sample,
        "temperature": temperature,
        "stop": stops or None,
    }


def complete_original_chat(
    prompt,
    model,
    *,
    api_base,
    allowed_model,
    log_file=None,
    max_tokens_to_sample=500,
    temperature=0.2,
    stop_sequences=None,
    timeout=600,
    opener=urlopen,
):
    endpoint = urlsplit(api_base)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
        or endpoint.path.rstrip("/") != "/v1"
    ):
        raise ValueError("owned loopback OpenAI-compatible /v1 service required")
    if model != allowed_model:
        raise ValueError("request must name the explicitly selected actual model")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive finite transport timeout required")
    body = original_chat_request(
        prompt,
        model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=temperature,
        stop_sequences=stop_sequences,
    )
    request = Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    # No transparent retry: original Agent owns its retry/step policy. Transport
    # errors propagate, preserving the failed task instead of resubmitting it.
    with opener(request, timeout=timeout) as response:
        result = json.loads(response.read())
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("one original completion required")
    completion = choices[0].get("message", {}).get("content")
    if not isinstance(completion, str):
        raise ValueError("model response must contain actual text content")
    if log_file is not None:
        receipt = {
            "request": body,
            "response": result,
            "usage_kind": "actual_service_usage_when_present_not_cl100k_estimate",
            "monetary_cost": None,
            "protocol": "original_openai_chat_with_explicit_local_model",
        }
        with Path(str(log_file) + ".local-transport.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    return completion
