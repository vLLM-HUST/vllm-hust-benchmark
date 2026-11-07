"""Explicit local-model respond interface for the original SAB chat baseline.

Preserves the original GPT-chat branch's2000 output budget and sampling fields,
without renaming the local model as GPT. This is only provider transport, not
ScienceAgent cost/tokenizer binding, native inference, scoring or delivery.
"""

import math
import re
from urllib.parse import urlsplit


class LocalScienceEngine:
    def __init__(self, model, api_base, *, disable_thinking=False, client=None):
        if not isinstance(api_base, str):
            raise ValueError("explicit owned loopback OpenAI endpoint required")
        endpoint = urlsplit(api_base)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.port is None
            or endpoint.username
            or endpoint.password
            or endpoint.path.rstrip("/") != "/v1"
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("explicit owned loopback OpenAI endpoint required")
        if not isinstance(model, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._/-]*", model
        ):
            raise ValueError("explicit real local model identifier required")
        if type(disable_thinking) is not bool:
            raise ValueError("explicit boolean thinking binding required")
        self.llm_engine_name = model
        self.disable_thinking = disable_thinking
        self._request_with_retry = self._request
        if client is None:
            import backoff
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                OpenAI,
                RateLimitError,
            )

            client = OpenAI(
                base_url=api_base.rstrip("/"),
                api_key="local-not-a-secret",  # pragma: allowlist secret - dummy loopback credential
                max_retries=10,
                timeout=120.0,
            )
            # Same SDK settings and outer exception backoff as the publisher.
            self._request_with_retry = backoff.on_exception(
                backoff.expo,
                (
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    InternalServerError,
                ),
            )(self._request)
        self.client = client

    def _request(self, messages, temperature, top_p):
        kwargs = {
            "model": self.llm_engine_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 2000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        if self.disable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return self.client.chat.completions.create(**kwargs)

    def respond(self, user_input, temperature, top_p):
        if (
            not isinstance(user_input, list)
            or not user_input
            or any(
                not isinstance(item, dict) or not isinstance(item.get("role"), str)
                for item in user_input
            )
        ):
            raise ValueError("unchanged original message list required")
        for value in (temperature, top_p):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("finite original sampling parameters required")
        response = self._request_with_retry(user_input, temperature, top_p)
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError(
                "one text response required; never promote hidden reasoning"
            )
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str):
            raise ValueError(
                "one text response required; never promote hidden reasoning"
            )
        # Empty text and length-finished answers remain model outputs, not
        # substituted completions or reasons to change the original budget.
        usage = getattr(response, "usage", None)
        counts = [
            getattr(usage, name, None)
            for name in ("prompt_tokens", "completion_tokens")
        ]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("actual nonnegative token usage required")
        return content, *counts
