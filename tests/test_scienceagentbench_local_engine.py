"""Transport fixtures only; not ScienceAgent or neural inference evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "science_local_engine",
    Path(__file__).resolve().parents[1] / "scripts/scienceagentbench_local_engine.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def fixture(response=None):
    calls = []
    response = response or SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="original output"),
                finish_reason="length",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=2000),
    )

    def create(**kwargs):
        calls.append(deepcopy(kwargs))
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, calls


def test_original_budget_sampling_and_messages_preserved_without_gpt_alias():
    client, calls = fixture()
    engine = entry.LocalScienceEngine(
        "Qwen/Qwen3-14B", "http://127.0.0.1:19468/v1", client=client
    )
    messages = [
        {"role": "user", "content": "  原始问题\n"},
        {"role": "assistant", "content": "history"},
    ]
    original = deepcopy(messages)
    assert engine.respond(messages, 0.2, 0.95) == ("original output", 12, 2000)
    assert messages == original
    assert calls == [
        {
            "model": "Qwen/Qwen3-14B",
            "messages": original,
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    ]


def test_thinking_mode_is_explicit_binding():
    client, calls = fixture()
    engine = entry.LocalScienceEngine(
        "Qwen/Qwen3-14B",
        "http://127.0.0.1:19468/v1",
        client=client,
        disable_thinking=True,
    )
    engine.respond([{"role": "user", "content": "fixture"}], 0.2, 0.95)
    assert calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "http://127.0.0.1/v1",
        "http://user:password@127.0.0.1:123/v1",  # pragma: allowlist secret - rejected synthetic credential fixture
        "http://127.0.0.1:123/wrong",
    ],
)
def test_no_implicit_external_or_credential_endpoint(url):
    client, _ = fixture()
    with pytest.raises(ValueError):
        entry.LocalScienceEngine("real-local-model", url, client=client)


def test_hidden_reasoning_not_substituted_for_missing_content():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning_content="hidden")
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=5),
    )
    client, _ = fixture(response)
    with pytest.raises(ValueError, match="hidden reasoning"):
        entry.LocalScienceEngine(
            "real-local-model", "http://127.0.0.1:123/v1", client=client
        ).respond([{"role": "user", "content": "x"}], 0.2, 0.95)


def test_empty_model_answer_is_retained():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
    )
    client, _ = fixture(response)
    assert entry.LocalScienceEngine(
        "real-local-model", "http://127.0.0.1:123/v1", client=client
    ).respond([{"role": "user", "content": "x"}], 0.2, 0.95) == ("", 1, 0)


def test_missing_usage_not_fabricated_as_zero():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="text"))], usage=None
    )
    client, _ = fixture(response)
    with pytest.raises(ValueError, match="actual"):
        entry.LocalScienceEngine(
            "real-local-model", "http://127.0.0.1:123/v1", client=client
        ).respond([{"role": "user", "content": "x"}], 0.2, 0.95)


def test_malformed_response_is_explicit_interface_failure():
    client, _ = fixture(SimpleNamespace(choices=None))
    with pytest.raises(ValueError, match="one text response"):
        entry.LocalScienceEngine(
            "real-local-model", "http://127.0.0.1:123/v1", client=client
        ).respond([{"role": "user", "content": "x"}], 0.2, 0.95)


def test_thin_sdk_pins_not_full_environment_claim():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/workload-native/scienceagentbench-sdk-client-requirements.txt"
    )
    assert [
        line
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ] == ["openai==1.54.4", "httpx==0.27.2", "backoff==2.2.1"]
