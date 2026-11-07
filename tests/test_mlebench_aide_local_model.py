"""Isolated local binding contract, not a real model request."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mlebench_aide_local_model.py"
BIND = runpy.run_path(str(SCRIPT))["bind_local_openai"]


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:1234/v1",
        "http://127.0.0.1/v1",
        "http://127.0.0.1:1234/v1?secret=anything",
        "http://user@127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://external.example:1234/v1",
    ],
)
def test_reject_nonliteral_loopback_endpoint(endpoint):
    with pytest.raises(ValueError, match="IPv4 loopback"):
        BIND(None, model="Qwen/example", api_base=endpoint)


def test_bind_preserves_openai_query_and_blocks_every_other_model(monkeypatch):
    query = object()
    original = SimpleNamespace(_client=None, query=query)
    backend = SimpleNamespace(
        backend_openai=original,
        provider_to_query_func={"openai": query, "openrouter": object()},
    )

    def sdk(**options):
        return SimpleNamespace(**options)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=sdk))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=sdk))
    BIND(backend, model="Qwen/example", api_base="http://127.0.0.1:1234/v1")
    assert backend.determine_provider("Qwen/example") == "openai"
    assert backend.provider_to_query_func["openai"] is query
    assert original._client.max_retries == 0
    assert not original._client.http_client.trust_env
    assert not original._client.http_client.follow_redirects
    with pytest.raises(ValueError, match="outside"):
        backend.determine_provider("gpt-other")
    with pytest.raises(RuntimeError, match="before"):
        BIND(backend, model="Qwen/example", api_base="http://127.0.0.1:1234/v1")
