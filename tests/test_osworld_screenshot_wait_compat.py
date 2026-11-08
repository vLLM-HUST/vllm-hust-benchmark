"""Transport-call fixtures; no real screenshots, desktops or model calls."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from osworld_screenshot_wait_compat import (
    screenshot_wait_requests,
    setup_readiness_requests,
)


def test_changes_only_declared_screenshot_timeout_and_returns_original_object():
    calls = []
    response = object()

    def get(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    original = SimpleNamespace(get=get, post=object())
    adapted = screenshot_wait_requests(
        original, controller_base="http://127.0.0.1:15000"
    )
    for url, timeout, expected in [
        ("http://127.0.0.1:15000/screenshot", 10, 300),
        ("http://127.0.0.1:15000/screenshot", 20, 20),
        ("http://127.0.0.1:15000/platform", 10, 10),
        ("http://127.0.0.1:19466/screenshot", 10, 10),
    ]:
        assert (
            adapted.get(url, timeout=timeout, headers={"fixture": "unchanged"})
            is response
        )
        assert calls[-1] == (
            (url,),
            {"timeout": expected, "headers": {"fixture": "unchanged"}},
        )
    assert adapted.post is original.post


@pytest.mark.parametrize("wait", [True, 0, 301, float("nan")])
def test_invalid_wait_rejected(wait):
    with pytest.raises(ValueError, match="read wait"):
        screenshot_wait_requests(
            object(), controller_base="http://localhost:15000", read_timeout=wait
        )


@pytest.mark.parametrize(
    "base",
    [
        "https://external.example",
        "http://localhost:15000?x=1",
        "http://localhost/path",
        "http://localhost",
    ],
)
def test_ambiguous_or_external_controller_rejected(base):
    with pytest.raises(ValueError, match="loopback"):
        screenshot_wait_requests(object(), controller_base=base)


def test_setup_probe_uses_genuine_platform_response_without_changing_observations():
    calls = []
    response = SimpleNamespace(status_code=503)

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    adapted = setup_readiness_requests(
        SimpleNamespace(get=get),
        controller_base="http://127.0.0.1:15000",
        screenshot_only=True,
    )
    assert adapted.get("http://127.0.0.1:15000/terminal", timeout=60) is response
    assert calls[-1] == ("http://127.0.0.1:15000/platform", {"timeout": 60})
    assert response.status_code == 503  # No fabricated successful health response.
    adapted.get("http://127.0.0.1:15000/terminal", timeout=10)
    assert calls[-1][0].endswith("/terminal")
    adapted.get("http://127.0.0.1:19466/terminal", timeout=60)
    assert calls[-1][0] == "http://127.0.0.1:19466/terminal"


def test_setup_probe_requires_declared_nonterminal_profile():
    with pytest.raises(ValueError, match="screenshot-only"):
        setup_readiness_requests(object(), controller_base="http://localhost:15000")
