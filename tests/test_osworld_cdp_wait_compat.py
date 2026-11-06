from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from osworld_cdp_wait_compat import cdp_wait_playwright


def test_only_exact_original_cdp_wait_changes():
    calls = []
    browser = object()

    class Context:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(connect_over_cdp=connect))

        def __exit__(self, *args):
            calls.append(("exit", args))

    def connect(target, **kwargs):
        calls.append((target, kwargs))
        return browser

    factory = cdp_wait_playwright(Context, endpoint="http://127.0.0.1:19222")
    with factory() as playwright:
        assert (
            playwright.chromium.connect_over_cdp(
                "http://127.0.0.1:19222", timeout=15000
            )
            is browser
        )
        playwright.chromium.connect_over_cdp("http://127.0.0.1:19223", timeout=15000)
        playwright.chromium.connect_over_cdp("http://127.0.0.1:19222", timeout=5000)
    assert calls[:3] == [
        ("http://127.0.0.1:19222", {"timeout": 180000}),
        ("http://127.0.0.1:19223", {"timeout": 15000}),
        ("http://127.0.0.1:19222", {"timeout": 5000}),
    ]
    assert calls[3] == ("exit", (None, None, None))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com:9222",
        "http://127.0.0.1:9222/path",
        "https://127.0.0.1:9222",
        "http://u:p@127.0.0.1:9222",
    ],
)
def test_reject_nonlocal_or_ambiguous_endpoint(endpoint):
    with pytest.raises(ValueError):
        cdp_wait_playwright(lambda: None, endpoint=endpoint)
