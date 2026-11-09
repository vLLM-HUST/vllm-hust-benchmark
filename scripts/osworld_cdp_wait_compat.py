"""Opt-in slow-VM CDP connection wait; never substitute browser state or scores."""

from urllib.parse import urlsplit


class _Proxy:
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)


def cdp_wait_playwright(original_factory, *, endpoint, timeout_ms=180000):
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or type(timeout_ms) is not int
        or not 15000 < timeout_ms <= 300000
    ):
        raise ValueError(
            "explicit local CDP endpoint and bounded transport wait required"
        )

    class BrowserType(_Proxy):
        def connect_over_cdp(self, target, *args, **kwargs):
            if target == endpoint and kwargs.get("timeout") == 15000:
                kwargs["timeout"] = timeout_ms
            return self.original.connect_over_cdp(target, *args, **kwargs)

    class Playwright(_Proxy):
        @property
        def chromium(self):
            return BrowserType(self.original.chromium)

    class Context(_Proxy):
        def __enter__(self):
            return Playwright(self.original.__enter__())

        def __exit__(self, *args):
            return self.original.__exit__(*args)

    return lambda: Context(original_factory())
