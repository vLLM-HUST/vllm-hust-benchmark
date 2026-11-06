"""Explicit read-wait compatibility for the owned slow ARM/TCG screenshot API.

Only the original controller's10s GET /screenshot wait changes. Preserve original
image bytes, retries, requests to other endpoints, task budgets and scoring.
This is a disclosed functional-validation profile, not original timing fidelity.
"""

import math
from urllib.parse import urlsplit


def screenshot_wait_requests(original, *, controller_base, read_timeout=300):
    parsed = urlsplit(controller_base)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.port is None
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("explicit owned loopback controller base required")
    if (
        type(read_timeout) not in {int, float}
        or not math.isfinite(read_timeout)
        or not 10 <= read_timeout <= 300
    ):
        raise ValueError(
            "finite screenshot read wait between10 and300 seconds required"
        )
    target = controller_base.rstrip("/") + "/screenshot"

    class ScreenshotWaitRequests:
        def get(self, url, *args, **kwargs):
            if url == target and kwargs.get("timeout") == 10:
                kwargs = dict(kwargs, timeout=read_timeout)
            return original.get(url, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(original, name)

    return ScreenshotWaitRequests()
