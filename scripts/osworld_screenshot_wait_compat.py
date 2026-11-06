"""Opt-in transport compatibility for an owned slow ARM/TCG desktop.

One adapter extends screenshot reads; another uses a genuine platform response
for setup's status-only probe. Task observations, retries, budgets and grading
are preserved. This functional-validation profile has no original timing claim.
"""

import math
from urllib.parse import urlsplit


def _loopback_controller(controller_base):
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
    return controller_base.rstrip("/")


def screenshot_wait_requests(original, *, controller_base, read_timeout=300):
    controller_base = _loopback_controller(controller_base)
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


def setup_readiness_requests(original, *, controller_base, screenshot_only=False):
    """Use a real platform response for SetupController's status-only probe.

    Install only in the setup module, not in the observation controller. No
    terminal body is fabricated or substituted for a task observation.
    """
    if screenshot_only is not True:
        raise ValueError("explicit screenshot-only validation profile required")
    base = _loopback_controller(controller_base)

    class SetupReadinessRequests:
        def get(self, url, *args, **kwargs):
            if url == base + "/terminal" and kwargs.get("timeout") == 60:
                url = base + "/platform"
            return original.get(url, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(original, name)

    return SetupReadinessRequests()
