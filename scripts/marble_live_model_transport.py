"""Task-scoped loopback forwarding; no orphan replay or model implementation.

The context exists only inside the original MARBLE child. Every invocation gets
a fresh queue. Its worker stops dispatching when the original child returns;
an in-flight response is retained before shutdown. Nothing restarts a model.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

from appworld_model_transport import model_proxy, process_one


def loopback_endpoint(value):
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.port is None
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ValueError("explicit owned loopback /v1 endpoint required")
    return value.rstrip("/")


@contextmanager
def live_model_transport(queue, api_base, *, port=14005, timeout=600):
    """Preserve original requests; only the disclosed nonthinking binding is added.

    This is an HTTP transport, not proof of a NPU-backed service. The operator
    must qualify the bound backend and resource window before native execution.
    Existing or uncertain queues are rejected, never resumed implicitly.
    """
    api_base = loopback_endpoint(api_base)
    if port and urlsplit(api_base).port == port:
        raise ValueError("backend must not point back to the model proxy")
    if timeout <= 0:
        raise ValueError("positive model transport timeout required")
    queue = Path(queue)
    if queue.exists() or queue.is_symlink():
        raise ValueError("fresh task-scoped model queue required; no orphan replay")
    queue.mkdir(mode=0o700)
    stop = threading.Event()
    errors = []

    def serve():
        try:
            while not stop.is_set():
                if not process_one(
                    queue,
                    api_base,
                    disable_thinking=True,
                    timeout=timeout,
                ):
                    stop.wait(0.05)
        except Exception as error:
            errors.append(error)
            with (queue / "worker-error-private.json").open("x") as stream:
                json.dump({"type": type(error).__name__, "message": str(error)}, stream)
            stop.set()

    with model_proxy(queue, port=port, timeout=timeout) as actual_port:
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            yield actual_port
        finally:
            stop.set()
            worker.join(timeout + 1)
            if worker.is_alive():
                raise RuntimeError(
                    "model transport still in flight; inspect claim, never resubmit"
                )
        if errors:
            raise RuntimeError(
                "model transport failed; retained private error"
            ) from errors[0]
