"""One original Python-tool call across an isolated Agent/tool boundary.

No generated code executes here. An external owned worker must run the request
with discoverybench_isolated_python.py and preserve its exact string output.
"""

import hashlib
import json
from pathlib import Path
import time

from appworld_model_transport import save_new


LIMIT_MESSAGE = (
    "You cannot run the code more than once - you have already run it earlier. "
    "Please provide the final answer based on whatever information you got till now. "
    "Do not attempt to run code again."
)


def budgeted_python_run(exchange):
    """Bind to the original class; preserve its max_turns counter and limit text."""

    def run(self, query, run_manager=None):
        if self.max_turns >= 1:
            return LIMIT_MESSAGE
        self.max_turns += 1
        return exchange(query)

    return run


def request_once(queue, code, *, timeout=1800):
    if not isinstance(code, str) or timeout <= 0:
        raise ValueError("original tool code and positive operational timeout required")
    queue = Path(queue).resolve(strict=True)
    if not queue.is_dir():
        raise ValueError("dedicated existing per-task tool queue required")
    identity = hashlib.sha256(code.encode()).hexdigest()
    request = queue / "original-python.request.json"
    response = queue / "original-python.response.json"
    if (
        request.exists()
        or request.with_suffix(".json.partial").exists()
        or response.exists()
    ):
        raise RuntimeError(
            "existing original tool call; inspect saved state, never resubmit"
        )
    save_new(request, json.dumps({"code": code, "code_sha256": identity}).encode())
    deadline = time.monotonic() + timeout
    while not response.exists():
        if (queue / "STOP").exists() or time.monotonic() >= deadline:
            raise TimeoutError(
                "tool observation stopped; retained request must not be resubmitted"
            )
        time.sleep(0.1)
    if response.is_symlink():
        raise ValueError("tool response must not be a symlink")
    payload = json.loads(response.read_bytes())
    if payload.get("code_sha256") != identity:
        raise ValueError("tool response does not match original generated code")
    if payload.get("status") != "ORIGINAL_TOOL_RETURNED":
        raise RuntimeError(
            "isolated tool infrastructure failed; retain original failure"
        )
    if not isinstance(payload.get("tool_output"), str):
        raise ValueError("exact original tool string result required")
    return payload["tool_output"]
