"""CPU-render dispatch for the unchanged author Matplotlib callable contract.

Only transfers already parsed code and returned image bytes. Never executes code,
replaces prompts/questions, calls models, or treats worker failures as model0.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import time

from run_structeval_matplotlib_worker import RENDERER_SHA256


class RenderBlocked(RuntimeError):
    """Native outer runner must stop before judging any resource-failed tasks."""


def publish_new(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("existing queue artifact differs; preserve evidence")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
    temporary.replace(path)


class MatplotlibRenderQueue:
    def __init__(self, root, *, image_root, image, timeout=600, poll_interval=0.2):
        self.root = Path(root).resolve(strict=True)
        self.image_root = Path(image_root).resolve()
        if (
            not self.root.is_dir()
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
            or timeout <= 0
            or poll_interval <= 0
        ):
            raise ValueError(
                "private existing queue, pinned image and positive bounds required"
            )
        self.image = image
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.failures = []
        self.receipts = []

    def render(self, task_id, python_code, img_output_path):
        """Same arguments/result as original render_matplotlib_and_screenshot."""
        try:
            return self._render(task_id, python_code, img_output_path)
        except RenderBlocked:
            raise
        except Exception as error:
            self.failures.append(
                {
                    "task_id": task_id,
                    "kind": "RENDER_DISPATCH_ERROR",
                    "error_type": type(error).__name__,
                }
            )
            raise RenderBlocked(
                "retained render dispatch failure; stop before VQA"
            ) from error

    def _render(self, task_id, parsed_code, img_output_path):
        if (
            not isinstance(task_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", task_id)
            or not isinstance(parsed_code, str)
        ):
            raise ValueError("unchanged original task ID and parsed-code text required")
        if Path(img_output_path).resolve() != self.image_root:
            raise ValueError("original workflow image destination differs")
        code = parsed_code.encode("utf-8")
        if len(code) > 16 * 1024**2:
            raise ValueError(
                "original parsed code exceeds worker resource bound; do not truncate"
            )
        request = {
            "schema_version": "structeval-render-request/v1",
            "task_id": task_id,
            "image": self.image,
            "original_renderer_sha256": RENDERER_SHA256,
            "parsed_code": parsed_code,
            "parsed_code_sha256": hashlib.sha256(code).hexdigest(),
        }
        raw = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        identity = digest[:32]
        request_path = self.root / (identity + ".request.json")
        response_path = self.root / (identity + ".response.json")
        cached = response_path.exists()
        publish_new(request_path, raw)
        deadline = time.monotonic() + self.timeout
        while not response_path.exists():
            if time.monotonic() >= deadline:
                self.failures.append(
                    {
                        "task_id": task_id,
                        "job_id": identity,
                        "kind": "RENDER_TRANSPORT_TIMEOUT_RETAINED",
                    }
                )
                raise RenderBlocked(
                    "render job timed out; retain same identity, do not replace task"
                )
            time.sleep(self.poll_interval)
        response = json.loads(response_path.read_bytes())
        if (
            response.get("schema_version") != "structeval-render-response/v1"
            or response.get("request_sha256") != digest
            or response.get("task_id") != task_id
        ):
            self.failures.append(
                {
                    "task_id": task_id,
                    "job_id": identity,
                    "kind": "RENDER_RESPONSE_IDENTITY_MISMATCH",
                }
            )
            raise RenderBlocked(
                "returned render artifact does not match original input"
            )
        result = response.get("controller_result", {})
        if result.get("status") != "ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING":
            self.failures.append(
                {
                    "task_id": task_id,
                    "job_id": identity,
                    "kind": "WORKER_OR_RESOURCE_BLOCKED",
                }
            )
            raise RenderBlocked(
                "do not convert render resource or controller failure into model0"
            )
        score = result.get("render_score")
        if (
            type(score) is not int
            or score not in (0, 1)
            or result.get("parsed_code_sha256") != request["parsed_code_sha256"]
            or result.get("image") != self.image
        ):
            self.failures.append(
                {
                    "task_id": task_id,
                    "job_id": identity,
                    "kind": "RENDER_RESULT_CONTRACT_MISMATCH",
                }
            )
            raise RenderBlocked("original renderer result/provenance mismatch")
        if score == 1:
            image = self.root / (identity + ".response.png")
            if (
                not image.is_file()
                or image.is_symlink()
                or image.stat().st_size > 64 * 1024**2
                or hashlib.sha256(image.read_bytes()).hexdigest()
                != result.get("png_sha256")
            ):
                self.failures.append(
                    {
                        "task_id": task_id,
                        "job_id": identity,
                        "kind": "RENDER_IMAGE_IDENTITY_MISMATCH",
                    }
                )
                raise RenderBlocked("original renderer image bytes missing or changed")
            self.image_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            publish_new(self.image_root / (task_id + ".png"), image.read_bytes())
        elif (self.image_root / (task_id + ".png")).exists():
            raise ValueError(
                "stale image exists for failed original render; preserve it and inspect run identity"
            )
        self.receipts.append(
            {
                "task_id": task_id,
                "job_id": identity,
                "retained_result_reused": cached,
                "original_render_score": score,
            }
        )
        return score

    def require_ready_for_judge(self):
        if self.failures:
            raise RenderBlocked(
                "original renderer failures may be swallowed upstream; stop before VQA"
            )
