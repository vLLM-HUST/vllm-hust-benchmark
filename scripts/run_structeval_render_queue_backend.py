"""Process one retained render request using the owned isolated controller.

No model calls or host execution of generated code. Default is plan-only.
An interrupted controller directory must be inspected, never blindly restarted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from run_structeval_matplotlib_isolated import run as run_isolated
from run_structeval_matplotlib_worker import RENDERER_SHA256
from structeval_render_queue import publish_new


def process(request_path, *, renderer, work_root, image, timeout=120, execute=False):
    request_path = Path(request_path)
    if (
        request_path.is_symlink()
        or not request_path.is_file()
        or request_path.stat().st_size > 32 * 1024**2
    ):
        raise ValueError("bounded regular private request required")
    raw = request_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    identity = digest[:32]
    if request_path.name != identity + ".request.json":
        raise ValueError("request filename/content identity mismatch")
    request = json.loads(raw)
    code = request.get("parsed_code")
    if (
        request.get("schema_version") != "structeval-render-request/v1"
        or not isinstance(code, str)
        or not isinstance(request.get("task_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["task_id"])
        or request.get("image") != image
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        or request.get("original_renderer_sha256") != RENDERER_SHA256
        or len(code.encode("utf-8")) > 16 * 1024**2
        or hashlib.sha256(code.encode("utf-8")).hexdigest()
        != request.get("parsed_code_sha256")
    ):
        raise ValueError("original request provenance mismatch")
    if not execute:
        return {"status": "PLAN_ONLY", "job_id": identity, "model_calls": 0}
    work_root = Path(work_root).resolve(strict=True)
    if not work_root.is_dir():
        raise ValueError("existing private work root required")
    code_path = work_root / (identity + ".parsed-code.txt")
    output = work_root / (identity + ".controller")
    # Atomic publication retains input. A partial previous controller run is not
    # proof of termination; operator must inspect its actual container handles.
    publish_new(code_path, code.encode("utf-8"))
    if output.exists():
        receipt = output / "controller-receipt.json"
        if not receipt.is_file() or receipt.is_symlink():
            raise ValueError("inspect retained incomplete controller; do not restart")
        result = json.loads(receipt.read_bytes())
        if result.get("status") != "ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING":
            raise ValueError("inspect retained blocked controller; do not restart")
    else:
        result = run_isolated(
            SimpleNamespace(
                image=image,
                task_id=request["task_id"],
                renderer=Path(renderer),
                code_file=code_path,
                code_sha256=request["parsed_code_sha256"],
                output=output,
                timeout=timeout,
                execute=True,
            )
        )
    if (
        result.get("task_id") != request["task_id"]
        or result.get("image") != image
        or result.get("parsed_code_sha256") != request["parsed_code_sha256"]
    ):
        raise ValueError("retained controller provenance mismatch")
    if result.get("status") == "ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING":
        if type(result.get("render_score")) is not int or result[
            "render_score"
        ] not in (0, 1):
            raise ValueError("invalid original renderer result")
        if result["render_score"] == 1:
            png = output / (request["task_id"] + ".png")
            if (
                png.is_symlink()
                or not png.is_file()
                or png.stat().st_size > 64 * 1024**2
                or hashlib.sha256(png.read_bytes()).hexdigest()
                != result.get("png_sha256")
            ):
                raise ValueError("retained original image missing or changed")
            publish_new(
                request_path.parent / (identity + ".response.png"), png.read_bytes()
            )
    response = {
        "schema_version": "structeval-render-response/v1",
        "request_sha256": digest,
        "task_id": request["task_id"],
        "controller_result": result,
    }
    publish_new(
        request_path.parent / (identity + ".response.json"),
        json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )
    return {"status": result["status"], "job_id": identity, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            process(
                args.request,
                renderer=args.renderer,
                work_root=args.work_root,
                image=args.image,
                timeout=args.timeout,
                execute=args.execute,
            )
        )
    )


if __name__ == "__main__":
    main()
