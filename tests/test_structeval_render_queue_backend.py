"""Backend IPC fixtures; no Docker, original tasks or generated-code execution."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_structeval_render_queue_backend as backend
from structeval_render_queue import (
    CanvasRenderQueue,
    HtmlRenderQueue,
    MarkdownRenderQueue,
    MatplotlibRenderQueue,
    SvgRenderQueue,
    TypstRenderQueue,
)


@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize(
    "kind", ["matplotlib", "typst", "svg", "html", "canvas", "markdown"]
)
def test_complete_dispatch_and_retained_resume_without_execution(
    tmp_path, monkeypatch, score, kind
):
    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    image = "sha256:" + "0" * 64
    queue_class = {
        "typst": TypstRenderQueue,
        "matplotlib": MatplotlibRenderQueue,
        "svg": SvgRenderQueue,
        "html": HtmlRenderQueue,
        "canvas": CanvasRenderQueue,
        "markdown": MarkdownRenderQueue,
    }[kind]
    queue = queue_class(
        queue_root,
        image_root=tmp_path / "images",
        image=image,
        timeout=5,
        poll_interval=0.01,
    )
    calls = []

    def fake(args):
        calls.append(args)
        assert args.code_file.read_bytes() == "# 原解析内容\n".encode()
        args.output.mkdir()
        result = dict(
            status="ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING",
            task_id=args.task_id,
            image=image,
            parsed_code_sha256=args.code_sha256,
            render_score=score,
            renderer_kind=kind,
        )
        if score:
            raw = b"inert IPC image fixture"
            (args.output / (args.task_id + ".png")).write_bytes(raw)
            result["png_sha256"] = hashlib.sha256(raw).hexdigest()
        (args.output / "controller-receipt.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(backend, "run_isolated", fake)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            queue.render, "000802", "# 原解析内容\n", tmp_path / "images"
        )
        deadline = time.monotonic() + 3
        while not list(queue_root.glob("*.request.json")):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        request = next(queue_root.glob("*.request.json"))
        kwargs = dict(renderer=tmp_path / "unused", work_root=work, image=image)
        assert backend.process(request, **kwargs)["status"] == "PLAN_ONLY"
        assert not calls
        backend.process(request, execute=True, **kwargs)
        assert future.result() == score
    response = next(queue_root.glob("*.response.json")).read_bytes()
    backend.process(request, execute=True, **kwargs)
    assert len(calls) == 1
    assert next(queue_root.glob("*.response.json")).read_bytes() == response
    queue.require_ready_for_judge()
    # A retained incomplete run must never trigger a second container launch.
    receipt = calls[0].output / "controller-receipt.json"
    receipt.write_text(json.dumps({"status": "PREPARING_PRIVATE_COMPONENT"}))
    with pytest.raises(ValueError, match="do not restart"):
        backend.process(request, execute=True, **kwargs)
    assert len(calls) == 1


def test_original_typst_score1_without_named_png_keeps_native_missing_image_rule(
    tmp_path, monkeypatch
):
    queue_root = tmp_path / "queue"
    queue_root.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    image = "sha256:" + "0" * 64
    queue = TypstRenderQueue(
        queue_root,
        image_root=tmp_path / "images",
        image=image,
        timeout=5,
        poll_interval=0.01,
    )

    def fake(args):
        assert args.renderer_kind == "typst"
        args.output.mkdir()
        result = dict(
            status="ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING",
            task_id=args.task_id,
            image=image,
            parsed_code_sha256=args.code_sha256,
            renderer_kind="typst",
            render_score=1,
            image_present=False,
            png_valid=False,
        )
        (args.output / "controller-receipt.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(backend, "run_isolated", fake)
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            queue.render, "typst_fixture", "inert-not-executed", tmp_path / "images"
        )
        deadline = time.monotonic() + 3
        while not list(queue_root.glob("*.request.json")):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        backend.process(
            next(queue_root.glob("*.request.json")),
            renderer=tmp_path / "unused",
            work_root=work,
            image=image,
            execute=True,
        )
        assert future.result() == 1
    assert not (tmp_path / "images/typst_fixture.png").exists()
    queue.require_ready_for_judge()


def test_content_identity_checked_before_any_execution(tmp_path, monkeypatch):
    request = tmp_path / ("0" * 32 + ".request.json")
    request.write_text("{}")

    def forbidden(*args):
        pytest.fail("must not execute")

    monkeypatch.setattr(backend, "run_isolated", forbidden)
    with pytest.raises(ValueError, match="identity mismatch"):
        backend.process(
            request,
            renderer=tmp_path,
            work_root=tmp_path,
            image="sha256:" + "0" * 64,
            execute=True,
        )
