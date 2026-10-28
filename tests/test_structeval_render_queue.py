"""IPC fixtures only; no original task, executable code, real PNG or model."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from structeval_render_queue import (
    MatplotlibRenderQueue,
    RenderBlocked,
    TypstRenderQueue,
)


def test_keyword_parameters_match_the_original_renderer(tmp_path):
    queue = MatplotlibRenderQueue(
        tmp_path, image_root=tmp_path / "images", image="sha256:" + "0" * 64
    )
    assert list(inspect.signature(queue.render).parameters) == [
        "task_id",
        "python_code",
        "img_output_path",
    ]


def test_typst_keyword_parameters_match_the_original_renderer(tmp_path):
    queue = TypstRenderQueue(
        tmp_path, image_root=tmp_path / "images", image="sha256:" + "0" * 64
    )
    assert list(inspect.signature(queue.render).parameters) == [
        "task_id",
        "typst_code",
        "img_output_path",
    ]


@pytest.mark.parametrize("score", [0, 1])
def test_original_code_result_and_cached_identity_preserved(tmp_path, score):
    queue = MatplotlibRenderQueue(
        tmp_path,
        image_root=tmp_path / "images",
        image="sha256:" + "0" * 64,
        timeout=5,
        poll_interval=0.01,
    )
    code = "# 中文字段原样保留\nnot_executed_fixture()\n"
    with ThreadPoolExecutor() as pool:
        future = pool.submit(queue.render, "000800", code, tmp_path / "images")
        deadline = time.monotonic() + 3
        while not list(tmp_path.glob("*.request.json")):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        path = next(tmp_path.glob("*.request.json"))
        raw = path.read_bytes()
        request = json.loads(raw)
        assert request["parsed_code"] == code
        assert (
            request["parsed_code_sha256"] == hashlib.sha256(code.encode()).hexdigest()
        )
        identity = path.name.removesuffix(".request.json")
        result = {
            "status": "ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING",
            "render_score": score,
            "parsed_code_sha256": request["parsed_code_sha256"],
            "image": request["image"],
        }
        if score:
            image = b"protocol-fixture-not-a-real-PNG"
            result["png_sha256"] = hashlib.sha256(image).hexdigest()
            (tmp_path / (identity + ".response.png")).write_bytes(image)
        response = {
            "schema_version": "structeval-render-response/v1",
            "request_sha256": hashlib.sha256(raw).hexdigest(),
            "task_id": "000800",
            "controller_result": result,
        }
        (tmp_path / (identity + ".response.json")).write_text(json.dumps(response))
        assert future.result() == score
    assert queue.render("000800", code, tmp_path / "images") == score
    assert len(list(tmp_path.glob("*.request.json"))) == 1
    assert queue.receipts[-1]["retained_result_reused"]
    queue.require_ready_for_judge()


def test_timeout_does_not_become_native_model_zero(tmp_path):
    queue = MatplotlibRenderQueue(
        tmp_path,
        image_root=tmp_path / "images",
        image="sha256:" + "0" * 64,
        timeout=0.02,
        poll_interval=0.002,
    )
    with pytest.raises(RenderBlocked):
        queue.render("000800", "fixture-only", tmp_path / "images")
    assert queue.failures and not queue.receipts
    with pytest.raises(RenderBlocked, match="stop before VQA"):
        queue.require_ready_for_judge()


def test_invalid_destination_still_marks_failure_before_upstream_swallows_it(tmp_path):
    queue = MatplotlibRenderQueue(
        tmp_path, image_root=tmp_path / "images", image="sha256:" + "0" * 64
    )
    try:
        queue.render("000800", "fixture-only", tmp_path / "wrong")
    except RenderBlocked:
        pass  # Simulate the original framework's catch-and-zero behavior.
    with pytest.raises(RenderBlocked):
        queue.require_ready_for_judge()
    assert not list(tmp_path.glob("*.request.json"))


@pytest.mark.parametrize("kind", ["resource_block", "bad_json", "wrong_identity"])
def test_bad_backend_result_cannot_silently_advance_to_judge(tmp_path, kind):
    queue = MatplotlibRenderQueue(
        tmp_path,
        image_root=tmp_path / "images",
        image="sha256:" + "0" * 64,
        timeout=5,
        poll_interval=0.01,
    )
    with ThreadPoolExecutor() as pool:
        future = pool.submit(
            queue.render, "000800", "fixture-only", tmp_path / "images"
        )
        deadline = time.monotonic() + 3
        while not list(tmp_path.glob("*.request.json")):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        path = next(tmp_path.glob("*.request.json"))
        identity = path.name.removesuffix(".request.json")
        response = {
            "schema_version": "structeval-render-response/v1",
            "request_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_id": "000800",
            "controller_result": {
                "status": "BLOCKED_COMPONENT_TIME_LIMIT_NOT_MODEL_SCORE",
                "render_score": 0,
            },
        }
        if kind == "wrong_identity":
            response["request_sha256"] = "0" * 64
        (tmp_path / (identity + ".response.json")).write_text(
            "{" if kind == "bad_json" else json.dumps(response)
        )
        with pytest.raises(RenderBlocked):
            future.result()
    assert queue.failures and not queue.receipts
    with pytest.raises(RenderBlocked):
        queue.require_ready_for_judge()
