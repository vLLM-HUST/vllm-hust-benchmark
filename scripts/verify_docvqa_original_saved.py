"""Reconstruct fixed original DocVQA requests and rescore saved outputs offline.

No model API or original task rerun. Full responses/images stay server-side.
"""

import argparse
import asyncio
import base64
from io import BytesIO
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from run_docvqa_original import FIXED_IDS, TASK_SHA256, digest, verify_originals


def decode_image(value):
    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("resolved original PNG attachment required")
    return base64.b64decode(value[len(prefix) :], validate=True)


async def check_sample(sample, row, png, native, target):
    """Check native log fields without repairing output or applying a new parser."""
    if sample.error is not None or len(sample.input) != 1:
        raise ValueError("saved native sample incomplete")
    initial = sample.input[0]
    if (
        initial.role != "user"
        or len(initial.content) != 2
        or initial.content[0].text != row["question"]
        or decode_image(initial.content[1].image) != png
        or sample.target != row["answers"]
    ):
        raise ValueError("original input, image or reference differs")
    events = [event for event in sample.events if event.event == "model"]
    if len(events) != 1:
        raise ValueError("exactly one original model call per fixed sample required")
    event = events[0]
    request, response = event.call.request, event.call.response
    if request.get("model") != "Qwen/Qwen3.5-9B" or request.get("max_tokens") != 16384:
        raise ValueError("declared original validation model/budget differs")
    messages = request["messages"]
    expected_text = native.FREEFORM_TEMPLATE.format(question=row["question"])
    if (
        len(event.input) != 1
        or event.input[0].role != "user"
        or len(event.input[0].content) != 2
        or event.input[0].content[0].text != expected_text
        or decode_image(event.input[0].content[1].image) != png
    ):
        raise ValueError(
            "recorded native model input differs from original template/image"
        )
    api_image = messages[0]["content"][1]["image_url"]["url"]
    api_image_redacted = api_image == "<base64-data-removed>"
    if (
        len(messages) != 1
        or messages[0]["role"] != "user"
        or len(messages[0]["content"]) != 2
        or messages[0]["content"][0]["text"] != expected_text
        or (not api_image_redacted and decode_image(api_image) != png)
    ):
        raise ValueError("saved API request differs from native template/image")
    completion = sample.output.completion
    if (
        response["choices"][0]["message"]["content"] != completion
        or event.output.completion != completion
    ):
        raise ValueError("saved raw/native output differs; do not clean it")
    score = await native.docvqa_scorer()(
        SimpleNamespace(output=SimpleNamespace(completion=completion)),
        target(row["answers"]),
    )
    saved = sample.scores["docvqa_scorer"]
    if score.value != saved.value or score.answer != saved.answer:
        raise ValueError("unchanged native ANLS rescore differs")
    return {
        "question_id": str(sample.id),
        "score": saved.value,
        "finish_reason": response["choices"][0]["finish_reason"],
        "input_tokens": response["usage"]["prompt_tokens"],
        "output_tokens": response["usage"]["completion_tokens"],
        "native_model_input_image_bytes_verified": True,
        "raw_api_image_log_redacted_not_byte_verifiable": api_image_redacted,
    }


def verify(args):
    output = Path(args.output).resolve()
    run = Path(args.run_root).resolve()
    data = Path(args.data_root).resolve()
    if output.exists() or any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (run, data)
    ):
        raise ValueError("fresh separate metadata audit output required")
    verify_originals(data, args.framework_source_root)
    prior = json.loads((run / "native-public.json").read_text("utf-8"))
    if (
        prior["status"]
        != "FIXED3_NATIVE_TASKS_SCORED_SAVED_AUDIT_AND_CI_HANDOFF_PENDING"
        or len(prior["attempts"]) != 3
        or prior["native_sample_errors"] != 0
    ):
        raise ValueError("original native run incomplete")
    output.mkdir(mode=0o700, parents=True)
    os.environ["XDG_DATA_HOME"] = str(output / "native-reader-data")
    native = importlib.import_module("inspect_evals.docvqa.docvqa")
    if digest(native.__file__) != TASK_SHA256:
        raise ValueError("original maintained scorer identity differs")
    from inspect_ai.log import read_eval_log
    from inspect_ai.scorer import Target
    from PIL import Image
    import pyarrow.parquet as pq

    paths = list((run / "native-logs").glob("*.json"))
    if len(paths) != 1:
        raise ValueError("one original full native evaluation log required")
    log = read_eval_log(paths[0], resolve_attachments=True)
    if (
        log.status != "success"
        or len(log.samples or []) != 3
        or {str(s.id) for s in log.samples} != set(FIXED_IDS)
    ):
        raise ValueError("original fixed sample set differs")
    first = data / "DocVQA/validation-00000-of-00006.parquet"
    rows = next(pq.ParquetFile(first).iter_batches(batch_size=3)).to_pylist()
    if [row["questionId"] for row in rows] != FIXED_IDS:
        raise ValueError("original fixed first3 positions differ")
    samples = {str(s.id): s for s in log.samples}
    results = []
    for row in rows:
        image = Image.open(BytesIO(row["image"]["bytes"]))
        image.thumbnail((1024, 1024))
        encoded = BytesIO()
        image.save(encoded, format=image.format)
        results.append(
            asyncio.run(
                check_sample(
                    samples[row["questionId"]], row, encoded.getvalue(), native, Target
                )
            )
        )
    result = {
        "source_id": "DOCVQA",
        "status": "FIXED3_ORIGINAL_REQUESTS_IMAGES_OUTPUTS_AND_ANLS_SAVED_AUDIT_VERIFIED_CI_PENDING",
        "source_checkbox_eligible": False,
        "original_fixed_question_ids": FIXED_IDS,
        "samples": results,
        "original_native_log_sha256": digest(paths[0]),
        "native_thumbnail1024_reconstructed_from_original_png": True,
        "native_template_and_raw_api_outputs_equal": True,
        "raw_api_image_redaction_boundary": "Native logger removes raw API image base64. Verify complete recorded native model-input attachment against original framework thumbnail; do not claim raw wire image bytes recovered from redacted log.",
        "unchanged_native_scorer_recomputed": True,
        "model_calls": 0,
        "responses_cleaned_or_replaced": 0,
        "aggregate_fixed_sample_anls": sum(s["score"] for s in results) / 3,
        "full_validation_model_run_claim": False,
        "performance_claim": False,
    }
    (output / "saved-audit-public.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "data-root", "framework-source-root", "output"):
        parser.add_argument("--" + name, required=True)
    print(json.dumps(verify(parser.parse_args())), flush=True)
