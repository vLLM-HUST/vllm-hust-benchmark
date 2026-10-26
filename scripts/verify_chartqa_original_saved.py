"""Recheck saved original ChartQA runs with zero new model/judge calls.

Requires the fixed original toolkit, raw Parquet, prepared view and saved native
requests/responses. Never overwrites the original run or treats an incomplete
run as complete. Native scorer outputs go into a new private audit directory.
"""

import argparse
import io
import json
import math
import os
from pathlib import Path
import ssl
import sys

from prepare_chartqa_native_toolkit import REVISION, TREE, verify
from prepare_chartqa_original_view import BYTES, NAME, ROWS, SHA256, digest, verify_view
from run_chartqa_original import select_positions, validate_completion


def validate_receipt(receipt):
    if (
        receipt.get("source_id") != "CHARTQA"
        or receipt.get("status")
        != "SELECTED_ORIGINAL_VISUAL_TASKS_AND_TOOLKIT_SCORE_COMPLETE_NOT_SOURCE_DELIVERY"
    ):
        raise ValueError("Complete saved original native execution required")
    positions = receipt.get("selected_original_positions")
    if not isinstance(positions, list):
        raise ValueError("Saved original positions missing")
    selected = select_positions(ROWS, positions)
    if (
        receipt.get("attempted_original_positions") != selected
        or receipt.get("completed_original_positions") != selected
    ):
        raise ValueError(
            "Missing attempted/completed original cases; never skip failures"
        )
    records = receipt.get("private_records", [])
    if [record.get("original_position") for record in records] != selected:
        raise ValueError("Saved request/response coverage or order differs")
    if receipt.get("official_api_budget") != 2048:
        raise ValueError("Original API budget differs")
    if receipt.get("model_chat_template_kwargs") not in (
        {},
        {"enable_thinking": False},
    ):
        raise ValueError("Unsupported or missing declared model profile")
    proof = receipt.get("source_proof", {})
    if proof.get("revision") != REVISION or proof.get("git_tree") != TREE:
        raise ValueError("Saved toolkit identity differs from pinned source")
    return selected


def verify_case(
    original,
    prediction,
    protocol,
    request,
    response,
    *,
    model,
    messages,
    disable_thinking,
):
    if (
        int(prediction["index"]) != protocol["original_position"]
        or prediction["question"] != original["query"]
        or json.loads(prediction["answer"]) != original["label"]
        or str(prediction["category"]) != str(original["human_or_machine"])
    ):
        raise ValueError("Saved original question/reference/category relation differs")
    answer = validate_completion(
        request,
        response,
        model=model,
        messages=messages,
        disable_thinking=disable_thinking,
    )
    if prediction["prediction"] != answer:
        raise ValueError("Saved prediction differs from actual response")
    usage = response["usage"]
    if (
        usage["prompt_tokens"] != protocol["prompt_tokens"]
        or usage["completion_tokens"] != protocol["completion_tokens"]
        or response["choices"][0]["finish_reason"] != protocol["finish_reason"]
    ):
        raise ValueError("Saved actual usage or finish reason differs")


def compare_metrics(saved, recalculated):
    recalculated = json.loads(json.dumps(recalculated))
    if not isinstance(saved, list) or not saved or len(saved) != len(recalculated):
        raise ValueError("Original metric rows differ")
    for before, after in zip(saved, recalculated):
        if "Overall" not in before or "Overall" not in after:
            raise ValueError("Original overall metric missing")
        if set(before) != set(after):
            raise ValueError("Original metric categories differ")
        for key, value in before.items():
            actual = after[key]
            if (
                type(value) not in (int, float)
                or type(actual) not in (int, float)
                or not math.isfinite(value)
                or not math.isfinite(actual)
                or not 0 <= value <= 100
                or abs(value - actual) > 1e-9
            ):
                raise ValueError("Original saved score differs from unchanged scorer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--toolkit-archive", action="store_true")
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--prepared-view", type=Path, required=True)
    parser.add_argument("--ffmpeg-library-dir", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY_NO_SCORING_OR_MODEL_CALL",
                    "new_model_calls": 0,
                    "paid_judge_calls": 0,
                }
            )
        )
        return
    source, data, prepared, run, output = (
        path.resolve()
        for path in (
            args.official_checkout,
            args.source_file,
            args.prepared_view,
            args.run,
            args.output,
        )
    )
    if prepared.is_relative_to(run) or run.is_relative_to(prepared):
        raise ValueError(
            "Toolkit image cache must be separate from the frozen original run"
        )
    for original in (source, prepared, run):
        if output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError(
                "Fresh audit output must not overlap original assets or run"
            )
    if output.exists() or data.is_relative_to(output):
        raise ValueError("Fresh private audit output required")
    verify(source, archive=args.toolkit_archive)
    if data.stat().st_size != BYTES or digest(data) != SHA256:
        raise ValueError("Original Parquet identity mismatch")
    receipt = json.loads((run / "receipt-private.json").read_text("utf-8"))
    selected = validate_receipt(receipt)
    sys.dont_write_bytecode = True
    os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", LMUData=str(prepared))
    os.environ["LD_LIBRARY_PATH"] = (
        str(args.ffmpeg_library_dir.resolve(strict=True))
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", "")
    )
    sys.path.insert(0, str(source))
    original_ssl = ssl._create_default_https_context
    try:
        from vlmeval.dataset import ImageVQADataset
        from vlmeval.api.gpt import OpenAIWrapper
        from vlmeval.smp import dump
    finally:
        ssl._create_default_https_context = original_ssl
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    originals = pq.read_table(data).to_pylist()
    view = prepared / f"{NAME}.tsv"
    if (
        len(originals) != ROWS
        or verify_view(originals, view) != ROWS
        or digest(view) != receipt["view_sha256"]
    ):
        raise ValueError("Original data/view relationships differ")
    dataset = ImageVQADataset(NAME)
    # Encoder only. Never call generate/generate_inner or any model endpoint.
    encoder = OpenAIWrapper(
        model=receipt["model"],
        key="local-only",
        api_base="http://127.0.0.1:19466/v1/chat/completions",
        retry=1,
    )  # pragma: allowlist secret - loopback dummy
    predictions = pd.read_excel(
        run / "predictions-private.xlsx",
        keep_default_na=False,
        dtype={"prediction": str, "question": str, "answer": str},
    )
    if len(predictions) != len(selected):
        raise ValueError("Missing or additional saved predictions")
    for offset, (index, protocol) in enumerate(
        zip(selected, receipt["private_records"])
    ):
        request_path, response_path = (
            run / f"row-{index}.request-private.json",
            run / f"row-{index}.response-private.json",
        )
        if (
            digest(request_path) != protocol["request_sha256"]
            or digest(response_path) != protocol["response_sha256"]
        ):
            raise ValueError("Saved request/response content identity differs")
        prompt = dataset.build_prompt(index)
        if (
            prompt[-1]["value"]
            != originals[index]["query"]
            + "\nAnswer the question using a single word or phrase."
        ):
            raise ValueError("Original toolkit prompt differs")
        image, original_image = (
            Image.open(prompt[0]["value"]).convert("RGB"),
            Image.open(io.BytesIO(originals[index]["image"]["bytes"])).convert("RGB"),
        )
        if (
            image.size != original_image.size
            or image.tobytes() != original_image.tobytes()
        ):
            raise ValueError("Prompt cache image pixels differ from original")
        verify_case(
            originals[index],
            predictions.iloc[offset],
            protocol,
            json.loads(request_path.read_text("utf-8")),
            json.loads(response_path.read_text("utf-8")),
            model=receipt["model"],
            messages=encoder.prepare_inputs(prompt, None),
            disable_thinking=bool(receipt["model_chat_template_kwargs"]),
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    try:
        audit_predictions = output / "predictions-audit-private.xlsx"
        dump(predictions, str(audit_predictions))
        recalculated = dataset.evaluate(
            str(audit_predictions), use_verifier=False
        ).to_dict(orient="records")
        compare_metrics(receipt.get("author_metrics"), recalculated)
        verify(source, archive=args.toolkit_archive)
        if digest(data) != SHA256 or verify_view(originals, view) != ROWS:
            raise ValueError("Original data changed during audit")
        summary = {
            "source_id": "CHARTQA",
            "status": "SAVED_NATIVE_REQUESTS_RESPONSES_AND_ORIGINAL_SCORER_VERIFIED",
            "original_positions": selected,
            "new_model_calls": 0,
            "paid_judge_calls": 0,
            "all_original_view_rows_verified": ROWS,
            "prompt_cache_pixels_match_original": True,
            "wire_payload_matches_original_encoder": True,
            "http_lossless_pixel_identity_claim": False,
            "author_metrics": recalculated,
            "original_run_modified": False,
            "source_checkbox_eligible": False,
            "raw_payloads_published": False,
        }
        (output / "receipt-public.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException as error:
        (output / "failure-private.json").write_text(
            json.dumps(
                {
                    "status": "FAILED_RETAINED",
                    "error_type": type(error).__name__,
                    "new_model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
