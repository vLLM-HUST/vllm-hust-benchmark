"""Run fixed original MVBench_MP4 tasks using unmodified VLMEvalKit methods.

Plan by default. Original role preprocessing, video sampling, API encoding and
exact-matching scoring are preserved. Never replace a failed selected question.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

from prepare_chartqa_native_toolkit import verify
from prepare_mvbench_original_view import prepare_view
from run_chartqa_original import validate_completion, validate_endpoint


def fixed_positions(data):
    groups = {}
    for index, row in enumerate(data):
        groups.setdefault(row["task_type"], []).append(index)
    selected = []
    for group in groups.values():
        if len(group) < 3:
            raise ValueError("Original task group has fewer than three cases")
        selected.extend(group[:3])
    if len(selected) != 60 or len(set(selected)) != 60:
        raise ValueError("Complete twenty-type fixed selection required")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--frame-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:19466/v1/chat/completions"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_endpoint(args.endpoint)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "selected_original_tasks": 60,
                    "frames_each": 16,
                    "scorer": "original_exact_matching",
                    "model_calls": 0,
                }
            )
        )
        return
    source = args.official_checkout.resolve(strict=True)
    snapshot = args.snapshot.resolve(strict=True)
    frames = args.frame_cache.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (source, snapshot, frames)
    ):
        raise ValueError(
            "Fresh private output separate from originals/frame cache required"
        )
    proof = verify(source, archive=True)
    prepare_view(source, snapshot)
    sys.dont_write_bytecode = True
    os.environ.update(
        LMUData=str(frames), TORCH_DEVICE_BACKEND_AUTOLOAD="0", HF_HUB_OFFLINE="1"
    )
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, str(source))
    import pandas as pd
    from vlmeval.api.gpt import OpenAIWrapper
    from vlmeval.dataset.mvbench import MVBench_MP4
    from vlmeval.smp import dump, load

    dataset = MVBench_MP4(dataset="MVBench_MP4", nframe=16, fps=-1)
    if len(dataset.data) != 4000:
        raise ValueError("Original complete dataset required")
    selected = fixed_positions(dataset.data.to_dict(orient="records"))
    model = OpenAIWrapper(
        model=args.model, key="local-only", api_base=args.endpoint, retry=1, timeout=300
    )  # pragma: allowlist secret - local dummy key
    if model.max_tokens != 2048 or model.temperature != 0:
        raise ValueError("Original model generation defaults drifted")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "MVBENCH",
        "status": "RUNNING",
        "model": args.model,
        "selected_original_positions": selected,
        "attempted_original_positions": [],
        "completed_original_positions": [],
        "source_checkbox_eligible": False,
        "paid_judge_calls": 0,
        "source_proof": proof,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": [],
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        predictions = []
        for local_index, index in enumerate(selected):
            row = dataset.data.iloc[index]
            prompt = dataset.build_prompt(index, video_llm=False)
            if sum(item["type"] == "image" for item in prompt) != 16:
                raise ValueError("Original sixteen-frame input required")
            # These are the exact preprocessing calls made by BaseAPI.generate.
            # The original wrapper puts its assistant-suffix text in user content;
            # do not invent assistant-prefill transport semantics here.
            message, system = model.preprocess_message_with_role(copy.deepcopy(prompt))
            message = model.preproc_content(message)
            kwargs = copy.deepcopy(model.default_kwargs)
            kwargs.update(
                system_prompt=system, chat_template_kwargs={"enable_thinking": False}
            )
            receipt["attempted_original_positions"].append(index)
            save()
            code, answer, response = model.generate_inner(message, **kwargs)
            raw = response.request.body
            raw = raw.encode() if isinstance(raw, str) else raw
            (output / f"row-{index}.request-private.json").write_bytes(raw)
            (output / f"row-{index}.response-private.json").write_text(
                response.text, encoding="utf-8"
            )
            if code != 0 or response.status_code != 200:
                raise ValueError(
                    "Original request failed; retained without retry or replacement"
                )
            verified = validate_completion(
                json.loads(raw),
                response.json(),
                model=args.model,
                messages=model.prepare_inputs(message, system),
                disable_thinking=True,
            )
            if not verified or answer != verified:
                raise ValueError("Missing or inconsistent native answer")
            record = row.to_dict()
            # Author scorer indexes dataframe by evaluation index. Preserve the
            # global native index separately; never change task/answer/media.
            record.update(index=local_index, source_index=index, prediction=answer)
            predictions.append(record)
            dump(pd.DataFrame(predictions), str(output / "predictions-private.xlsx"))
            receipt["records"].append(
                {
                    "source_index": index,
                    "evaluation_index": local_index,
                    "task_type": row["task_type"],
                    "usage": response.json()["usage"],
                    "finish_reason": response.json()["choices"][0]["finish_reason"],
                    "request_sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            receipt["completed_original_positions"].append(index)
            save()
        prediction_file = output / "predictions-private.xlsx"
        receipt["author_metrics"] = dataset.evaluate(
            str(prediction_file), model="exact_matching"
        )
        score = load(str(output / "predictions-private_score.xlsx"))
        if len(score) != 60 or not score["score"].isin([0, 1]).all():
            raise ValueError(
                "Native scoring incomplete; never count missing predictions as success"
            )
        verify(source, archive=True)
        receipt["status"] = (
            "FIXED_60_ORIGINAL_VIDEO_TASKS_AND_NATIVE_SCORING_COMPLETE_NOT_SOURCE_DELIVERY"
        )
    except BaseException as error:
        receipt.update(
            status="FAILED_RETAINED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "source_id",
                    "status",
                    "completed_original_positions",
                    "author_metrics",
                    "source_checkbox_eligible",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
