"""Run the pinned publisher QA scorer with an explicit local judge binding.

Original prompts, yes-substring parsing and aggregation remain publisher code.
This is not reproduction of the publisher's recommended 70B/GPT-4 judges.
"""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from run_longmemeval_original import DATA_FILES, endpoint


SCORER_SHA256 = "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"  # pragma: allowlist secret - publisher source digest
ALIAS = "explicit-local-npu-judge"


def checked_inputs(reference, hypotheses):
    if not isinstance(reference, list) or not isinstance(hypotheses, list):
        raise ValueError("original reference list and hypothesis list required")
    by_id = {row["question_id"]: row for row in reference}
    ids = [row["question_id"] for row in hypotheses]
    if len(by_id) != len(reference) or not ids or len(set(ids)) != len(ids):
        raise ValueError("nonempty unique original question IDs required")
    if any(qid not in by_id for qid in ids):
        raise ValueError("unknown question ID; publisher silent skipping refused")
    if any(not isinstance(row.get("hypothesis"), str) for row in hypotheses):
        raise ValueError("saved string hypotheses required")
    if any("autoeval_label" in row for row in hypotheses):
        raise ValueError("already scored hypotheses must not be overwritten")
    return {by_id[qid]["question_type"] for qid in ids}


def checked_controls(report, model, question_types):
    cases = report.get("control_cases", [])
    expected = {
        "positive-equivalent": True,
        "negative-wrong": False,
        "negative-empty": False,
    }
    if (
        report.get("model") != model
        or report.get("status") != "LOCAL_JUDGE_SELECTED_TYPE_SANITY_PASSED"
        or question_types != {"single-session-assistant"}
        or len(cases) != 3
        or {x.get("control_id") for x in cases} != set(expected)
        or any(
            x.get("expected") is not expected[x["control_id"]]
            or x.get("actual_original_parse_label") is not expected[x["control_id"]]
            or x.get("finish_reason") != "stop"
            for x in cases
        )
    ):
        raise ValueError("matching selected-type local judge sanity controls required")


def bound_program(raw, model):
    if hashlib.sha256(raw).hexdigest() != SCORER_SHA256:
        raise ValueError("pinned publisher scorer identity mismatch")
    tree = ast.parse(raw)
    assignments = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "model_zoo" for t in n.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Dict):
        raise ValueError("unique publisher model selector required")
    assignments[0].value.keys.append(ast.Constant(ALIAS))
    assignments[0].value.values.append(
        ast.Tuple(elts=[ast.Constant(model), ast.Constant("local")], ctx=ast.Load())
    )
    constructors = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "OpenAI"
    ]
    if len(constructors) != 1:
        raise ValueError("unique publisher transport constructor required")
    constructors[0].func.id = "_BoundOpenAI"
    return compile(
        ast.fix_missing_locations(tree), "<pinned-longmemeval-scorer>", "exec"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--variant", choices=DATA_FILES, required=True)
    parser.add_argument("--hypotheses", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", type=endpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = args.source_root / "src/evaluation/evaluate_qa.py"
    if any(
        p.is_symlink() for p in (source, args.reference, args.hypotheses, args.controls)
    ):
        raise ValueError("symlink inputs refused")
    raw = source.read_bytes()
    program = bound_program(raw, args.model)
    ref_raw = args.reference.read_bytes()
    if hashlib.sha256(ref_raw).hexdigest() != DATA_FILES[args.variant][1]:
        raise ValueError("complete original reference asset identity mismatch")
    hypotheses = [
        json.loads(line) for line in args.hypotheses.read_text("utf-8").splitlines()
    ]
    question_types = checked_inputs(json.loads(ref_raw), hypotheses)
    checked_controls(
        json.loads(args.controls.read_text("utf-8")), args.model, question_types
    )
    plan = {
        "source_id": "LONGMEMEVAL",
        "variant": args.variant,
        "model": args.model,
        "selected_original_cases": len(hypotheses),
        "publisher_recommended_judge_reproduced": False,
        "original_prompt_parser_and_aggregation_preserved": True,
        "temperature": 0,
        "max_tokens": 10,
        "sdk_retries": 0,
        "transport_errors_fail_without_replay": True,
        "model_calls": 0,
        "status": "PLAN_ONLY",
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    from openai import OpenAI

    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    saved = args.output / "hypotheses.private.jsonl"
    saved.write_text("".join(json.dumps(x) + "\n" for x in hypotheses), "utf-8")
    client = OpenAI(
        base_url=args.endpoint,
        api_key="local-unused",  # pragma: allowlist secret - dummy loopback key
        max_retries=0,
        timeout=120,
    )
    calls = []

    def create(**kwargs):
        if kwargs["model"] != args.model or kwargs["max_tokens"] != 10:
            raise RuntimeError("publisher judge binding changed")
        request = {
            **kwargs,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        index = len(calls) + 1
        (args.output / f"call-{index}.request.private.json").write_text(
            json.dumps(request), "utf-8"
        )
        calls.append(index)
        try:
            reply = client.chat.completions.create(**request)
        except Exception as error:
            (args.output / f"call-{index}.error.json").write_text(
                json.dumps(
                    {
                        "error_type": type(error).__name__,
                        "outcome": "UNKNOWN_NOT_A_TASK_SCORE",
                    }
                ),
                "utf-8",
            )
            raise RuntimeError("judge transport failed; no automatic replay") from error
        (args.output / f"call-{index}.response.private.json").write_text(
            reply.model_dump_json(), "utf-8"
        )
        if reply.choices[0].finish_reason != "stop":
            raise RuntimeError("incomplete judge response retained, not scored")
        return reply

    def constructor(**_kwargs):
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

    previous_argv = sys.argv
    try:
        sys.argv = [str(source), ALIAS, str(saved), str(args.reference)]
        with (args.output / "publisher-stdout.private.log").open(
            "w", encoding="utf-8"
        ) as stream:
            with redirect_stdout(stream):
                exec(program, {"__name__": "__main__", "_BoundOpenAI": constructor})
        scored = [
            json.loads(line)
            for line in Path(str(saved) + ".eval-results-" + ALIAS)
            .read_text("utf-8")
            .splitlines()
        ]
        if len(scored) != len(hypotheses) or len(calls) != len(hypotheses):
            raise RuntimeError("incomplete original scorer output")
        plan.update(
            status="SELECTED_ORIGINAL_QA_SCORED_LOCAL_JUDGE", model_calls=len(calls)
        )
        (args.output / "receipt-public.json").write_text(
            json.dumps(plan, indent=2), "utf-8"
        )
        print(json.dumps(plan, indent=2))
    finally:
        sys.argv = previous_argv
        client.close()


if __name__ == "__main__":
    main()
