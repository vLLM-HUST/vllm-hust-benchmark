"""Original candidate relationships for feedback-based repair; no code execution."""

import json


def compatible_check_prompt(original, question, result, metadata):
    """Approved error-message-only compatibility; never edits original metadata."""
    try:
        return original(question, result, metadata)
    except KeyError as error:
        if len(error.args) != 1 or error.args[0] not in {"inputs", "expected", "error"}:
            raise
        feedback = json.loads(metadata)
        if (
            feedback.get("error_code") != -4
            or {"inputs", "expected", "error"} <= feedback.keys()
            or not isinstance(feedback.get("error_message"), str)
            or not feedback["error_message"].strip()
        ):
            raise
        return (
            "The above code is incorrect. The original evaluator reported the following error:\n"
            + feedback["error_message"]
        )


def native_repair_formatter(mode):
    import lcb_runner.prompts.self_repair as native

    original = getattr(native, "_hust_original_check_prompt", native.get_check_prompt)
    native._hust_original_check_prompt = original
    if mode == "strict":
        native.get_check_prompt = original
        return native.format_prompt_self_repair
    if mode != "error-message-only":
        raise ValueError("explicit native or approved compatibility mode required")

    def checked(question, result, metadata):
        return compatible_check_prompt(original, question, result, metadata)

    native.get_check_prompt = checked
    return native.format_prompt_self_repair


def repair_identities(evaluation):
    if not isinstance(evaluation, list) or not evaluation:
        raise ValueError("original nonempty evaluated candidate list required")
    seen = set()
    identities = []
    for row in evaluation:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("distinct original question identifiers required")
        seen.add(qid)
        if any(
            not isinstance(row.get(key), list) or len(row[key]) != 10
            for key in ("output_list", "code_list", "graded_list", "metadata")
        ):
            raise ValueError(
                "all ten evaluated original candidates and feedback required"
            )
        if not isinstance(row.get("question_content"), str):
            raise ValueError("original question content required")
        for index, grade in enumerate(row["graded_list"]):
            if type(grade) is not bool:
                raise ValueError("original boolean grade required")
            if not isinstance(row["metadata"][index], str):
                raise ValueError("unchanged original JSON feedback string required")
            if not grade:
                identities.append(f"{qid}::candidate:{index}")
    return identities


def build_repair_requests(evaluation, formatter, model_style, native_kwargs):
    expected = repair_identities(evaluation)
    if native_kwargs.get("n") != 1:
        raise ValueError(
            "author self-repair requires n=1 per original failed candidate"
        )
    requests = []
    for row in evaluation:
        for index in range(10):
            prompt = formatter(
                row["question_content"],
                model_style,
                row["code_list"][index],
                row["graded_list"][index],
                row["metadata"][index],
            )
            if row["graded_list"][index]:
                if prompt != "":
                    raise ValueError(
                        "author formatter must carry accepted candidate unchanged"
                    )
                continue
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(
                    "author generic API failed-candidate repair prompt required"
                )
            body = {
                key: value for key, value in native_kwargs.items() if key != "timeout"
            }
            body["messages"] = prompt
            requests.append(
                {
                    "question_id": row["question_id"],
                    "candidate_index": index,
                    "request_id": f"{row['question_id']}::candidate:{index}",
                    "request": body,
                }
            )
    if [row["request_id"] for row in requests] != expected:
        raise ValueError("original repair candidate relationships incomplete")
    return requests


def repaired_outputs(evaluation, requests, read_output):
    expected = repair_identities(evaluation)
    if [row["request_id"] for row in requests] != expected:
        raise ValueError("complete original failed-candidate repair requests required")
    lookup = {
        row["request_id"]: (position, row) for position, row in enumerate(requests)
    }
    result = []
    for row in evaluation:
        outputs = []
        for index in range(10):
            if row["graded_list"][index]:
                outputs.append(row["output_list"][index])
            else:
                position, request = lookup[f"{row['question_id']}::candidate:{index}"]
                response = read_output(position, request)
                if not isinstance(response, list) or len(response) != 1:
                    raise ValueError(
                        "one repaired output per original failed candidate required"
                    )
                outputs.append(response)
        result.append(outputs)
    return result
