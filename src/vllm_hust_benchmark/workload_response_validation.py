"""Response protocol/declared-format checks, never semantic task correctness.

No observation, token timing, model judging or dataset rewriting is performed.
The original request supplies any JSON schema; unavailable validation fails closed.
"""

import json


def _nonfinite(value):
    raise ValueError("non-finite JSON")


def check_response(response, original_request):
    """Return payload-free integrity metadata, not an official task score."""
    base = {
        "scope": "ORIGINAL_REQUEST_PROTOCOL_AND_DECLARED_FORMAT_ONLY",
        "semantic_quality": "NOT_EVALUATED_NO_INDEPENDENT_ORACLE",
    }

    def result(status, **extra):
        return {**base, "status": status, **extra}

    if not isinstance(original_request, dict):
        return result("INVALID_SOURCE_REQUEST")
    if not isinstance(response, dict) or response.get("error"):
        return result("INVALID_RESPONSE")
    streaming = original_request.get("stream") is True
    if streaming and (
        response.get("stream") is not True
        or response.get("terminated") is not True
        or response.get("termination") != "[DONE]"
    ):
        return result("STREAM_NOT_TERMINATED")
    if not streaming and response.get("stream"):
        return result("RESPONSE_PROTOCOL_MISMATCH")
    events = response.get("events") if streaming else [response]
    if not isinstance(events, list) or not events:
        return result("EMPTY_RESPONSE")
    choices = {}
    for event in events:
        if not isinstance(event, dict) or event.get("error"):
            return result("INVALID_RESPONSE_EVENT")
        if not isinstance(event.get("choices", []), list):
            return result("INVALID_CHOICES")
        for choice in event.get("choices", []):
            if not isinstance(choice, dict):
                return result("INVALID_CHOICE")
            index = choice.get("index", 0)
            if type(index) is not int or index < 0:
                return result("INVALID_CHOICE_INDEX")
            item = choices.setdefault(
                index, {"content": [], "finish": [], "tools": False}
            )
            content = (
                choice.get("delta", {}) if streaming else choice.get("message", {})
            )
            if not isinstance(content, dict):
                return result("INVALID_MESSAGE")
            text = content.get("content")
            if text is not None and not isinstance(text, str):
                return result("UNSUPPORTED_RESPONSE_CONTENT")
            if text:
                item["content"].append(text)
            item["tools"] |= bool(
                content.get("tool_calls") or content.get("function_call")
            )
            if choice.get("finish_reason") is not None:
                item["finish"].append(choice["finish_reason"])
    expected = original_request.get("n", 1)
    if (
        type(expected) is not int
        or expected <= 0
        or len(choices) != expected
        or any(index >= expected for index in choices)
    ):
        return result("CHOICE_COUNT_MISMATCH")
    declared_format = original_request.get("response_format")
    if declared_format is None:
        declared_format = {"type": "text"}
    if not isinstance(declared_format, dict):
        return result("INVALID_SOURCE_FORMAT")
    kind = declared_format.get("type")
    if not isinstance(kind, str):
        return result("INVALID_SOURCE_FORMAT")
    for choice in choices.values():
        if choice["tools"] or "tool_calls" in choice["finish"]:
            return result("NATIVE_TOOL_LOOP_REQUIRED")
        if "length" in choice["finish"]:
            return result("BUDGET_EXHAUSTED_QUALITY_UNVERIFIED")
        if choice["finish"] != ["stop"]:
            return result("MISSING_OR_UNSUPPORTED_TERMINATION")
        text = "".join(choice["content"])
        if not text.strip():
            return result("EMPTY_OUTPUT")
        if kind == "text":
            continue
        if kind not in {"json_object", "json_schema"}:
            return result("UNSUPPORTED_SOURCE_FORMAT")
        try:
            value = json.loads(text, parse_constant=_nonfinite)
        except (ValueError, TypeError):
            return result("INVALID_JSON_OUTPUT")
        if kind == "json_object":
            if not isinstance(value, dict):
                return result("OUTPUT_NOT_JSON_OBJECT")
        else:
            specification = declared_format.get("json_schema")
            if not isinstance(specification, dict) or not isinstance(
                specification.get("schema"), (dict, bool)
            ):
                return result("INVALID_SOURCE_SCHEMA")
            try:
                import jsonschema
                from referencing import Registry
            except ImportError:
                return result("SCHEMA_VALIDATOR_UNAVAILABLE")
            try:
                schema = specification["schema"]
                validator = (
                    jsonschema.validators.validator_for(schema, default=None)
                    if isinstance(schema, dict) and "$schema" in schema
                    else jsonschema.Draft202012Validator
                )
                if validator is None:
                    return result("UNSUPPORTED_SCHEMA_DIALECT")
                validator.check_schema(schema)
                # Never fetch remote $refs; internal references remain supported.
                validator(schema, registry=Registry()).validate(value)
            except jsonschema.SchemaError:
                return result("INVALID_SOURCE_SCHEMA")
            except jsonschema.ValidationError:
                return result("OUTPUT_SCHEMA_VIOLATION")
            except Exception as error:
                return result(
                    "SCHEMA_VALIDATION_UNAVAILABLE", error_type=type(error).__name__
                )
    return result("RESPONSE_INTEGRITY_VERIFIED", choices=expected, declared_format=kind)
