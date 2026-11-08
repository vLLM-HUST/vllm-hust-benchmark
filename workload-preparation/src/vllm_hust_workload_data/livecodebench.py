"""Original task envelopes; no prompt generation, execution or private-test decoding."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from .assets import iter_rows
from .records import WorkloadRecord


def original_rows(path):
    for row in iter_rows(Path(path)):
        # Parquet dates are not JSON primitives. Explicit type metadata retains
        # the original value without silently converting dates to plain text.
        row = deepcopy(row)
        for key, value in row.items():
            if isinstance(value, datetime):
                row[key] = {
                    "original_type": "timestamp",
                    "iso8601": value.isoformat(),
                }
        yield row


def adapt_case(row, *, source, index):
    if "question_content" in row and "test_id" not in row:
        scenario = "codegeneration"
        inputs = {"question_title", "question_content", "starter_code"}
        references = {"public_test_cases", "private_test_cases", "metadata"}
        required = inputs | references | {"question_id"}
        identity = row.get("question_id")
    elif "test_id" in row:
        scenario = "testoutputprediction"
        inputs = {"question_title", "question_content", "starter_code", "function_name"}
        references = {"test"}
        required = inputs | references | {"question_id", "test_id"}
        identity = json.dumps(
            [row.get("question_id"), row.get("test_id")],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    elif "numsteps" in row:
        scenario = "codeexecution"
        inputs = {"function_name", "code", "input"}
        references = {"output"}
        required = inputs | references | {"question_id", "id", "numsteps"}
        identity = row.get("id")
    else:
        raise ValueError("unknown original LiveCodeBench scenario schema")
    if required - row.keys() or identity is None:
        raise ValueError("incomplete original LiveCodeBench task")
    if not isinstance(identity, str) or not identity:
        raise ValueError("original native case identity required")
    annotations = {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in inputs | references
    }
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": identity,
            "identity_kind": "original_native_case_identity",
            "native_unit": scenario,
            "input": {key: deepcopy(row[key]) for key in inputs},
        },
        runtime={
            "kind": "native_code_task",
            "execution": "not_executed",
            "requirements": [
                "Original author formatter and scorer required; not a ready API request",
                "Never deserialize private tests or evaluate model output on the host",
                "Self-repair additionally requires actual prior output and scorer feedback",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "livecodebench_original_tasks",
            "source_row_index": index,
            "order": "original_file_order_not_request_arrival",
            "original_annotations": annotations,
            "unmapped_field_names": [],
            "synthetic_fallback": False,
            "timestamp_encoding": "explicit original_type and ISO8601 value",
            "private_tests_decoded": False,
        },
        outcome={key: deepcopy(row[key]) for key in references},
    )
