"""Original StableToolBench tasks; execution remains in the author environment."""

from copy import deepcopy

from .records import WorkloadRecord


def adapt_case(row, *, source, index):
    if (
        source.get("source_id") != "STABLETOOLBENCH"
        or not isinstance(row, dict)
        or type(row.get("query_id")) is not int
        or not isinstance(row.get("query"), str)
        or not row["query"].strip()
        or not isinstance(row.get("api_list"), list)
        or not row["api_list"]
        or any(not isinstance(api, dict) for api in row["api_list"])
        or not isinstance(row.get("relevant APIs"), list)
    ):
        raise ValueError("original StableToolBench query, APIs and reference required")
    inputs = deepcopy(row)
    references = inputs.pop("relevant APIs")
    return WorkloadRecord(
        source=deepcopy(source),
        task={"task_id": str(row["query_id"]), "input": inputs},
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "environment_ref": "THUNLP-MT/StableToolBench server + original ToolEnv",
            "scorer_ref": "toolbench/tooleval/fac_eval.py",
            "requirements": [
                "Use the original Agent and cached/simulated tool backend, not static request replay",
                "Keep all advertised APIs; relevant APIs are reference annotations, not a tool filter",
                "Preserve original tool observations, Finish termination and separate official scoring",
                "Simulated APIs do not establish live ToolBench API behavior",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "stabletoolbench_original_tasks",
            "source_row_index": index,
            "order": "original_file_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={"references": {"relevant APIs": references}},
    )


def restore_original_task(record):
    if record.source.get("source_id") != "STABLETOOLBENCH":
        raise ValueError("original StableToolBench record required")
    row = deepcopy(record.task["input"])
    references = record.outcome["references"]
    if (
        "relevant APIs" in row
        or set(references) != {"relevant APIs"}
        or str(row["query_id"]) != record.task["task_id"]
    ):
        raise ValueError("original task/reference separation or identity changed")
    row.update(deepcopy(references))
    return row
