"""Join frozen execution contracts with their public and optional private evidence."""

import math
import re
import statistics

from analyze import qa_f1
from evidence import digest, read_json, require, verify_case_order, verify_response
from export import RESULT_FIELDS
from private_inputs import CASE_METADATA, request_bytes

CONTRACT_V1 = "94b1f3b26f5d2ba3d5ad7ae75ed685035a0fa34a976ddb6324e755c694e7a4ad"
CONTRACT_V2 = "787559a9cfc357664749c18fc656d81a42358822f430116f8a744da7b7c0046a"
WHEEL_SHA256 = "46a0a9b2dd02b7523fe61d5032a78abd7dceb653b660d0304f2dc52d9ae5ae81"
WHEEL_NAME = "vllm_ascend_pyramidkv_hust-0.4.0rc1-py3-none-any.whl"
NAMES = (
    "quality8-off",
    "quality8-on",
    "service8-off-1",
    "service8-on-1",
    "service8-on-2",
    "service8-off-2",
    "service2-off-1",
    "service2-on-1",
    "service2-on-2",
    "service2-off-2",
    "capacity2-off",
    "capacity2-on",
)


def verify_contracts(raw: dict[str, bytes]) -> tuple[dict, dict]:
    for name, expected in (
        ("contract-v1.json", CONTRACT_V1),
        ("contract-v2.json", CONTRACT_V2),
        ("contract.json", CONTRACT_V2),
    ):
        require(
            name in raw and digest(raw[name]) == expected,
            f"Contract missing or changed: {name}",
        )
    first, current = read_json(raw["contract-v1.json"]), read_json(raw["contract.json"])
    require(
        WHEEL_NAME in raw and digest(raw[WHEEL_NAME]) == WHEEL_SHA256,
        "Contract public wheel is missing or changed",
    )
    state = read_json(raw["series-state.json"])
    require(
        state["phase"] == "all_requests_executed" and state["completed"] == list(NAMES),
        "Contract series is not completely executed in frozen order",
    )
    require(
        [row["name"] for row in current["lifecycles"]] == list(NAMES),
        "Contract lifecycle membership differs",
    )
    require(
        current["total_request_budget"] == 1834
        and current["maximum_output_tokens"] == 210112,
        "Contract request budget differs",
    )
    names = {name.split("/")[1] for name in raw if name.startswith("runs/")}
    require(names == set(NAMES), "Contract run inventory differs")
    private = read_json(raw["private-inputs.json"])
    for name, expected in current["files_sha256"].items():
        actual = (
            digest(raw[name]) if name in raw else private.get(name, {}).get("sha256")
        )
        require(actual == expected, f"Contract frozen file missing or changed: {name}")
    return first, current


def verify_execution(
    lifecycle: dict, plan: dict, execution: dict, summary: dict, rows: list[dict]
) -> dict:
    manifest = read_json(plan["manifest_bytes"])
    require(
        execution["plan_manifest_sha256"] == digest(plan["manifest_bytes"]),
        "Execution plan digest differs",
    )
    require(
        execution["plan_manifest"] == summary["plan_manifest"] == manifest,
        "Execution plan identity differs",
    )
    arm = "pyramidkv" if lifecycle["enabled"] else "baseline"
    require(execution["arm"] == summary["arm"] == arm, "Execution arm differs")
    cases = plan["cases"]
    require(
        all(not set(case) - CASE_METADATA for case in cases),
        "Execution public index contains private or unknown fields",
    )
    lookup = {case["case_id"]: case for case in cases}
    require(
        len(lookup) == len(cases)
        and all(re.fullmatch(r"[A-Za-z0-9_-]+", key) for key in lookup),
        "Execution case IDs are duplicated or unsafe",
    )
    groups = [
        group for group in plan["groups"] if group["kind"] in lifecycle["group_kinds"]
    ]
    quality = (
        []
        if lifecycle["skip_quality"]
        else [case["case_id"] for case in cases if case["kind"] == "quality"]
    )
    for group in groups:
        require(
            all(
                key in lookup and lookup[key]["kind"] == group["kind"]
                for key in group["case_ids"]
            ),
            "Execution group membership or kind differs",
        )
    ids = [key for group in groups for key in group["case_ids"]] + quality
    require(
        execution["group_names"] == [group["name"] for group in groups]
        and execution["quality_case_ids"] == quality,
        "Execution selection differs",
    )
    require(
        len(ids) == lifecycle["request_count"] == summary["requests"],
        "Execution request count differs",
    )
    require(
        sum(lookup[key]["max_tokens"] for key in ids) == lifecycle["max_output_tokens"],
        "Execution output budget differs",
    )
    verify_case_order([lookup[key] for key in ids], rows)
    require(
        summary["failures"] == sum(not row["success"] for row in rows) == 0,
        "Execution contains failed requests",
    )
    scores = {}
    for row in rows:
        if row["kind"] == "quality":
            value = row["qa_f1"]
            require(
                math.isfinite(value) and 0 <= value <= 1,
                "Execution quality score is invalid",
            )
            scores.setdefault(row["task"], []).append(value)
    require(
        set(scores) == set(summary["quality_f1"]),
        "Execution quality task inventory differs",
    )
    for task, values in scores.items():
        require(
            math.isclose(
                summary["quality_f1"][task],
                100 * statistics.mean(values),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            "Execution quality aggregate differs",
        )
    return {"case_ids": ids, "groups": groups}


def verify_requests(
    raw: dict,
    prefix: str,
    cases: list[dict],
    rows: list[dict],
    private: dict,
    references: dict | None = None,
) -> dict:
    verify_case_order(cases, rows)
    require(
        len({row["request_id"] for row in rows}) == len(rows),
        "Request IDs are duplicated",
    )
    public_names, private_names, hashes = set(), set(), {}
    coverage = dict.fromkeys(
        (
            "requests_verified",
            "synthetic_payloads_verified",
            "private_payloads_verified",
            "official_f1_recomputed",
        ),
        0,
    )
    for case, row in zip(cases, rows):
        stem = prefix + case["case_id"]
        public_names.update((stem + ".json", stem + ".sse"))
        require(
            not set(row) - RESULT_FIELDS, "Request result has private or unknown fields"
        )
        require(
            stem + ".json" in raw and read_json(raw[stem + ".json"]) == row,
            "Request result original differs",
        )
        require(stem + ".sse" in raw, "Request raw SSE is missing")
        payload = None
        key = stem + ".request.json"
        if case["kind"] == "quality":
            private_names.add(key)
            record = private.get(key, {})
            require(
                re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))
                and type(record.get("bytes")) is int
                and record["bytes"] > 0,
                "Request private payload inventory missing or invalid",
            )
            hashes[case["case_id"]] = record["sha256"]
            if references is not None:
                reference = references.get(case["case_id"])
                require(
                    reference is not None
                    and case.get("dataset_id") == reference["dataset_id"]
                    and case.get("answers") == reference["answers"],
                    "Request official reference identity differs",
                )
                encoded = request_bytes(case)
                require(
                    len(encoded) == record["bytes"]
                    and digest(encoded) == record["sha256"],
                    "Request reconstructed private payload differs",
                )
                payload = read_json(encoded)
                require(
                    math.isclose(
                        qa_f1(row["output"], reference["answers"]),
                        row["qa_f1"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    "Request official F1 differs",
                )
                coverage["private_payloads_verified"] += 1
                coverage["official_f1_recomputed"] += 1
        else:
            public_names.add(key)
            require(
                key in raw and raw[key] == request_bytes(case),
                "Request synthetic payload differs or is missing",
            )
            payload = read_json(raw[key])
            hashes[case["case_id"]] = digest(raw[key])
            coverage["synthetic_payloads_verified"] += 1
        verify_response(case, row, payload, raw[stem + ".sse"])
        coverage["requests_verified"] += 1
    require(
        {name for name in raw if name.startswith(prefix)} == public_names,
        "Request public file inventory differs or leaks private payloads",
    )
    require(
        {name for name in private if name.startswith(prefix)} == private_names,
        "Request private file inventory differs",
    )
    return {**coverage, "payload_sha256": hashes}


def verify_request_counts(before: str, after: str, log: str, count: int) -> None:
    def completed(text):
        entries = re.findall(
            r"^(vllm:request_success_total(?:\{[^\n]*\})?)\s+(\S+)$", text, re.MULTILINE
        )
        require(
            len(entries) == len({key for key, _ in entries}),
            "Request count has duplicate metric series",
        )
        values = [float(value) for _, value in entries]
        require(
            all(
                math.isfinite(value) and value >= 0 and value.is_integer()
                for value in values
            ),
            "Request count has invalid metric values",
        )
        return sum(values)

    require(
        completed(after) - completed(before) == count,
        "Request count differs from engine completions",
    )
    statuses = re.findall(r'"POST /v1/completions HTTP/1\.[01]" (\d{3})', log)
    require(
        statuses == ["200"] * count,
        "Request count differs from HTTP completions or includes failures",
    )


def pressure_cache_review(
    cases: list[dict], rows: list[dict], groups: list[dict], resources: dict
) -> dict:
    salts = [case.get("cache_salt") for case in cases]
    require(
        all(isinstance(salt, str) and salt for salt in salts)
        and len(set(salts)) == len(salts),
        "Pressure request salts are missing or shared",
    )
    phases = {key: group["name"] for group in groups for key in group["case_ids"]}
    cached = []
    for row in rows:
        tokens = row["usage"]["prompt_tokens_details"]["cached_tokens"]
        if not tokens:
            continue
        name = phases.get(row["case_id"])
        phase = resources["phases"].get(name)
        require(phase is not None, "Pressure cached case lacks an observed phase")
        preemptions = (
            phase["last_counters"]["vllm:num_preemptions_total"]
            - phase["first_counters"]["vllm:num_preemptions_total"]
        )
        require(
            preemptions > 0, "Pressure cached case lacks supporting phase preemptions"
        )
        cached.append(
            {
                "case_id": row["case_id"],
                "cached_tokens": tokens,
                "phase": name,
                "phase_preemptions": preemptions,
            }
        )
    return {
        "unique_request_salts": len(salts),
        "cached_tokens": sum(row["cached_tokens"] for row in cached),
        "cached_cases": cached,
        "individual_request_preemption_trace_available": False,
    }
