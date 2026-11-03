"""Join frozen execution contracts with their public and optional private evidence."""

import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path

from analyze import (
    COUNTERS,
    counter_deltas,
    hbm_snapshot,
    qa_f1,
    quality_comparison,
    resource_summary,
)
from evidence import (
    digest,
    read_json,
    require,
    verify_case_order,
    verify_groups,
    verify_response,
    verify_transactions,
)
from export import RESULT_FIELDS
from private_inputs import (
    CASE_METADATA,
    load_references,
    public_plan,
    reconstruct_cases,
    request_bytes,
)
from runtime import EXPERIMENT, MODEL, verify_lifecycle_records, verify_startup_log

# Public preregistered artifact digests, not credentials.
CONTRACT_V1 = "94b1f3b26f5d2ba3d5ad7ae75ed685035a0fa34a976ddb6324e755c694e7a4ad"  # pragma: allowlist secret
CONTRACT_V2 = "787559a9cfc357664749c18fc656d81a42358822f430116f8a744da7b7c0046a"  # pragma: allowlist secret
WHEEL_SHA256 = "46a0a9b2dd02b7523fe61d5032a78abd7dceb653b660d0304f2dc52d9ae5ae81"  # pragma: allowlist secret
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


def verify_frozen_files(raw: dict, private: dict, expected: dict) -> None:
    confidential = {
        f"{plan}/cases.json"
        for plan in ("quality-plan", "service-plan", "service-pressure-plan")
    }
    for name, sha in expected.items():
        if name in confidential:
            require(
                name not in raw and private.get(name, {}).get("sha256") == sha,
                f"Contract private frozen file missing, changed or exposed: {name}",
            )
        else:
            require(
                name in raw and digest(raw[name]) == sha,
                f"Contract public frozen file missing or changed: {name}",
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
    verify_frozen_files(raw, private, current["files_sha256"])
    return first, current


def verify_telemetry(
    samples: list[dict],
    groups: list[dict],
    rows: list[dict],
    times: dict,
    before: str,
    after: str,
) -> dict:
    resources = resource_summary(samples, [1, 6])
    phases = ["initial", *(group["name"] for group in groups)]
    phases += list(
        dict.fromkeys(
            "quality-" + row["task"] for row in rows if row["kind"] == "quality"
        )
    )
    zero = "\n".join(f"vllm:{name}_total 0" for name in COUNTERS)
    lower, upper = counter_deltas(zero, before), counter_deltas(zero, after)
    previous = -1
    for sample in samples:
        require(sample["phase"] in phases, "Telemetry phase is not registered")
        ordinal = phases.index(sample["phase"])
        require(ordinal >= previous, "Telemetry phase order is reversed")
        previous = ordinal
        require(
            times["ready_at"] <= sample["time_unix"] <= times["completed_at"],
            "Telemetry sample is outside the ready/completion interval",
        )
        require(
            all(
                lower[name] <= sample["metrics"][f"vllm:{name}_total"] <= upper[name]
                for name in COUNTERS
            ),
            "Telemetry counters exceed process boundaries",
        )
    return {"resources": resources, "counter_deltas": counter_deltas(before, after)}


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


def verify_evidence(raw: dict[str, bytes], private_root: Path | None = None) -> dict:
    """Verify the complete preregistered experiment; private coverage is explicit."""
    first, contract = verify_contracts(raw)
    private = read_json(raw["private-inputs.json"])
    verify_frozen_files(raw, private, first["files_sha256"])
    require(
        digest(raw["env.sh.source"])
        == contract["external_files_sha256"][
            "/root/workspace/pyramidkv-runtime/env.sh"
        ],
        "Frozen launch environment differs",
    )
    references = None
    quality_cases = None
    if private_root is not None:
        private_root = Path(private_root)
        manifest = read_json(raw["quality-plan/manifest.json"])
        references = load_references(private_root, manifest)
        content = (private_root / "quality-plan/cases.json").read_bytes()
        require(
            digest(content) == manifest["cases_sha256"],
            "Private quality preparation differs",
        )
        quality_cases = read_json(content)

    plans, expected_private = {}, set()
    for name in sorted({row["plan"] for row in contract["lifecycles"]}):
        manifest_bytes = raw[f"{name}/manifest.json"]
        manifest = read_json(manifest_bytes)
        index = read_json(raw[f"{name}/case-index.json"])
        synthetic = read_json(raw[f"{name}/synthetic-cases.json"])
        projected, _ = public_plan(synthetic)
        require(
            projected == [case for case in index if case["kind"] != "quality"],
            "Public synthetic plan projection differs",
        )
        require(
            manifest["cases_sha256"] == private[f"{name}/cases.json"]["sha256"]
            and manifest["groups_sha256"] == digest(raw[f"{name}/groups.json"]),
            "Plan frozen input identity differs",
        )
        if quality_cases is not None:
            full = reconstruct_cases(
                index, synthetic, quality_cases, manifest["cases_sha256"]
            )
            encoded = (json.dumps(full, ensure_ascii=False, indent=2) + "\n").encode()
            require(
                len(encoded) == private[f"{name}/cases.json"]["bytes"],
                "Private reconstructed plan byte length differs",
            )
        else:
            lookup = {case["case_id"]: case for case in synthetic}
            full = [lookup.get(case["case_id"], case) for case in index]
        plans[name] = {
            "cases": index,
            "full": {case["case_id"]: case for case in full},
            "manifest_bytes": manifest_bytes,
            "groups": read_json(raw[f"{name}/groups.json"]),
        }
        expected_private.add(f"{name}/cases.json")
    quality_manifest = read_json(raw["quality-plan/manifest.json"])
    for task, sha in quality_manifest["dataset_files_sha256"].items():
        name = f"data/{task}.jsonl"
        expected_private.add(name)
        require(
            name not in raw and private.get(name, {}).get("sha256") == sha,
            "Private dataset custody differs",
        )
        if private_root is not None:
            require(
                (private_root / name).stat().st_size == private[name]["bytes"],
                "Private dataset byte length differs",
            )
    for name, record in private.items():
        require(
            type(record.get("bytes")) is int
            and record["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")),
            f"Private inventory entry invalid: {name}",
        )

    def idle(name):
        text = raw[name].decode()
        hbm_snapshot(text, [1, 6])
        require(
            set(re.findall(r"No running processes found in NPU (\d+)", text))
            == {"1", "6"},
            "Lifecycle devices were not free",
        )
        require(
            not re.search(r"^\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", text, re.MULTILINE),
            "Lifecycle device process remains",
        )

    coverage = dict.fromkeys(
        (
            "requests_verified",
            "synthetic_payloads_verified",
            "private_payloads_verified",
            "official_f1_recomputed",
        ),
        0,
    )
    previous = datetime.fromisoformat(contract["created_at_utc"]).timestamp()
    identities, snapshots_missing = set(), []
    common_env = common_args = None
    runs, payloads, results = {}, {}, {}
    for ordinal, current in enumerate(contract["lifecycles"]):
        name = current["name"]
        frozen = first["lifecycles"][ordinal] if ordinal < 3 else current
        prefix = f"runs/{name}/"
        require(prefix + "failure.json" not in raw, "Lifecycle has a retained failure")
        records = {
            key: read_json(raw[prefix + key + ".json"])
            for key in ("process", "ready", "completion", "shutdown")
        }
        records["snapshot"] = (
            read_json(raw[prefix + "observed-runtime.json"])
            if prefix + "observed-runtime.json" in raw
            else None
        )
        times = verify_lifecycle_records(
            frozen, records, CONTRACT_V1 if ordinal < 3 else CONTRACT_V2, previous
        )
        identity = tuple(times["identity"])
        require(identity not in identities, "Lifecycle process identity was reused")
        identities.add(identity)
        require(
            times["shutdown_at"] <= contract["window_end_unix"]
            and times["ready_at"] - times["started_at"]
            <= contract["startup_timeout_seconds"]
            and times["completed_at"] - times["ready_at"]
            <= frozen["client_timeout_seconds"],
            "Lifecycle exceeded a preregistered time window",
        )
        previous = times["shutdown_at"]
        if not times["actual_runtime_snapshot"]:
            snapshots_missing.append(name)
        else:
            env = {
                key: value
                for key, value in records["snapshot"]["selected_environment"].items()
                if key
                not in (
                    "PYRAMIDKV_ENABLED",
                    "VLLM_ASCEND_PYRAMIDKV_ENABLED",
                    "PYRAMIDKV_KV_CACHE_MEMORY_BYTES",
                )
            }
            require(
                common_env is None or env == common_env,
                "Lifecycle common environment differs",
            )
            common_env = env
        log = raw[prefix + "server.log"].decode()
        args = verify_startup_log(log, frozen)
        args.pop("kv_cache_memory_bytes")
        require(
            common_args is None or args == common_args,
            "Lifecycle common parsed arguments differ",
        )
        common_args = args
        for boundary in ("before", "after"):
            idle(prefix + f"npu-{boundary}.txt")
        expected_client = [
            "/root/workspace/pyramidkv-runtime/venv/bin/python",
            EXPERIMENT + "/evaluate.py",
            "run",
            "--plan",
            EXPERIMENT + "/" + frozen["plan"],
            "--output",
            EXPERIMENT + "/" + prefix + "evaluation",
            "--arm",
            "pyramidkv" if frozen["enabled"] else "baseline",
            "--base-url",
            "http://127.0.0.1:8013",
            "--group-kinds",
            *frozen["group_kinds"],
        ]
        if frozen["skip_quality"]:
            expected_client.append("--skip-quality")
        require(
            read_json(raw[prefix + "client-command.json"]) == expected_client,
            "Lifecycle client command differs",
        )
        expected_launcher = f'''#!/usr/bin/env bash
set -eo pipefail
source /root/workspace/pyramidkv-runtime/env.sh
export PYTHONPATH={EXPERIMENT}/release-overlay:$PYTHONPATH
export MODEL_PATH={MODEL}
export SERVE_PORT=8013 PYRAMIDKV_ENABLED={int(frozen["enabled"])}
export VLLM_ASCEND_PYRAMIDKV_BATCHED_METADATA=1
export PYRAMIDKV_CONFIG={EXPERIMENT}/pyramidkv-aligned.json
export PYRAMIDKV_KV_CACHE_MEMORY_BYTES={frozen["kv_bytes_per_worker"]}
unset VLLMHUST_EXT_ENABLED_BUNDLES VLLM_EXTENSION_MANIFESTS VLLM_EXTENSION_BUNDLES VLLM_ASCEND_KVCOMPRESS_ENABLED VLLM_ASCEND_KVCOMPRESS_CONFIG
python -c 'import vllm_ascend_pyramidkv as p; assert p.__file__.startswith("{EXPERIMENT}/release-overlay/"); print("Installed plugin:",p.__file__,p.__version__,flush=True)'
exec bash {EXPERIMENT}/serve.sh
'''
        require(
            raw[prefix + "serve.sh"].decode() == expected_launcher,
            "Lifecycle written launch source differs",
        )
        advice = read_json(raw[prefix + "startup-cache-events.json"])
        require(
            len(advice) == records["ready"]["cache_advice_events"],
            "Lifecycle startup advice count differs",
        )
        plan = plans[frozen["plan"]]
        evaluation = prefix + "evaluation/"
        rows = read_json(raw[evaluation + "results.json"])
        selected = verify_execution(
            frozen,
            plan,
            read_json(raw[evaluation + "execution.json"]),
            read_json(raw[evaluation + "summary.json"]),
            rows,
        )
        cases = [plan["full"][key] for key in selected["case_ids"]]
        groups = read_json(raw[evaluation + "groups.json"])
        verify_groups(selected["groups"], groups, rows)
        checked = verify_requests(
            raw, evaluation + "requests/", cases, rows, private, references
        )
        for key in coverage:
            coverage[key] += checked[key]
        payloads[name] = checked["payload_sha256"]
        expected_private.update(
            evaluation + "requests/" + case["case_id"] + ".request.json"
            for case in cases
            if case["kind"] == "quality"
        )
        before, after = (
            raw[evaluation + f"metrics-{boundary}.txt"].decode()
            for boundary in ("before", "after")
        )
        verify_request_counts(before, after, log, len(rows))
        telemetry = verify_telemetry(
            [
                read_json(line)
                for line in raw[evaluation + "telemetry.jsonl"].splitlines()
            ],
            groups,
            rows,
            times,
            before,
            after,
        )
        transactions = verify_transactions(
            log,
            rows,
            frozen["enabled"],
            pressure=frozen["kv_bytes_per_worker"] == 2147483648,
        )
        runs[name] = {
            "requests": len(rows),
            "times": times,
            **telemetry,
            "compression_transactions": len(transactions["transactions"]),
            "repeated_commit_cases": transactions["repeated_commit_cases"],
            "log_recoveries": transactions["log_recoveries"],
            "cached_prompt_tokens": sum(
                row["usage"]["prompt_tokens_details"]["cached_tokens"] for row in rows
            ),
        }
        if name.startswith("service2-"):
            runs[name]["pressure_cache_review"] = pressure_cache_review(
                cases, rows, groups, telemetry["resources"]
            )
        results[name] = rows
    require(
        set(private) == expected_private,
        "Private inventory has missing or unclassified originals",
    )
    require(
        coverage["requests_verified"] == 1834
        and coverage["synthetic_payloads_verified"] == 1138
        and coverage["private_payloads_verified"]
        == coverage["official_f1_recomputed"]
        == (696 if private_root is not None else 0),
        "Global verification coverage differs",
    )
    require(
        payloads["quality8-off"] == payloads["quality8-on"],
        "Quality paired payloads differ",
    )
    require(
        all(
            payloads[name] == payloads["service8-off-1"]
            for name in NAMES
            if name.startswith("service")
        ),
        "Service payloads differ across budgets or arms",
    )
    for name in ("capacity2-off", "capacity2-on"):
        require(
            payloads[name]
            == {key: payloads["quality8-off"][key] for key in payloads[name]},
            "Capacity payloads differ across budgets or arms",
        )
    final = read_json(raw["final-identity.json"])
    host = read_json(raw["host-identity.json"])
    require(
        final["at"] >= previous
        and final["host_sources"] == host["host_sources"]
        and final["native_operators"] == host["native_operators"]
        and final["external_files_sha256"] == contract["external_files_sha256"]
        and final["model_stat_files_checked"] == len(contract["model_stats"]),
        "Final host or model-stat identity differs",
    )
    idle("npu-final.txt")
    require(
        read_json(raw["preregistration.json"])["contract_sha256"] == CONTRACT_V1
        and read_json(raw["pressure-policy-preregistration.json"])["contract_sha256"]
        == CONTRACT_V2,
        "Public preregistration identity differs",
    )
    amended_at = datetime.fromisoformat(
        read_json(raw["pressure-policy-amendment.json"])["created_at_utc"]
    ).timestamp()
    require(
        amended_at < runs["service2-off-1"]["times"]["started_at"],
        "Pressure amendment was not before pressure execution",
    )
    require(
        "socket" in read_json(raw["timeout-clarification.json"])["body"],
        "Socket timeout clarification missing",
    )
    quality = quality_comparison(results["quality8-off"], results["quality8-on"])
    return {
        "coverage": coverage,
        "lifecycles": runs,
        "runtime_snapshot_gaps": snapshots_missing,
        "quality_gate_passed": quality["gate_passed"],
        "model_weights_rehashed_after_series": final[
            "model_weights_rehashed_after_series"
        ],
    }
