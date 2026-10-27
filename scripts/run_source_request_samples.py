"""Replay three preselected original requests, never synthesize or shorten them.

Output is PRIVATE: responses can contain enterprise data. This tests protocol
integration, not task quality, official scoring, representativeness or performance.
"""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError

from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.execution import replay_request
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_benchmark.workload_response_validation import check_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--service-profile",
        help="Declared server configuration identity; not hardware proof",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Socket read timeout, not an end-to-end performance deadline",
    )
    args = parser.parse_args()
    binding = bind_source(
        args.source_id,
        data_root=args.data_root,
        files=args.files,
        revision=args.revision,
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = args.output / "binding.json"
    manifest.write_text(json.dumps(binding), encoding="utf-8")
    run(
        {
            "sources": [
                {
                    "source_id": args.source_id,
                    "data_root": str(args.data_root),
                    "manifest": str(manifest),
                    "limit": 3,
                }
            ]
        },
        output=args.output / "prepared",
    )
    database = args.output / "prepared/workload.sqlite"
    selected = list(prepared_rows(database))
    if len(selected) != 3:
        raise ValueError("exactly three preselected records required")
    records = [
        resolve_input(database, row["record_id"], data_root=args.data_root)
        for row in selected
    ]
    if any(record.runtime["kind"] != "request_replay" for record in records):
        raise ValueError(
            "only original request_replay sources; no task renderer fallback"
        )
    receipt = {
        "schema_version": "source-request-validation/v1",
        "source_id": args.source_id,
        "selection": "FIRST_3_ORIGINAL_RECORDS_BEFORE_EXECUTION",
        "selected_samples": [row["record_id"] for row in selected],
        "model": args.model,
        "request_change": "EXPLICIT_MODEL_ROUTING_ONLY",
        "scoring": "NO_INDEPENDENT_ORACLE_SUPPLIED",
        "status": "STARTED",
        "samples": [],
        "service_profile": args.service_profile or "NOT_DECLARED",
        "quality_claim": False,
    }
    target = args.output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    for row, record in zip(selected, records):
        item = {
            "record_id": row["record_id"],
            "original_model": record.request_body().get("model"),
            "original_stream": record.request_body().get("stream"),
            "original_max_tokens": record.request_body().get(
                "max_tokens", record.request_body().get("max_completion_tokens")
            ),
            "status": "NOT_RUN",
        }
        try:
            response = replay_request(
                record, endpoint=args.endpoint, model=args.model, timeout=args.timeout
            )
            with (args.output / (row["record_id"] + ".private.json")).open(
                "x", encoding="utf-8"
            ) as output:
                json.dump(response, output, ensure_ascii=False)
            events = response["events"] if response.get("stream") else [response]
            reasons = [
                choice["finish_reason"]
                for event in events
                for choice in event.get("choices", [])
                if choice.get("finish_reason")
            ]
            item.update(
                status="REQUEST_PROTOCOL_COMPLETED"
                if reasons
                else "RESPONSE_WITHOUT_TERMINATION",
                finish_reasons=reasons,
                model_quality="NOT_EVALUATED",
                budget_exhausted="length" in reasons,
                response_validation=check_response(response, record.request_body()),
            )
        except HTTPError as error:
            # Keep the server error private; it may echo original input.
            with (args.output / (row["record_id"] + ".private-error.txt")).open(
                "x", encoding="utf-8"
            ) as output:
                output.write(error.read().decode("utf-8", errors="replace"))
            item.update(
                status="SERVER_REJECTED_ORIGINAL_REQUEST", http_status=error.code
            )
        except Exception as error:
            item.update(
                status="HARNESS_OR_PROTOCOL_ERROR", error_type=type(error).__name__
            )
        receipt["samples"].append(item)
        target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    verified = all(
        s["status"] == "REQUEST_PROTOCOL_COMPLETED"
        and s.get("response_validation", {}).get("status")
        == "RESPONSE_INTEGRITY_VERIFIED"
        for s in receipt["samples"]
    )
    receipt["status"] = (
        "REQUESTS_EXECUTED_AND_INTEGRITY_VALIDATED_NO_QUALITY_ORACLE"
        if verified
        else "PARTIAL_OR_FAILED"
    )
    if verified:
        receipt["scoring"] = "NON_SCORING_INTEGRITY_VERIFIED"
    target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "status": receipt["status"],
                "samples": receipt["samples"],
            }
        )
    )
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
