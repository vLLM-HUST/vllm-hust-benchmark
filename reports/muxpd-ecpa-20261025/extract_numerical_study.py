"""Reduce saved response texts without treating HTTP success as a quality oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reduce_study(root: Path) -> dict:
    sources = []

    def read(name: str):
        path = root / name
        raw = path.read_bytes()
        sources.append({"path": str(path), "bytes": len(raw), "sha256": sha(raw)})
        return json.loads(raw)

    receipt = read("qualification.json")
    if not receipt.get("finished_at_utc"):
        raise ValueError("Study is still active")
    plan = receipt["plan"]
    roles = plan["endpoint_order"]
    if roles != ["native", "candidate"]:
        raise ValueError("Unexpected endpoint order")
    concurrent, isolated, rows, payloads = {}, {}, [], {}
    for role in roles:
        concurrent[role] = []
        for repeat in range(plan["concurrent_rounds"]):
            values = read(f"{role}-round-{repeat}-natural.json")
            values.sort(key=lambda row: row["case"])
            if [r["case"] for r in values] != list(range(plan["cases_per_round"])):
                raise ValueError("Missing or duplicated concurrent case")
            concurrent[role].append(values)
        isolated[role] = read(f"{role}-isolated.json")
        if [r["repeat"] for r in isolated[role]] != list(
            range(plan["isolated_cache_explanations"])
        ):
            raise ValueError("Missing or duplicated isolated repeat")
        batches = [
            ("concurrent", repeat, values)
            for repeat, values in enumerate(concurrent[role])
        ]
        batches += [("isolated", None, isolated[role])]
        for phase, repeat, values in batches:
            for value in values:
                case = value["case"] if phase == "concurrent" else 2
                payload = value["payload"]
                if payloads.setdefault(case, payload) != payload:
                    raise ValueError("Payload changed across repeats/roles")
                for key in ("temperature", "seed", "max_tokens"):
                    if payload[key] != plan[key]:
                        raise ValueError(
                            "Payload differs from registered sampling parameters"
                        )
                response = value["response"]
                choice = response["choices"][0]
                text = choice["message"]["content"]
                passed = (
                    value["http_status"] == 200 and isinstance(text, str) and bool(text)
                )
                if value["passed"] is not passed:
                    raise ValueError(
                        "Producer passed field differs from its HTTP/nonempty criterion"
                    )
                rows.append(
                    {
                        "role": role,
                        "phase": phase,
                        "repeat": repeat if phase == "concurrent" else value["repeat"],
                        "case": case,
                        "http_status": value["http_status"],
                        "nonempty_http_success": passed,
                        "text_sha256": sha(text.encode()),
                        "finish_reason": choice["finish_reason"],
                        "usage": response["usage"],
                        "token_ids_present": choice.get("token_ids") is not None,
                        "logprobs_present": choice.get("logprobs") is not None,
                    }
                )

    def text(value):
        return value["response"]["choices"][0]["message"]["content"]

    summary = {
        role: {
            "concurrent_distinct_texts_by_case": {
                str(case): len({text(batch[case]) for batch in concurrent[role]})
                for case in range(plan["cases_per_round"])
            },
            "isolated_cache_distinct_texts": len({text(row) for row in isolated[role]}),
        }
        for role in roles
    }
    summary["cross_endpoint_round_exact_matches"] = [
        sum(text(a) == text(b) for a, b in zip(left, right, strict=True))
        for left, right in zip(
            concurrent["native"], concurrent["candidate"], strict=True
        )
    ]
    summary["isolated_cross_endpoint_exact_matches"] = sum(
        text(a) == text(b)
        for a, b in zip(isolated["native"], isolated["candidate"], strict=True)
    )
    if summary != receipt["summary"]:
        raise ValueError("Recomputed summary differs from captured controller summary")
    for name in (
        "repeat_outputs.py",
        "helpers.py",
        "configuration.json",
        "native-service.log",
        "candidate-service.log",
    ):
        path = root / name
        raw = path.read_bytes()
        sources.append({"path": str(path), "bytes": len(raw), "sha256": sha(raw)})
    return {
        "schema": "muxpd-ecpa-numerical-study-facts-v1",
        "extractor_sha256": sha(Path(__file__).read_bytes()),
        "issue": "https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23",
        "controller": {
            key: receipt[key]
            for key in (
                "passed",
                "errors",
                "started_at_utc",
                "finished_at_utc",
                "scope",
            )
        },
        "artifact_sources": receipt["artifact_sources"],
        "plan": plan,
        "services": receipt["services"],
        "request_count": len(rows),
        "nonempty_http_success_count": sum(r["nonempty_http_success"] for r in rows),
        "summary": summary,
        "concurrent_exact_total": sum(summary["cross_endpoint_round_exact_matches"]),
        "concurrent_pairs": plan["concurrent_rounds"] * plan["cases_per_round"],
        "requests": rows,
        "limitations": [
            "Saved-response reduction, not a raw wire audit or semantic quality oracle.",
            "No output token IDs or logits retained; text differences do not identify a cause.",
            "One service startup per role, native then candidate; arrivals and device batches not fixed.",
            "Native variation does not explain away candidate effects; no equivalence or speedup claim.",
        ],
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reduce_study(args.study)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
