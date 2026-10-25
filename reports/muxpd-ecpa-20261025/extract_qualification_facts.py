"""Extract bounded facts from immutable qualification receipts; no runtime imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


def record(path: Path) -> dict:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def json_log(path: Path):
    text = path.read_text()
    for match in re.finditer(r"(?m)^\s*\{", text):
        try:
            return json.loads(text[match.start() :])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No complete JSON object in {path}")


def natural_comparison(candidate: list, native: list) -> dict:
    if len(candidate) != len(native) or not candidate:
        raise ValueError("Natural request counts differ or are empty")
    rows = []
    for left, right in zip(candidate, native, strict=True):
        if left["case"] != right["case"] or left["payload"] != right["payload"]:
            raise ValueError("Natural payload/case mismatch")
        lc, rc = left["response"]["choices"][0], right["response"]["choices"][0]
        lt, rt = lc["message"]["content"], rc["message"]["content"]
        rows.append(
            {
                "case": left["case"],
                "payload_equal": True,
                "text_exact": lt == rt,
                "candidate_text_sha256": hashlib.sha256(lt.encode()).hexdigest(),
                "native_text_sha256": hashlib.sha256(rt.encode()).hexdigest(),
                "candidate_http_status": left["http_status"],
                "native_http_status": right["http_status"],
                "candidate_check_passed": left["passed"],
                "native_check_passed": right["passed"],
                "candidate_finish_reason": lc["finish_reason"],
                "native_finish_reason": rc["finish_reason"],
                "candidate_usage": left["response"]["usage"],
                "native_usage": right["response"]["usage"],
                "token_ids_observed": lc.get("token_ids") is not None
                and rc.get("token_ids") is not None,
            }
        )
    return {
        "requests_per_role": len(rows),
        "exact_text_count": sum(r["text_exact"] for r in rows),
        "rows": rows,
        "scope": "Text comparison only; no token equivalence or cause inferred.",
    }


def extract(final: Path, failed: Path, cpu: Path) -> dict:
    sources = {}

    def read(root: Path, name: str, *, log: bool = False):
        path = root / name
        sources[str(path)] = record(path)
        return json_log(path) if log else json.loads(path.read_text())

    qualification = read(final, "qualification.json")
    initial = read(failed, "qualification.json")
    cpu_result = read(cpu, "validation.json")
    cpu_index = read(cpu, "receipt-files.json")
    for item in cpu_index["files"]:
        path = cpu / item["path"]
        if not path.resolve().is_relative_to(cpu.resolve()):
            raise ValueError("CPU receipt escapes root")
        observed = record(path)
        if any(observed[key] != item[key] for key in ("bytes", "sha256")):
            raise ValueError(f"CPU receipt mismatch: {item['path']}")
        sources[str(path)] = observed
    for item in cpu_result["test_sources"]:
        path = cpu / "tests" / item["name"]
        observed = record(path)
        if observed["sha256"] != item["sha256"]:
            raise ValueError(f"CPU copied test mismatch: {item['name']}")
        sources[str(path)] = observed

    natural = natural_comparison(
        read(final, "candidate-natural.json"), read(final, "native-natural.json")
    )
    recorded = qualification["natural_exact_matches"]
    if [r["text_exact"] for r in natural["rows"]] != [r["exact"] for r in recorded]:
        raise ValueError("Controller natural comparison differs from raw response text")
    smoke = {}
    for name in ("candidate-smoke", "after-cancel-smoke", "native-smoke"):
        receipt = read(final, name + ".json")
        smoke[name] = {
            "reported_passed": receipt["passed"],
            "test_count": len(receipt["tests"]),
            "passed_count": sum(t["passed"] is True for t in receipt["tests"]),
            "tests": [
                {"name": t["name"], "passed": t["passed"]} for t in receipt["tests"]
            ],
        }
    statuses = {}
    for name in (
        "installed-disabled-status",
        "enabled-before-start-status",
        "healthy-before-request-status",
        "candidate-effective-status",
        "candidate-stopped-status",
        "disabled-status",
        "native-running-status",
    ):
        receipt = read(final, name + ".log", log=True)
        statuses[name] = {
            key: receipt[key] for key in ("extension_id", "states", "evidence")
        }

    graphs = []
    candidate_log = final / "candidate-service.log"
    sources[str(candidate_log)] = record(candidate_log)
    for line_no, line in enumerate(candidate_log.read_text().splitlines(), 1):
        match = re.search(r"MuxPD decode graph final rank=(\d+) (\{.*\})$", line)
        if match:
            graphs.append(
                {
                    "rank": int(match[1]),
                    "line": line_no,
                    "counters": ast.literal_eval(match[2]),
                }
            )
    if len({row["rank"] for row in graphs}) != len(graphs):
        raise ValueError("Repeated final graph record for a rank")
    warnings = []
    native_log = final / "native-rollback-service.log"
    sources[str(native_log)] = record(native_log)
    for line_no, line in enumerate(native_log.read_text().splitlines(), 1):
        if "EngineDeadError" in line or "resource_tracker: There appear to be" in line:
            warnings.append({"line": line_no, "text": line})
    npu = {}
    for name in ("npu-before", "npu-after-candidate", "npu-after-native"):
        path = final / (name + ".log")
        sources[str(path)] = record(path)
        npu[name] = {
            "reported_no_process_devices": [
                int(x)
                for x in re.findall(
                    r"No running processes found in NPU (\d+)", path.read_text()
                )
            ],
            "scope": "npu-smi process-table statements, not zero allocated HBM",
        }
    dry_run = read(final, "native-dry-run.log", log=True)
    selectors = ("--scheduler-cls", "--worker-cls", "--batch-admission-policy")
    commands = cpu_result["commands"]
    cpu_test = cpu / "installed-pure-tests.stdout"
    pinned_test = final / "full-pinned-tests.log"
    sources[str(pinned_test)] = record(pinned_test)
    return {
        "schema": "muxpd-ecpa-qualification-facts-v1",
        "extractor": record(Path(__file__).resolve()),
        "scope": "Functional qualification of the captured builds, not performance or universal numerical equivalence.",
        "qualification": {
            key: qualification[key]
            for key in (
                "scope",
                "passed",
                "errors",
                "git_commit",
                "started_at_utc",
                "finished_at_utc",
            )
        },
        "steps": [
            {key: step[key] for key in ("name", "exit_code", "passed", "log")}
            for step in qualification["steps"]
        ],
        "services": qualification["services"],
        "build_identities": read(final, "build-identities.json"),
        "model_host_provenance": read(final, "model-host-provenance.json"),
        "configuration": read(final, "configuration.json"),
        "smoke": smoke,
        "natural": natural,
        "manager_statuses": statuses,
        "trace_audit": read(final, "trace-audit.log", log=True),
        "client_probes": {
            name: {
                key: value
                for key, value in read(final, name + ".json").items()
                if key
                in (
                    "passed",
                    "cancel_prefill",
                    "cancel_delay_s",
                    "decode_content_events_before_prefill_first_token",
                )
            }
            for name in ("overlap", "prefill-cancel")
        },
        "graph_final_snapshots": graphs,
        "graph_scope": "Host path counters include startup warmups; not kernel concurrency or performance evidence.",
        "native_rollback": {
            "dry_run_muxpd_selectors": [
                x for x in dry_run["command"] if x in selectors or "vllm_muxpd" in x
            ],
            "native_extension_manifests": dry_run["native_extension_manifests"],
            "shutdown_warnings": warnings,
            "warning_scope": "Warnings retained from SIGINT teardown after request checks; shutdown is not warning-free.",
        },
        "npu_process_tables": npu,
        "cpu": {
            "reported_passed": cpu_result["passed"],
            "command_count": len(commands),
            "expected_returncodes_match": all(
                c["returncode"] == c["expected_returncode"] for c in commands
            ),
            "commands": [
                {key: c[key] for key in ("label", "returncode", "expected_returncode")}
                for c in commands
            ],
            "native_packages_absent": cpu_result["native_packages_absent"],
            "package_file_counts": {
                k: v["file_count"] for k, v in cpu_result["packages"].items()
            },
            "indexed_files_verified": len(cpu_index["files"]),
            "copied_tests_verified": len(cpu_result["test_sources"]),
            "test_summary_lines": re.findall(
                r"(?m)^.*\d+ passed.*$", cpu_test.read_text()
            ),
            "pinned_source_test_summary_lines": re.findall(
                r"(?m)^.*\d+ passed.*$", pinned_test.read_text()
            ),
        },
        "retained_failed_attempt": {
            key: initial[key]
            for key in (
                "passed",
                "errors",
                "git_commit",
                "started_at_utc",
                "finished_at_utc",
            )
        },
        "sources": list(sources.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("final", "failed", "cpu", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.final, args.failed, args.cpu)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
