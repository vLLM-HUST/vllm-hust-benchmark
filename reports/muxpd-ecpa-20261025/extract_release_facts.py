"""Bind the later hardware run to the final wheel through explicit runtime bytes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

from extract_qualification_facts import json_log, natural_comparison, record


def extract(hardware: Path, cpu: Path) -> dict:
    sources = {}

    def source(path):
        result = record(path)
        sources[str(path)] = result
        return result

    def load(root, name, log=False):
        path = root / name
        source(path)
        return json_log(path) if log else json.loads(path.read_text())

    run = load(hardware, "qualification.json")
    if not run.get("finished_at_utc"):
        raise ValueError("Hardware run has not terminated")
    builds = load(hardware, "build-identities.json")
    cpu_result = load(cpu, "validation.json")
    cpu_index = load(cpu, "receipt-files.json")
    for row in cpu_index["files"]:
        path = cpu / row["path"]
        if not path.resolve().is_relative_to(cpu.resolve()):
            raise ValueError("CPU index escapes root")
        observed = source(path)
        if any(observed[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("CPU indexed file changed")
    bridge = cpu_result["manager_hardware_runtime_bridge"]
    archives = {}
    for key in ("hardware_wheel", "final_wheel"):
        item = bridge[key]
        path = Path(item["path"])
        if source(path)["sha256"] != item["sha256"]:
            raise ValueError("Wheel digest differs from bridge")
        with zipfile.ZipFile(path) as stream:
            archives[key] = {name: stream.read(name) for name in stream.namelist()}
    runtime_names = {
        name for name in archives["hardware_wheel"] if name.startswith("vllm_hust_ext/")
    }
    if runtime_names != {
        name for name in archives["final_wheel"] if name.startswith("vllm_hust_ext/")
    }:
        raise ValueError("Runtime package member set changed")
    if runtime_names != {row["path"] for row in bridge["runtime_files"]}:
        raise ValueError("Bridge omits runtime files")
    for row in bridge["runtime_files"]:
        old, new = (
            archives[key][row["path"]] for key in ("hardware_wheel", "final_wheel")
        )
        if (
            old != new
            or len(new) != row["bytes"]
            or hashlib.sha256(new).hexdigest() != row["sha256"]
        ):
            raise ValueError("Runtime byte bridge differs")
    if builds["manager"]["wheel_sha256"] != bridge["hardware_wheel"]["sha256"]:
        raise ValueError("Bridge is not bound to the actual hardware Manager")
    if builds["manager"]["commit"] != bridge["hardware_commit"]:
        raise ValueError("Hardware Manager source differs")
    if set(run["artifacts"].values()).isdisjoint({bridge["hardware_wheel"]["sha256"]}):
        raise ValueError("Hardware controller did not record the bridged wheel")
    natural = natural_comparison(
        load(hardware, "candidate-natural.json"), load(hardware, "native-natural.json")
    )
    if [r["text_exact"] for r in natural["rows"]] != [
        r["exact"] for r in run["natural_exact_matches"]
    ]:
        raise ValueError("Hardware natural result differs")
    smoke = {}
    for name in ("candidate-smoke", "after-cancel-smoke", "native-smoke"):
        value = load(hardware, name + ".json")
        smoke[name] = {
            "reported_passed": value["passed"],
            "test_count": len(value["tests"]),
            "passed_count": sum(row["passed"] is True for row in value["tests"]),
        }
    statuses = {
        name: load(hardware, name + ".log", True)
        for name in (
            "healthy-before-request-status",
            "candidate-effective-status",
            "candidate-stopped-status",
            "disabled-status",
            "native-running-status",
        )
    }
    graph, warnings, npu = [], [], {}
    for filename in ("candidate-service.log", "native-rollback-service.log"):
        path = hardware / filename
        source(path)
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = re.search(r"MuxPD decode graph final rank=(\d+) (\{.*\})$", line)
            if match:
                graph.append(
                    {
                        "rank": int(match[1]),
                        "line": number,
                        "counters": ast.literal_eval(match[2]),
                    }
                )
            if (
                "EngineDeadError" in line
                or "resource_tracker: There appear to be" in line
            ):
                warnings.append({"file": filename, "line": number, "text": line})
    for name in ("npu-before", "npu-after-candidate", "npu-after-native"):
        path = hardware / (name + ".log")
        source(path)
        npu[name] = [
            int(x)
            for x in re.findall(
                r"No running processes found in NPU (\d+)", path.read_text()
            )
        ]
    manager_tests = load(cpu, "installed-manager-tests.json")
    pure_text = (cpu / "installed-pure-tests.stdout").read_text()
    return {
        "schema": "muxpd-release-qualification-facts-v1",
        "extractor": source(Path(__file__).resolve()),
        "qualification": {
            key: run[key]
            for key in (
                "scope",
                "passed",
                "errors",
                "started_at_utc",
                "finished_at_utc",
            )
        },
        "build_identities": builds,
        "artifact_sha256": run["artifacts"],
        "services": run["services"],
        "smoke": smoke,
        "natural": natural,
        "manager_statuses": statuses,
        "trace_audit": load(hardware, "trace-audit.log", True),
        "graph_final_snapshots": graph,
        "model_host": load(hardware, "model-host-provenance.json"),
        "configuration": load(hardware, "configuration.json"),
        "native_dry_run": load(hardware, "native-dry-run.log", True),
        "shutdown_warnings": warnings,
        "no_running_npu_processes_reported": npu,
        "cpu": {
            "validation": source(cpu / "validation.json"),
            "reported_passed": cpu_result["passed"],
            "command_count": len(cpu_result["commands"]),
            "indexed_files_verified": len(cpu_index["files"]),
            "command_returncodes_match": all(
                r["returncode"] == r["expected_returncode"]
                for r in cpu_result["commands"]
            ),
            "native_packages_absent": cpu_result["native_packages_absent"],
            "plugin_test_summary_lines": re.findall(r"(?m)^.*\d+ passed.*$", pure_text),
            "installed_manager_test_receipt": manager_tests,
            "all_subprocesses_terminal": cpu_result["all_subprocesses_terminal"],
        },
        "manager_runtime_bridge": bridge,
        "independently_compared_runtime_files": len(runtime_names),
        "boundaries": [
            "ac0 wheel was used on hardware; b78 wheel was not relabelled as a hardware run.",
            "All runtime package members compared directly; wheel metadata has a separate identity.",
            "No assertion that the current worktree is clean; source-object audit is scoped to listed package members.",
            "No whole-repository CI pass, numerical equivalence or performance improvement is inferred.",
        ],
        "sources": list(sources.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hardware", "cpu", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.hardware, args.cpu)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
