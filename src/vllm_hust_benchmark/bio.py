"""Run and audit biological representative-model Benchmarks across repositories."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from vllm_hust_benchmark.bio_report import render_markdown, summarize
from vllm_hust_benchmark.models import render_parameter_flags

BENCHMARK_ROOT = Path(__file__).resolve().parents[2]


def configure_scientist(path):
    path = Path(path).resolve()
    if not (path / "experiments/benchmark/representative/__main__.py").is_file():
        raise ValueError(
            "The selected scientist checkout has no representative-model driver"
        )
    sys.path.insert(0, str(path))
    return path


def source_snapshot(scientist, registry=None):
    from experiments.benchmark.contracts import file_digest
    from experiments.benchmark.phase1 import source_manifest

    from experiments.benchmark.representative.suite import load_registry

    registry = registry or load_registry()
    scientist_files = source_manifest()
    for path in [
        *(scientist / "experiments/benchmark/representative").glob("*.py"),
        *(scientist / "docs/benchmark").glob("phase2*"),
        scientist / "architectures/transformer/encoder/bio_encoder.py",
        scientist / "architectures/transformer/decoder/progen2.py",
        scientist / "architectures/transformer/decoder/reference_rotary.py",
        scientist / "runtime/registration/decoder/gptj_adapters.py",
        scientist / "tests/unit/test_bio_encoder.py",
        scientist / "tests/unit/test_progen_attention.py",
        scientist / "tests/unit/test_benchmark_representative.py",
        scientist / "tests/unit/test_benchmark_registered.py",
    ]:
        if path.is_file():
            scientist_files[str(path.relative_to(scientist))] = file_digest(path)
    benchmark_files = [
        "src/vllm_hust_benchmark/bio.py",
        "src/vllm_hust_benchmark/bio_report.py",
        "src/vllm_hust_benchmark/bio_pair.py",
        "src/vllm_hust_benchmark/models.py",
        "pyproject.toml",
        "tests/test_bio_benchmark.py",
        "docs/BIO_NATIVE_BENCHMARK.zh-CN.md",
        "README.md",
    ]
    for path in [
        *(scientist / "docs/benchmark").glob("registered_*"),
        scientist / "docs/benchmark/native_registry.json",
    ]:
        scientist_files[str(path.relative_to(scientist))] = file_digest(path)
    benchmark_files += [
        "src/vllm_hust_benchmark/bio_registry.py",
        "src/vllm_hust_benchmark/bio_interfaces.py",
        "src/vllm_hust_benchmark/data/bio_registry.schema.json",
        "tests/test_bio_registry.py",
        "docs/BIO_ONBOARDING.zh-CN.md",
    ]
    for relative in registry.document.get("scientist_sources", []):
        path = (scientist / relative).resolve(strict=True)
        if not path.is_relative_to(scientist) or not path.is_file():
            raise ValueError(f"Invalid registered Scientist source: {relative}")
        scientist_files[relative] = file_digest(path)
    for path in registry.source_files():
        if path.is_relative_to(scientist):
            scientist_files[str(path.relative_to(scientist))] = file_digest(path)
        elif path.is_relative_to(BENCHMARK_ROOT):
            benchmark_files.append(str(path.relative_to(BENCHMARK_ROOT)))
        elif path != registry.source:
            raise ValueError(
                "Registered component source must belong to a selected source checkout"
            )
    return {
        "registry": registry.identity(),
        "asset_definitions_sha256": file_digest(
            registry.source.parent / registry.document["asset_manifest"]
        ),
        "scientist": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=scientist, text=True
            ).strip(),
            "files": scientist_files,
        },
        "benchmark": {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=BENCHMARK_ROOT, text=True
            ).strip(),
            "files": {
                name: file_digest(BENCHMARK_ROOT / name)
                for name in benchmark_files
                if (BENCHMARK_ROOT / name).is_file()
            },
        },
    }


def archive_sources(snapshot, scientist, output):
    with tarfile.open(output / "sources.tar.gz", "w:gz") as archive:
        for label, root in [("scientist", scientist), ("benchmark", BENCHMARK_ROOT)]:
            for relative in sorted(snapshot[label]["files"]):
                archive.add(
                    root / relative, arcname=label + "/" + relative, recursive=False
                )


def child_command(args, scientist, mode, item):
    flags = {
        "suite": str(args.suite.resolve()),
        "registry": str(args.registry.resolve()),
        "data": str(args.data.resolve()),
        "assets": str(args.assets.resolve()),
        "upstream": str(args.upstream.resolve()),
        "output": str((args.output / item["directory"]).resolve()),
        "profile": item["profile_id"],
        "start_index": item["start_index"],
        "pilot": args.pilot,
        "fault": item["fault"],
    }
    return [
        str(args.python),
        "-m",
        "experiments.benchmark.representative",
        mode,
        *render_parameter_flags(flags),
    ]


def run(args, scientist):
    from experiments.benchmark.audit import (
        check_assets,
        run_process,
        runtime_environment,
    )
    from experiments.benchmark.contracts import file_digest, write_json
    from experiments.benchmark.phase1 import probe_environment, worker_environment
    from experiments.benchmark.representative.suite import validate_suite

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output = output
    registry = args._registry
    suite, cases = validate_suite(args.suite, args.data, registry)
    protocol = dict(suite["protocol"])
    if args.pilot:
        protocol.update(
            independent_starts=1, warmup_repetitions=1, steady_repetitions=2
        )
    profiles = sorted({case["profile_id"] for case in cases})
    runs = []
    for profile in profiles:
        runs.append(
            {
                "profile_id": profile,
                "start_index": -1,
                "fault": True,
                "directory": profile + "/fault",
            }
        )
        for start in range(protocol["independent_starts"]):
            runs.append(
                {
                    "profile_id": profile,
                    "start_index": start,
                    "fault": False,
                    "directory": profile + "/start" + str(start),
                }
            )
    initial = source_snapshot(scientist, registry)
    manifest = {
        "schema_version": 2,
        "registry": registry.identity(),
        "registry_source": str(args.registry.resolve()),
        "pilot": args.pilot,
        "sources": initial,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_sha256": file_digest(args.suite),
        "effective_protocol": protocol,
        "resolved_cases": cases,
        "runs": runs,
        "argv": sys.argv,
        "scope": "biological native-task representative-model project evaluation",
        "roots": {"scientist": str(scientist), "benchmark": str(BENCHMARK_ROOT)},
    }
    write_json(output / "manifest.json", manifest)
    shutil.copyfile(args.suite, output / "suite.json")
    shutil.copyfile(args.registry, output / "registry.json")
    asset_file = registry.source.parent / registry.document["asset_manifest"]
    shutil.copyfile(asset_file, output / registry.document["asset_manifest"])
    (output / "data").mkdir()
    for definition in suite["data"].values():
        shutil.copyfile(
            args.data / definition["file"], output / "data" / definition["file"]
        )
    archive_sources(initial, scientist, output)
    checks, executions = [], []

    def record(name, passed, details):
        checks.append({"name": name, "passed": bool(passed), "details": details})
        print(json.dumps(checks[-1], ensure_ascii=False), flush=True)
        write_json(
            output / "progress.json", {"checks": checks, "executions": executions}
        )

    try:
        record(
            "suite_contract",
            True,
            {"cases": len(cases), "data_records": len(suite["data"])},
        )
        for label, root, pattern in [
            ("scientist", scientist, "test_benchmark_representative.py"),
            ("benchmark", BENCHMARK_ROOT, "test_bio_benchmark.py"),
            ("scientist_registry", scientist, "test_benchmark_registered.py"),
            ("benchmark_registry", BENCHMARK_ROOT, "test_bio_registry.py"),
        ]:
            environment = worker_environment()
            environment["PYTHONPATH"] = (
                str(scientist)
                + os.pathsep
                + str(BENCHMARK_ROOT / "src")
                + os.pathsep
                + environment["PYTHONPATH"]
            )
            directory = (
                "tests/unit"
                if label.startswith("scientist")
                else str(BENCHMARK_ROOT / "tests")
            )
            command = [
                str(args.python),
                "-m",
                "unittest",
                "discover",
                "-s",
                directory,
                "-p",
                pattern,
                "-v",
            ]
            unit = run_process(
                command,
                log=output / (label + "-unit.log"),
                timeout=120,
                env=environment,
            )
            record(label + "_unit_regression", unit["exit_code"] == 0, unit)
            if unit["exit_code"] != 0:
                raise ValueError("Representative-model protocol regression failed")
        existing = run_process(
            [
                str(args.python),
                "-m",
                "pytest",
                "-q",
                "tests/unit/test_bio_encoder.py",
                "tests/unit/test_progen_attention.py",
            ],
            log=output / "existing-model-unit.log",
            timeout=180,
            env=environment,
        )
        record("existing_model_unit_regression", existing["exit_code"] == 0, existing)
        if existing["exit_code"] != 0:
            raise ValueError("Existing scientific adapter unit regression failed")
        assets = check_assets(
            args.assets, args.upstream, registry.asset_definitions(profiles)
        )
        write_json(output / "assets.json", assets)
        record(
            "assets",
            assets["passed"],
            "Original weights, runtime views and fixed author source",
        )
        environment = probe_environment(output, 90)
        record("environment_and_device", environment["passed"], environment["errors"])
        if not assets["passed"] or not environment["passed"]:
            raise ValueError("Asset or device prerequisite failed")
        effective = worker_environment()
        effective["PYTHONPATH"] = (
            str(scientist)
            + os.pathsep
            + str(BENCHMARK_ROOT / "src")
            + os.pathsep
            + effective.get("PYTHONPATH", "")
        )
        if registry.document["legacy_phase1_regression"]:
            regression = run_process(
                [
                    str(args.python),
                    "-m",
                    "experiments.benchmark.phase1",
                    "accept",
                    "--assets",
                    str(args.assets.resolve()),
                    "--upstream",
                    str(args.upstream.resolve()),
                    "--output",
                    str(output / "phase1-regression"),
                ],
                log=output / "phase1-regression.log",
                timeout=1800,
                env=effective,
            )
            record("phase1_regression", regression["exit_code"] == 0, regression)
            if regression["exit_code"] != 0:
                raise ValueError(
                    "Existing phase-1 native and reference regression failed"
                )
        for item in runs:
            for mode in ["native"] if item["fault"] else ["native", "reference"]:
                logfile = item["directory"].replace("/", "-") + "-" + mode + ".log"
                command = child_command(args, scientist, mode, item)
                execution = run_process(
                    command,
                    log=output / logfile,
                    timeout=protocol["timeout_seconds"],
                    env=effective,
                )
                executions.append(
                    item
                    | {
                        "mode": mode,
                        "log": logfile,
                        "result": execution,
                        "command": command,
                    }
                )
                native_path = (
                    output
                    / item["directory"]
                    / ("native.json" if mode == "native" else "comparison.json")
                )
                payload = (
                    json.loads(native_path.read_text()) if native_path.is_file() else {}
                )
                if item["fault"]:
                    passed = (
                        execution["exit_code"] == 1
                        and payload.get("closed")
                        and payload.get("error", {}).get("message")
                        == "benchmark_injected_host_failure_after_native_output"
                    )
                else:
                    passed = execution["exit_code"] == 0 and payload.get("passed")
                record(item["directory"] + "/" + mode, passed, execution)
                if not passed:
                    raise ValueError("Representative task subprocess failed")
        summary = summarize(output)
        write_json(output / "summary.json", summary)
        (output / "report.md").write_text(render_markdown(summary))
        record(
            "native_numerical_statistics_and_cleanup",
            True,
            {
                "requests": summary["request_count"],
                "numerical_checks": summary["numerical_checks"],
                "rejections": summary["rejected_inputs"],
                "engines": len(summary["engines"]),
            },
        )
        recomputed = summarize(output)
        record(
            "report_recomputation",
            summary == recomputed,
            "Recomputed raw clocks, work, numerical outputs and statistics",
        )
        final_assets = check_assets(
            args.assets, args.upstream, registry.asset_definitions(profiles)
        )
        write_json(output / "assets-after.json", final_assets)
        record(
            "asset_consistency",
            final_assets == assets and final_assets["passed"],
            "Before/after asset equality",
        )
        final_environment = runtime_environment()
        write_json(output / "environment-after.json", final_environment)
        keys = [
            "versions",
            "tool_versions",
            "repositories",
            "installed_overlay",
            "registration",
            "system_versions",
            "device_nodes",
        ]
        record(
            "runtime_consistency",
            final_environment["passed"]
            and all(environment[key] == final_environment[key] for key in keys),
            "Before/after runtime identity",
        )
        validate_suite(args.suite, args.data, registry)
        record(
            "dataset_consistency",
            True,
            "Dataset sources and resolved samples revalidated",
        )
    except Exception as error:
        record(
            "execution_error",
            False,
            {"type": type(error).__name__, "message": str(error)},
        )
    record(
        "source_consistency",
        source_snapshot(scientist, registry) == initial,
        "Before/after sources and configurations",
    )
    result = {
        "schema_version": 1,
        "stage": "registered_tasks_pilot" if args.pilot else "registered_tasks",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "evidence": [
            {"path": str(path.relative_to(output)), "sha256": file_digest(path)}
            for path in sorted(output.rglob("*"))
            if path.is_file()
        ],
    }
    write_json(output / "result.json", result)
    return result["passed"]


def verify(output, scientist):
    from experiments.benchmark.contracts import file_digest
    from vllm_hust_benchmark.bio_report import read_json

    result = read_json(output / "result.json")
    manifest = read_json(output / "manifest.json")
    registry = None
    if "registry" in manifest:
        from vllm_hust_benchmark.bio_registry import Registry

        registry = Registry.load(output / "registry.json")
        current = Registry.load(manifest["registry_source"])
        if (
            registry.identity() != manifest["registry"]
            or current.identity() != registry.identity()
        ):
            raise ValueError("Registered bindings differ from the tested registry")
    required = {
        "suite_contract",
        "scientist_unit_regression",
        "benchmark_unit_regression",
        "existing_model_unit_regression",
        "phase1_regression",
        "assets",
        "environment_and_device",
        "native_numerical_statistics_and_cleanup",
        "report_recomputation",
        "asset_consistency",
        "runtime_consistency",
        "dataset_consistency",
        "source_consistency",
    }
    if registry is not None:
        required.update(
            {"scientist_registry_unit_regression", "benchmark_registry_unit_regression"}
        )
    if registry is not None and not registry.document["legacy_phase1_regression"]:
        required.remove("phase1_regression")
    names = [item["name"] for item in result["checks"]]
    if (
        len(set(names)) != len(names)
        or not required.issubset(names)
        or not result["passed"]
        or not all(item["passed"] for item in result["checks"])
    ):
        raise ValueError("Representative-model acceptance is incomplete")
    paths = [item["path"] for item in result["evidence"]]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate evidence paths")
    actual_paths = {
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path != output / "result.json"
    }
    if set(paths) != actual_paths:
        raise ValueError("The artifact index does not cover the complete run directory")
    for item in result["evidence"]:
        path = (output / item["path"]).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or file_digest(path) != item["sha256"]
        ):
            raise ValueError("Evidence file changed")
    manifest = read_json(output / "manifest.json")
    from experiments.benchmark.representative.suite import validate_suite

    suite, cases = validate_suite(output / "suite.json", output / "data", registry)
    expected_protocol = dict(suite["protocol"])
    if manifest["pilot"]:
        expected_protocol.update(
            independent_starts=1, warmup_repetitions=1, steady_repetitions=2
        )
    if (
        manifest["effective_protocol"] != expected_protocol
        or cases != manifest["resolved_cases"]
    ):
        raise ValueError(
            "The execution plan differs from the archived suite and source data"
        )
    if file_digest(output / "suite.json") != manifest["suite_sha256"]:
        raise ValueError("The archived suite differs from the executed configuration")
    if (
        source_snapshot(scientist, current if registry is not None else None)
        != manifest["sources"]
    ):
        raise ValueError(
            "Current source differs from the tested cross-repository snapshot"
        )
    for item in manifest["runs"]:
        for mode in ["native"] if item["fault"] else ["native", "reference"]:
            if item["directory"] + "/" + mode not in names:
                raise ValueError("Missing subprocess acceptance check")
    stage = "registered_tasks" if registry is not None else "representative_models"
    if result["stage"] != (stage + "_pilot" if manifest["pilot"] else stage):
        raise ValueError("Pilot and full-stage identities disagree")
    actual = summarize(output)
    if (
        actual != read_json(output / "summary.json")
        or render_markdown(actual) != (output / "report.md").read_text()
    ):
        raise ValueError(
            "Archived summary or report differs from recomputed raw records"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "run", "verify", "report"])
    parser.add_argument(
        "--scientist-root",
        type=Path,
        default=BENCHMARK_ROOT.parent / "scientist-lightweight-vllm-workspace",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    selected_python = os.path.abspath(
        shutil.which(str(args.python)) or str(args.python)
    )
    if selected_python != os.path.abspath(sys.executable):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = (
            str(BENCHMARK_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        os.execve(
            selected_python,
            [selected_python, "-m", "vllm_hust_benchmark.bio", *sys.argv[1:]],
            environment,
        )
    scientist = configure_scientist(args.scientist_root)
    from experiments.benchmark.representative.suite import (
        DEFAULT_REGISTRY,
        load_registry,
    )

    args.registry = args.registry or DEFAULT_REGISTRY
    args._registry = load_registry(args.registry)
    args.suite = args.suite or scientist / "docs/benchmark/registered_suite.json"
    args.data = args.data or scientist / ".local/acceptance/data"
    args.assets = args.assets or scientist / ".local/models"
    args.upstream = args.upstream or scientist / ".local/upstream"
    from experiments.benchmark.contracts import write_json
    from experiments.benchmark.representative.suite import validate_suite

    if args.command == "check":
        registry = args._registry
        suite, cases = validate_suite(args.suite, args.data, registry)
        print(
            json.dumps(
                {
                    "passed": True,
                    "suite": suite["id"],
                    "cases": len(cases),
                    "data_records": len(suite["data"]),
                }
            )
        )
        return
    if args.output is None:
        parser.error("--output is required")
    args.output = args.output.resolve()
    if args.command == "run":
        passed = run(args, scientist)
    elif args.command == "verify":
        passed = verify(args.output, scientist)["passed"]
    else:
        if args.destination is None:
            parser.error(
                "--destination is required for an independent report reconstruction"
            )
        summary = summarize(args.output)
        args.destination.mkdir(parents=True, exist_ok=False)
        write_json(args.destination / "summary.json", summary)
        (args.destination / "report.md").write_text(render_markdown(summary))
        passed = True
    print(
        json.dumps(
            {"command": args.command, "passed": passed, "output": str(args.output)}
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
