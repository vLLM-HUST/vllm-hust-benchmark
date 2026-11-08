"""Prepare original scientific cases and run them in an explicitly owned Docker.

Preparation emits publisher Dockerfiles and private case directories. Build the
images with these recipes, or supply previously qualified image IDs; run uses
only exact saved NPU CodeBERT scalars and never executes model programs on host.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from types import ModuleType
import uuid


ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/scienceagentbench-author-software-pins.json"
DATA_SHA = "c6f937863a220bd1762a00c20a0f79cc8dfca900b819bdb552150310731ae147"  # pragma: allowlist secret - original verified metadata


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def child_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("original relative asset path required")
    path = Path(root) / str(relative)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("asset escapes declared root")
    return path


def load_harness(source):
    profile = json.loads(PINS.read_text(encoding="utf-8"))
    for item in profile["files"]:
        path = child_path(source, item["path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"original software differs: {item['path']}")
    prefix = "_sab_pinned_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    package = ModuleType(prefix)
    package.__path__ = []
    sys.modules[prefix] = package
    modules = {}
    for name in ["constants", "dockerfiles", "test_spec"]:
        path = source / "evaluation/harness" / (name + ".py")
        module = ModuleType(prefix + "." + name)
        module.__package__ = prefix
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
        modules[name] = module
    return modules


def original_instance_recipe(factory, platform, base_image, program):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./:_@-]*", base_image):
        raise ValueError("explicit safe Docker base image identifier required")
    keys = [
        "OPENAI_API_KEY",
        "AZURE_OPENAI_KEY",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
    previous = {key: os.environ.pop(key) for key in keys if key in os.environ}
    try:
        return factory.get_dockerfile_instance(platform, base_image, program)
    finally:
        os.environ.update(previous)


def prepare(args):
    if digest(args.metadata) != DATA_SHA:
        raise ValueError("original verified Parquet identity required")
    import pyarrow.parquet as pq

    rows = pq.read_table(args.metadata).to_pylist()
    if len(rows) != 102 or args.start < 0 or args.limit <= 0 or args.start >= len(rows):
        raise ValueError("valid original102-row selection required")
    if args.full and (args.start != 0 or args.limit != 3):
        raise ValueError("do not combine full and range selection")
    selected = rows if args.full else rows[args.start : args.start + args.limit]
    modules = load_harness(args.source_dir)
    wrapper = Path(__file__).with_name("scienceagentbench_isolated_scoring.py")
    spec = importlib.util.spec_from_file_location("_sab_scalar_check", wrapper)
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    metric_receipt = json.loads(args.npu_receipt.read_text(encoding="utf-8"))
    scalar = metric_module.verified_npu_scalar(metric_receipt)
    args.work_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {
        "source_id": "SCIENCEAGENTBENCH",
        "status": "CASES_PREPARED_NOT_EXECUTED",
        "source_checkbox_eligible": False,
        "original_metadata_sha256": DATA_SHA,
        "cases": [],
    }
    cache = args.work_dir / "_asset_cache"
    cache.mkdir(mode=0o700)
    assets = {}

    def asset(relative, target):
        source = child_path(args.asset_root, relative)
        if source.is_symlink() or not source.is_file():
            raise ValueError("original regular asset file required")
        if relative not in assets:
            identity = digest(source)
            stored = child_path(cache, relative)
            stored.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, stored)
            if digest(stored) != identity:
                raise ValueError("copied original asset differs")
            stored.chmod(0o444)
            assets[relative] = (stored, identity)
        stored, identity = assets[relative]
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        os.link(stored, target)
        return identity

    for position, original in enumerate(selected, 0 if args.full else args.start):
        row = dict(original)
        row["instance_id"] = str(row["instance_id"])
        ident = row["instance_id"]
        if not ident.isdigit():
            raise ValueError("original numeric case ID required")
        case = args.work_dir / ident
        case.mkdir(mode=0o700)
        work = case / "testbed"
        work.mkdir(mode=0o755)
        work.chmod(0o755)
        name = row["gold_program_name"]
        if Path(name).name != name or "\\" in name or not name.endswith(".py"):
            raise ValueError("simple original Python filename required")
        program = args.program_root / ("pred_" + name)
        gold = child_path(args.asset_root, "benchmark/gold_programs/" + name)
        value = scalar(row["task_inst"], program, gold)
        test = modules["test_spec"].make_test_spec(
            row, "/testbed/benchmark/", "/testbed/pred_programs/"
        )
        if test.arch != args.arch:
            raise ValueError(
                "original case architecture differs from selected executor"
            )
        files = {}
        dataset = row["dataset_folder_tree"].split("\n")[0][4:].strip().rstrip("/")
        dataset_dir = child_path(args.asset_root, "benchmark/datasets/" + dataset)
        for source in sorted(dataset_dir.rglob("*")):
            if source.is_symlink():
                raise ValueError("scientific input symlinks not supported")
            if source.is_file():
                relative = source.relative_to(args.asset_root).as_posix()
                files[relative] = asset(relative, child_path(work, relative))
        if not files:
            raise ValueError("complete original scientific dataset directory required")
        for relative in [
            "benchmark/gold_programs/" + name,
            "benchmark/eval_programs/" + row["eval_script_name"],
        ]:
            files[relative] = asset(relative, child_path(work, relative))
        for path in (work / "benchmark/datasets").rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
        for folder in [
            "benchmark",
            "benchmark/datasets",
            "program_to_eval",
            "instance_path",
            "instance_path/input",
            "instance_path/output",
            "instance_path/pred_results",
        ]:
            path = work / folder
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o755)
        for folder in ["benchmark/gold_programs", "benchmark/eval_programs"]:
            (work / folder).chmod(0o700)
        os.chown(work / "instance_path/pred_results", 1000, 1000)
        for relative, content in [
            ("program_to_eval/pred_" + name, program.read_bytes()),
            ("compute_scores.py", (args.source_dir / "compute_scores.py").read_bytes()),
            ("scienceagentbench_isolated_scoring.py", wrapper.read_bytes()),
            ("npu-metric-receipt.json", args.npu_receipt.read_bytes()),
            ("instance_path/input/input.json", json.dumps(test.to_dict()).encode()),
        ]:
            path = work / relative
            path.write_bytes(content)
            path.chmod(0o644 if relative.endswith(".py") else 0o600)
            files[relative] = digest(path)
        recipe = case / "recipe"
        recipe.mkdir(mode=0o700)
        shutil.copyfile(program, recipe / ("pred_" + name))
        shutil.copyfile(
            args.source_dir / "config_conda_env.py", recipe / "config_conda_env.py"
        )
        text = original_instance_recipe(
            modules["dockerfiles"], test.platform, args.base_image, "pred_" + name
        )
        if args.xlsx_reader and any(
            p.suffix.lower() in {".xlsx", ".xlsm"} for p in dataset_dir.rglob("*")
        ):
            text += "\nRUN /opt/miniconda3/bin/python -m pip install --no-cache-dir openpyxl==3.1.5\n"
        (recipe / "Dockerfile").write_text(text, encoding="utf-8")
        manifest["cases"].append(
            {
                "instance_id": ident,
                "original_position": position,
                "expected_codebert_score": value,
                "files": files,
            }
        )
    (args.work_dir / "Dockerfile.base").write_text(
        modules["dockerfiles"].get_dockerfile_base(
            "linux/arm64/v8" if args.arch == "arm64" else "linux/x86_64", args.arch
        ),
        encoding="utf-8",
    )
    (args.work_dir / "manifest-private.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run(args):
    docker = ["docker", "--host", args.docker_host]
    info = json.loads(
        subprocess.check_output(docker + ["info", "--format", "{{json .}}"], text=True)
    )
    if info["ID"] != args.docker_id:
        raise ValueError("owned Docker identity mismatch; no task started")
    manifest = json.loads(
        (args.work_dir / "manifest-private.json").read_text(encoding="utf-8")
    )
    images = json.loads(args.image_map.read_text(encoding="utf-8"))
    results = {
        "source_id": "SCIENCEAGENTBENCH",
        "status": "SCIENTIFIC_EXECUTION_RUNNING",
        "source_checkbox_eligible": False,
        "cases": [],
    }
    receipt = args.work_dir / "scientific-receipt-public.json"
    if receipt.exists():
        raise FileExistsError(
            "reuse existing evidence; never blindly rerun completed tasks"
        )
    receipt.write_text(json.dumps(results, indent=2), encoding="utf-8")
    try:
        for item in manifest["cases"]:
            ident = item["instance_id"]
            case = args.work_dir / ident
            work = case / "testbed"
            for relative, identity in item["files"].items():
                if digest(child_path(work, relative)) != identity:
                    raise ValueError("prepared original case bytes changed")
            image = json.loads(
                subprocess.check_output(
                    docker + ["image", "inspect", images[ident]], text=True
                )
            )[0]["Id"]
            check = subprocess.run(
                docker
                + [
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--entrypoint",
                    "/opt/miniconda3/bin/python",
                    image,
                    "-m",
                    "pip",
                    "check",
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            (case / "pip-check-private.log").write_text(check.stdout + check.stderr)
            if check.returncode:
                raise RuntimeError("image dependencies invalid before task execution")
            name = "sab-original-" + uuid.uuid4().hex
            command = (
                docker
                + [
                    "run",
                    "--rm",
                    "--name",
                    name,
                    "--pull",
                    "never",
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--cap-add",
                    "SETUID",
                    "--cap-add",
                    "SETGID",
                    "--cap-add",
                    "DAC_OVERRIDE",
                    "--security-opt",
                    "no-new-privileges",
                    "--pids-limit",
                    "512",
                    "--memory",
                    "8g",
                    "--cpus",
                    "2",
                    "--tmpfs",
                    "/tmp:rw,nosuid,nodev,size=512m",
                    "--mount",
                    "type=bind,src="
                    + str(args.daemon_work_dir / ident / "testbed")
                    + ",dst=/testbed",
                    "--workdir",
                    "/testbed",
                    "--env",
                    "HOME=/tmp",
                    "--env",
                    "MPLCONFIGDIR=/tmp/grader-matplotlib",  # pragma: allowlist secret - private temporary cache path
                    "--env",
                    "OMP_NUM_THREADS=2",
                    "--env",
                    "OPENBLAS_NUM_THREADS=2",
                    "--entrypoint",
                    "/opt/miniconda3/bin/python",
                    image,
                    "/testbed/scienceagentbench_isolated_scoring.py",
                    "--original-source",
                    "/testbed/compute_scores.py",
                    "--npu-receipt",
                    "/testbed/npu-metric-receipt.json",
                ]
            )
            with (case / "native-private.log").open("xb") as log:
                try:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, timeout=1800
                    )
                except subprocess.TimeoutExpired:
                    subprocess.run(
                        docker + ["stop", "--time", "10", name],
                        capture_output=True,
                        timeout=30,
                    )
                    raise
            output = work / "instance_path/output/result.json"
            if completed.returncode or not output.exists():
                raise RuntimeError(
                    "original scientific execution infrastructure failed; retain log"
                )
            value = json.loads(output.read_text())
            if len(value) != 4 or value[1] != (
                1.0 if value[2] == 1 else item["expected_codebert_score"]
            ):
                raise ValueError(
                    "original result tuple differs from original score rule"
                )
            results["cases"].append(
                {
                    "instance_id": ident,
                    "image_id": image,
                    "valid_program": value[0],
                    "codebert_score": value[1],
                    "success": value[2],
                    "original_result_sha256": digest(output),
                    "original_log_info_retained_privately": True,
                }
            )
            receipt.write_text(json.dumps(results, indent=2), encoding="utf-8")
        results["status"] = (
            "ORIGINAL_SELECTED_SCIENTIFIC_PROTOCOL_TERMINAL_NOT_SOURCE_DELIVERY"
        )
    except BaseException as exc:
        results.update(
            status="FAILED_RETAINED_NO_TASK_REPLACEMENT", error_type=type(exc).__name__
        )
        raise
    finally:
        receipt.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["prepare", "run"], required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--program-root", type=Path)
    parser.add_argument("--npu-receipt", type=Path)
    parser.add_argument("--base-image")
    parser.add_argument("--arch", choices=["arm64", "x86_64"], default="arm64")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--xlsx-reader", action="store_true")
    parser.add_argument("--image-map", type=Path)
    parser.add_argument("--docker-host")
    parser.add_argument("--docker-id")
    parser.add_argument("--daemon-work-dir", type=Path)
    args = parser.parse_args()
    if os.name != "posix" or os.geteuid() != 0:
        parser.error(
            "use a dedicated Linux CPU executor that can prepare root-owned references"
        )
    required = (
        [
            "source_dir",
            "metadata",
            "asset_root",
            "program_root",
            "npu_receipt",
            "base_image",
        ]
        if args.action == "prepare"
        else ["image_map", "docker_host", "docker_id", "daemon_work_dir"]
    )
    if any(getattr(args, field) is None for field in required):
        parser.error("missing action-specific inputs: " + ", ".join(required))
    prepare(args) if args.action == "prepare" else run(args)
