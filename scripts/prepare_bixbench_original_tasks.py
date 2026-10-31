"""Prepare fixed original scientific tasks through publisher code; plan by default.

No download, model call, notebook execution or scoring. Original inputs remain
unchanged; publisher preprocessing may remove reference files only in new copies.
Run execution with the separately qualified Python3.12+ official client.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "workload-preparation/data/catalogs/bixbench_original_files_v1.json"


def verify_asset(root, entry):
    path = root / entry["path"]
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("original asset escapes data root")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("original publisher asset identity mismatch")
    return path


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            original_name = item.orig_filename
            member = PurePosixPath(original_name)
            if (
                member.is_absolute()
                or ".." in member.parts
                or "\\" in original_name
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("unsafe original ZIP member; do not extract")
        if archive.testzip() is not None:
            raise ValueError("original ZIP CRC mismatch")


def prepare(author, data, output, workspace, limit=3, execute=False):
    if not 1 <= limit <= 205:
        raise ValueError("original prefix limit must be between1 and205")
    author, data, output, workspace = [
        Path(path).resolve() for path in (author, data, output, workspace)
    ]
    report = {
        "source_id": "BIXBENCH",
        "status": "PLAN_ONLY",
        "selection": "first N unchanged original JSONL rows, no replacement",
        "limit": limit,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if not execute:
        return report
    if sys.platform != "linux" or sys.version_info < (3, 12):
        raise ValueError("qualified Linux Python3.12+ client required")
    destinations = (output, workspace)
    if (
        any(
            target.exists()
            or not target.parent.is_dir()
            or target.is_relative_to(ROOT)
            or target.is_relative_to(source)
            or source.is_relative_to(target)
            for target in destinations
            for source in (author, data)
        )
        or output.is_relative_to(workspace)
        or workspace.is_relative_to(output)
    ):
        raise ValueError("fresh disjoint external destinations required")
    # Helpers are repository software; publisher imports remain deferred.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_jsonschemabench_author_source import verify_source

    pins = ROOT / "configs/workload-native/bixbench-author-source.json"
    verify_source(author, pins)
    entries = {item["path"]: item for item in json.loads(CATALOG.read_text())["files"]}
    metadata = verify_asset(data, entries["BixBench.jsonl"])
    records = [json.loads(line) for line in metadata.read_text().splitlines()]
    if len(records) != 205 or len({row["question_id"] for row in records}) != 205:
        raise ValueError("original205 unique questions required")
    selected = records[:limit]
    capsules = sorted({row["data_folder"] for row in selected})
    for name in capsules:
        verify_zip(verify_asset(data, entries[name]))
    # Do not import or execute code from the capsules themselves.
    sys.path.insert(0, str(author))
    import yaml
    from bixbench.generate_trajectories import TrajectoryGenerator

    os.umask(0o077)
    output.mkdir(mode=0o700)
    workspace.mkdir(mode=0o700)
    config = yaml.safe_load(
        (author / "bixbench/run_configuration/generate_trajectories.yaml").read_text()
    )
    config["paths"].update(
        workspace_dir=str(workspace),
        trajectories_dir=str(output / "trajectories"),
        data_folder=str(output / "processed"),
    )
    config_path = output / "paths-only-config.private.yaml"
    config_path.write_text(yaml.safe_dump(config))
    generator = TrajectoryGenerator(config_path)
    for name in capsules:
        working_zip = generator.config.local_data_folder / name
        shutil.copy2(data / name, working_zip)
        generator._extract_and_process_files(
            working_zip, generator.config.local_data_folder / name.removesuffix(".zip")
        )
    tasks = []
    for original in selected:
        question = copy.deepcopy(original)
        question["local_data_folder"] = generator.config.local_data_folder / original[
            "data_folder"
        ].removesuffix(".zip")
        env = generator.environment_factory(question)
        if list(env.work_dir.rglob("*.ipynb")):
            raise ValueError("reference notebook remains; preserve failure, do not run")
        tasks.append(
            {
                "question_id": original["question_id"],
                "data_files": sum(p.is_file() for p in env.work_dir.rglob("*")),
            }
        )
    # Verify originals again after publisher code deleted only copied ZIPs.
    verify_asset(data, entries["BixBench.jsonl"])
    for name in capsules:
        verify_asset(data, entries[name])
    verify_source(author, pins)
    prepared_files = {}
    for path in sorted(generator.config.local_data_folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("unexpected prepared asset symlink")
        if path.is_file():
            prepared_files[
                path.relative_to(generator.config.local_data_folder).as_posix()
            ] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    report.update(
        status="AUTHOR_SELECTED_TASK_PREPARATION_PASSED_NOT_NATIVE_EXECUTION",
        tasks=tasks,
        capsules=len(capsules),
        original_max_steps=generator.config.rollout.max_steps,
        original_inputs_and_author_source_unchanged=True,
        prepared_files=prepared_files,
        container_workspace_permissions_still_require_explicit_operator_setup=True,
    )
    (output / "preparation-public.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("author-root", "data-root", "output", "workspace"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.author_root,
                args.data_root,
                args.output,
                args.workspace,
                args.limit,
                args.execute,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
