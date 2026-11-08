"""Run the original fixed-three scorer with disclosed ARM deployment bindings.

Trusted controller only: it requires Docker access. Generated task code runs in
separate network-disabled, capability-dropped containers without host mounts.
Neither original test logic nor scoring logic is patched.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import sys
import tarfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from recover_swesmith_fixed3_repository import COMMITS

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SHA256 = "2e5b7930d3546680bfc63849e10675c8d90ecd95da60201d4c5817cf841a0469"  # pragma: allowlist secret - fixed public author archive digest


def verify_author(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != AUTHOR_SHA256:
        raise ValueError("fixed original author archive required")
    package = Path(importlib.metadata.distribution("swesmith").locate_file("swesmith"))
    names = [
        "harness/eval.py",
        "harness/grading.py",
        "harness/utils.py",
        "profiles/base.py",
        "profiles/python.py",
    ]
    with tarfile.open(archive) as tar:
        for name in names:
            members = [m for m in tar if m.name.endswith("/swesmith/" + name)]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("unique original author module required")
            if tar.extractfile(members[0]).read() != (package / name).read_bytes():
                raise ValueError("installed author module differs: " + name)


def deployment_kwargs(kwargs, image, run_id):
    if kwargs.get("image") != image or not kwargs.get("name", "").startswith(
        "swesmith.eval." + run_id + "."
    ):
        raise ValueError("only this run's explicitly bound task image is allowed")
    if any(
        kwargs.get(key)
        for key in ["volumes", "mounts", "devices", "device_requests", "privileged"]
    ):
        raise ValueError("task must not receive host mounts, devices or privileges")
    if kwargs.get("network_mode") not in {None, "none"}:
        raise ValueError("task must not receive an externally connected network")
    return {
        **kwargs,
        "platform": "linux/arm64/v8",
        "network_disabled": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges"],
        "pids_limit": 512,
    }


def validate_predictions(path):
    if path.suffix == ".json":
        records = json.loads(path.read_text())
        if not isinstance(records, dict):
            raise ValueError("original JSON predictions must be keyed by instance ID")
        pairs = list(records.items())
    elif path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines()]
        pairs = [
            (record.get("instance_id") if isinstance(record, dict) else None, record)
            for record in records
        ]
    else:
        raise ValueError("original JSON/JSONL predictions required")
    seen = set()
    for identity, record in pairs:
        if (
            not isinstance(identity, str)
            or identity not in COMMITS
            or identity in seen
            or not isinstance(record, dict)
            or record.get("instance_id") != identity
            or "model_patch" not in record
            or (
                record["model_patch"] is not None
                and not isinstance(record["model_patch"], str)
            )
        ):
            raise ValueError(
                "predictions must preserve unique frozen task IDs and original string/null patches"
            )
        seen.add(identity)
    if not pairs:
        raise ValueError("nonempty predictions required")


def fixed_original_rows(shard):
    pins = json.loads(
        (
            ROOT / "configs/workload-native/swesmith-original-task-assets.json"
        ).read_text()
    )
    first = pins["dataset"]["files"][0]
    if (
        shard.stat().st_size != first["bytes"]
        or hashlib.sha256(shard.read_bytes()).hexdigest() != first["sha256"]
    ):
        raise ValueError("fixed original first shard identity differs")
    import pyarrow.parquet as pq

    rows = next(pq.ParquetFile(shard).iter_batches(batch_size=3)).to_pylist()
    if [r["instance_id"] for r in rows] != list(COMMITS):
        raise ValueError("original frozen task IDs differ")
    return rows


def run(args):
    if not re.fullmatch(r"workload-native-[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("task-owned unique run ID required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", args.image_namespace):
        raise ValueError("explicit local image namespace required")
    if args.image_namespace in {"swebench", "jyangballin"}:
        raise ValueError("compatibility image must not impersonate a publisher image")
    if args.output.exists():
        raise ValueError("fresh output required; inspect previous results first")
    verify_author(args.author_archive)
    rows = fixed_original_rows(args.original_shard)
    import docker
    from docker.models.containers import ContainerCollection
    from swesmith.profiles import registry

    profiles = [registry.get_from_inst(row) for row in rows]
    if len({id(p) for p in profiles}) != 1:
        raise ValueError(
            "this compatibility binding is for the single frozen oauthlib profile"
        )
    profile = profiles[0]
    original_command = profile.test_cmd
    expected = "source /opt/miniconda3/bin/activate; conda activate testbed; pytest --disable-warnings --color=no --tb=no --verbose"
    if original_command != expected:
        raise ValueError("original activation/test command changed; review binding")
    profile.org_dh = args.image_namespace
    image = profile.image_name
    client = docker.from_env()
    actual_image = client.images.get(image).id
    if actual_image != args.expected_image_id:
        raise ValueError("local task image ID differs from the qualified artifact")
    args.output.mkdir(mode=0o700, parents=True)
    original_rows = args.output / "original-tasks.private.json"
    original_rows.write_text(json.dumps(rows))
    original_rows.chmod(0o600)
    predictions = (
        args.predictions
        if args.predictions == "gold"
        else str(Path(args.predictions).resolve(strict=True))
    )
    if predictions != "gold":
        validate_predictions(Path(predictions))
    original_create = ContainerCollection.create
    created = []

    def create(collection, *positional, **kwargs):
        if positional:
            raise ValueError("original named container arguments required")
        bound = deployment_kwargs(kwargs, image, args.run_id)
        result = original_create(collection, **bound)
        created.append(
            {"name": kwargs["name"], "image": image, "image_id": actual_image}
        )
        return result

    original_dir, original_argv = Path.cwd(), sys.argv
    profile.test_cmd = original_command.split("; ")[-1]
    ContainerCollection.create = create
    try:
        os.chdir(args.output)
        sys.argv = [
            "swesmith.harness.eval",
            "--dataset_path",
            str(original_rows),
            "--predictions_path",
            predictions,
            "--run_id",
            args.run_id,
            "--workers",
            "1",
        ]
        runpy.run_module("swesmith.harness.eval", run_name="__main__")
    finally:
        ContainerCollection.create = original_create
        profile.test_cmd = original_command
        sys.argv = original_argv
        os.chdir(original_dir)
        (args.output / "deployment-bindings.json").write_text(
            json.dumps(
                {
                    "arm_image": image,
                    "image_id": actual_image,
                    "created_containers": created,
                    "original_test_command": original_command,
                    "bound_test_command": original_command.split("; ")[-1],
                    "difference": "Real system Python replaces Conda activation; same Python/package versions and unchanged pytest command. ARM and task-container security bindings only.",
                    "source_checkbox_eligible": False,
                },
                indent=2,
            )
        )
    verify_author(args.author_archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--image-namespace", default="workload-native")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Original predictions file or gold reference control",
    )
    parser.add_argument(
        "--author-archive", type=Path, default=Path("/opt/swesmith-author.tar.gz")
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    run(args)


if __name__ == "__main__":
    main()
