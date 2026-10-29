"""Prepare original SWE repository images with explicit supported architecture.

The pinned release CLI defaults to x86_64 and has no architecture flag. This
adapter passes original TestSpec objects with the requested architecture to
unmodified original builders. It never changes tasks, scripts or dependencies.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from prepare_jsonschemabench_author_source import verify_source
from run_swebench_original_agent import select_records

ROOT = Path(__file__).resolve().parents[1]


class BuildAPI:
    def __init__(self, original, allowed_images, output, network):
        self.original = original
        self.allowed_images = set(allowed_images)
        self.output = Path(output).resolve()
        self.network = network
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def build(self, **kwargs):
        if kwargs.get("tag") not in self.allowed_images:
            raise ValueError("original build requested an unselected image")
        if not Path(kwargs["path"]).resolve().is_relative_to(self.output):
            raise ValueError(
                "original build context must remain in private output directory"
            )
        if self.network == "host":
            kwargs["network_mode"] = "host"
        self.calls += 1
        return self.original.build(**kwargs)


class BuildClient:
    def __init__(self, original, specs, output, network):
        self.original = original
        images = {
            key
            for spec in specs
            for key in (
                spec.base_image_key,
                spec.env_image_key,
                spec.instance_image_key,
            )
        }
        self.api = BuildAPI(original.api, images, output, network)

    def __getattr__(self, name):
        return getattr(self.original, name)


def execute(args, rows):
    import docker

    args.harness_root = args.harness_root.resolve(strict=True)
    args.output = args.output.resolve()
    verify_source(
        args.harness_root,
        ROOT / "configs/workload-native/swebench-v410-author-source.json",
    )
    sys.path.insert(0, str(args.harness_root))
    builders = importlib.import_module("swebench.harness.docker_build")
    specs_module = importlib.import_module("swebench.harness.test_spec.test_spec")
    if not Path(builders.__file__).resolve().is_relative_to(args.harness_root):
        raise ValueError("loaded builders differ from original pinned source")
    if not args.docker_url.startswith("unix:///") or args.max_workers < 1:
        raise ValueError(
            "explicit local Docker socket and positive worker count required"
        )
    if args.output.is_relative_to(ROOT):
        raise ValueError("original build files must remain outside checkout")
    specs = [specs_module.make_test_spec(row, arch=args.arch) for row in rows]
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    original = docker.DockerClient(base_url=args.docker_url)
    client = BuildClient(original, specs, args.output, args.build_network)
    report = {
        "source_id": args.source_id,
        "status": "PREPARING_ORIGINAL_IMAGES",
        "architecture": args.arch,
        "build_network": args.build_network,
        "trusted_software_build_only_not_generated_code_network_policy": True,
        "original_source_or_scripts_modified": False,
        "records_sha256": hashlib.sha256(args.records_json.read_bytes()).hexdigest(),
        "instances": [],
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    receipt = args.output / "images-public.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

    previous = Path.cwd()
    save()
    try:
        os.chdir(args.output)
        _, failed = builders.build_env_images(
            client, specs, force_rebuild=False, max_workers=args.max_workers
        )
        if failed:
            raise RuntimeError(
                "original environment image build failed; inspect retained logs"
            )
        for spec in specs:
            builders.build_instance_image(spec, client, logger=None, nocache=False)
            image = original.images.get(spec.instance_image_key)
            report["instances"].append(
                {
                    "instance_id": spec.instance_id,
                    "image_key": spec.instance_image_key,
                    "image_id": image.id,
                    "original_eval_script_sha256": hashlib.sha256(
                        spec.eval_script.encode()
                    ).hexdigest(),
                }
            )
            save()
        report.update(
            status="ORIGINAL_SELECTED_IMAGES_READY_NOT_TASK_EXECUTION",
            actual_build_api_calls=client.api.calls,
        )
    except Exception as error:
        report.update(
            status="ORIGINAL_IMAGE_PREPARATION_FAILED_RETAINED",
            error_type=type(error).__name__,
        )
        raise
    finally:
        os.chdir(previous)
        save()
        original.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-id", choices=["SWE_BENCH", "SWE_VERIFIED"], required=True
    )
    parser.add_argument(
        "--records-file",
        "--records-json",
        dest="records_json",
        type=Path,
        required=True,
    )
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-url", required=True)
    parser.add_argument("--arch", choices=["arm64", "x86_64"], default="arm64")
    parser.add_argument("--build-network", choices=["bridge", "host"], default="bridge")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    rows = select_records(args.records_json, full=args.all)
    if args.execute:
        result = execute(args, rows)
    else:
        result = {
            "status": "PLAN_ONLY",
            "architecture": args.arch,
            "build_network": args.build_network,
            "original_instance_ids": [r["instance_id"] for r in rows],
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
