"""Run pinned unmodified SWE-bench functional tests and original report logic.

The SDK boundary adds container isolation only. Original patch application,
test scripts, FAIL_TO_PASS/PASS_TO_PASS and scoring are not replaced. Gold
controls are separately labelled and never claimed as model performance.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

from prepare_jsonschemabench_author_source import verify_source
from run_swebench_original_agent import select_records

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges"],
    "network_mode": "none",
    "mem_limit": 8 * 1024**3,
    "memswap_limit": 8 * 1024**3,
    "nano_cpus": 2_000_000_000,
    "pids_limit": 512,
    "shm_size": 256 * 1024**2,
}


class RestrictedContainers:
    def __init__(self, original, allowed_images, created):
        self.original = original
        self.allowed_images = set(allowed_images)
        self.created = created

    def __getattr__(self, name):
        return getattr(self.original, name)

    def create(self, **kwargs):
        allowed = {"image", "name", "user", "detach", "command", "platform", "cap_add"}
        if set(kwargs) - allowed or kwargs.get("image") not in self.allowed_images:
            raise ValueError("unexpected original grading container binding")
        if kwargs.get("cap_add"):
            raise ValueError("grading cannot request additional Linux capabilities")
        kwargs.update(LIMITS)
        container = self.original.create(**kwargs)
        container.reload()
        config = container.attrs["HostConfig"]
        if (
            config["Privileged"]
            or config["CapDrop"] != ["ALL"]
            or config["NetworkMode"] != "none"
            or config["Memory"] != LIMITS["mem_limit"]
            or config["NanoCpus"] != LIMITS["nano_cpus"]
            or config["PidsLimit"] != LIMITS["pids_limit"]
            or not any(s.startswith("no-new-privileges") for s in config["SecurityOpt"])
            or container.attrs["Mounts"]
            or config.get("Devices")
            or config.get("DeviceRequests")
        ):
            container.remove(force=True)
            raise RuntimeError("new grading container failed isolation verification")
        self.created.append(
            {
                "id": container.id,
                "image": kwargs["image"],
                "network": "none",
                "host_mounts": 0,
                "memory": config["Memory"],
                "cpu_quota_nanos": config["NanoCpus"],
                "pids_limit": config["PidsLimit"],
            }
        )
        return container


class RestrictedClient:
    def __init__(self, client, allowed_images, created):
        self.original = client
        self.containers = RestrictedContainers(
            client.containers, allowed_images, created
        )

    def __getattr__(self, name):
        return getattr(self.original, name)


def predictions_for(rows, predictions=None, *, gold=False):
    selected = {}
    for row in rows:
        identity = row["instance_id"]
        if gold:
            value = {
                "instance_id": identity,
                "model_name_or_path": "gold-original-reference-control",
                "model_patch": row["patch"],
            }
        else:
            if not isinstance(predictions, dict) or identity not in predictions:
                raise ValueError(
                    "missing original Agent prediction; do not fabricate an empty answer"
                )
            value = predictions[identity]
        if not isinstance(value, dict) or value.get("instance_id") != identity:
            raise ValueError("prediction identity differs from original task")
        if not isinstance(value.get("model_name_or_path"), str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_./-]*", value["model_name_or_path"]
        ):
            raise ValueError("original model identity required")
        if "model_patch" not in value or not isinstance(
            value["model_patch"], (str, type(None))
        ):
            raise ValueError("original patch or explicit native empty outcome required")
        selected[identity] = dict(value)
    return selected


def execute(args, rows, predictions):
    import docker

    args.harness_root = args.harness_root.resolve(strict=True)
    args.output = args.output.resolve()
    verify_source(
        args.harness_root,
        ROOT / "configs/workload-native/swebench-v410-author-source.json",
    )
    sys.path.insert(0, str(args.harness_root))
    evaluation = importlib.import_module("swebench.harness.run_evaluation")
    if (
        not Path(evaluation.__file__)
        .resolve()
        .is_relative_to(args.harness_root.resolve())
    ):
        raise ValueError("loaded scoring code differs from pinned original source")
    if not args.docker_url.startswith("unix:///"):
        raise ValueError("explicit private local Docker socket required")
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("private grading files must remain outside checkout")
    original_client = docker.DockerClient(base_url=args.docker_url)
    try:
        specs = [evaluation.make_test_spec(row, arch=args.arch) for row in rows]
        for spec in specs:
            original_client.images.get(spec.instance_image_key)
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except BaseException:
        original_client.close()
        raise
    report = {
        "source_id": args.source_id,
        "status": "RUNNING_ORIGINAL_FUNCTIONAL_GRADING",
        "role": "ORIGINAL_GOLD_ENVIRONMENT_CONTROL_NOT_MODEL_RESULT"
        if args.gold_control
        else "ORIGINAL_AGENT_PREDICTION_GRADING",
        "records_sha256": hashlib.sha256(args.records_json.read_bytes()).hexdigest(),
        "original_instance_ids": [r["instance_id"] for r in rows],
        "original_test_timeout_seconds": 1800,
        "original_scoring_code_modified": False,
        "original_tests_or_tolerances_modified": False,
        "original_summary_image_inventory_not_requested": True,
        "created_isolated_containers": [],
        "instance_results": [],
        "source_checkbox_eligible": False,
    }
    client = RestrictedClient(
        original_client,
        [s.instance_image_key for s in specs],
        report["created_isolated_containers"],
    )
    receipt = args.output / "grade-public.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")

    previous_cwd = Path.cwd()
    save()
    try:
        os.chdir(args.output)
        for spec in specs:
            prediction = predictions[spec.instance_id]
            if prediction["model_patch"] in ("", None):
                # Same exclusion as original get_dataset_from_preds/reporting;
                # an explicit empty model submission is not an invented patch.
                result = {
                    "instance_id": spec.instance_id,
                    "status": "OFFICIAL_EMPTY_PATCH_OUTCOME",
                }
            else:
                actual = evaluation.run_instance(
                    spec,
                    prediction,
                    rm_image=False,
                    force_rebuild=False,
                    client=client,
                    run_id=args.run_id,
                    timeout=1800,
                    rewrite_reports=False,
                )
                result = {"instance_id": spec.instance_id, "official_result": actual}
            report["instance_results"].append(result)
            save()
        # client=None avoids rebuilding default-x86 image specs merely for image
        # inventory. The unchanged original quality report still reads all the
        # original per-instance logs/outcomes and includes explicit empty cases.
        summary = evaluation.make_run_report(
            predictions, rows, args.run_id, client=None
        )
        report["original_summary_file"] = str(summary)
        report["original_summary"] = json.loads(summary.read_text())
        report["status"] = "ORIGINAL_FUNCTIONAL_GRADING_FINISHED_AUDIT_REQUIRED"
    except Exception as error:
        report.update(
            status="ORIGINAL_GRADING_FAILURE_RETAINED", error_type=type(error).__name__
        )
        raise
    finally:
        os.chdir(previous_cwd)
        save()
        original_client.close()
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
        help="Original Parquet or JSON record array; legacy JSON flag retained",
    )
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-url", required=True)
    parser.add_argument("--arch", choices=["arm64", "x86_64"], default="arm64")
    parser.add_argument("--run-id", default="original-native")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--gold-control", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("safe original grading run ID required")
    rows = select_records(args.records_json, full=args.all)
    predictions = predictions_for(
        rows,
        json.loads(args.predictions.read_text(encoding="utf-8"))
        if args.predictions
        else None,
        gold=args.gold_control,
    )
    if args.execute:
        result = execute(args, rows, predictions)
    else:
        result = {
            "status": "PLAN_ONLY",
            "original_instance_ids": list(predictions),
            "gold_control": args.gold_control,
            "original_test_timeout_seconds": 1800,
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
