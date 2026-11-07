"""Verify pinned VisualWebArena prerequisites, without tasks, models or installs."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_crosscodeeval_original_assets import verify_git_blob
from prepare_xsum_original_assets import bound_path


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/visualwebarena-author-source.json"
)
TASK_FILES = (
    "config_files/vwa/test_classifieds.raw.json",
    "config_files/vwa/test_reddit.raw.json",
    "config_files/vwa/test_shopping.raw.json",
)


def patch_captioner_device(source):
    """Bind only the two original BLIP-2 initialization sites to one local NPU."""
    edits = (
        ('choices=["cpu", "cuda"],', 'choices=["cpu", "cuda", "npu:0"],'),
        (
            '    if args.observation_type in [\n        "accessibility_tree_with_captioner",',
            """    if args.eval_captioning_model_device != "npu:0":
        raise ValueError("Prepared NPU launcher requires --eval_captioning_model_device npu:0")
    if args.captioning_model != "Salesforce/blip2-flan-t5-xl":
        raise ValueError("Prepared NPU launcher supports the original BLIP-2 only")
    local_captioner = Path(os.environ["VWA_BLIP2_ROOT"]).resolve(strict=True)
    if "blip2" not in str(local_captioner) or not local_captioner.is_dir():
        raise ValueError("VWA_BLIP2_ROOT must name a local BLIP-2 directory")
    import torch_npu

    if torch.npu.device_count() != 1:
        raise RuntimeError("Expose exactly one assigned NPU to the captioner")

    if args.observation_type in [
        "accessibility_tree_with_captioner",""",
        ),
        (
            '        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"\n'
            "        dtype = torch.float16 if torch.cuda.is_available() else torch.float32\n",
            '        device = torch.device("npu:0")\n        dtype = torch.float32\n',
        ),
        (
            "            device, dtype, args.captioning_model\n",
            "            device, dtype, str(local_captioner)\n",
        ),
        (
            """                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,""",
            """                torch.device("npu:0"),
                torch.float32,
                str(local_captioner),""",
        ),
    )
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(
                "Pinned launcher captioner binding changed; review before preparing"
            )
        source = source.replace(old, new, 1)
    compile(source, "run.py", "exec")
    return source


def prepare_npu_launcher(author_root, output):
    report = inspect_original(author_root)
    root = Path(author_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the pinned author tree")
    raw = (root / "run.py").read_bytes()
    source = raw.decode("utf-8")
    patched = patch_captioner_device(source)
    diff = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile="author/run.py",
            tofile="prepared/run.py",
        )
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.py").write_bytes(patched.encode("utf-8"))
    (output / "captioner-device.patch").write_bytes(diff.encode("utf-8"))
    report.update(
        status="NPU_CAPTIONER_LAUNCHER_PREPARED_NOT_EXECUTED",
        original_launcher_sha256=hashlib.sha256(raw).hexdigest(),
        prepared_launcher_sha256=hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        patch_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        prepared_captioner_device="npu:0",
        prepared_captioner_precision="float32",
    )
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def inspect_original(author_root, *, pins_path=PINS):
    pins = json.loads(Path(pins_path).read_text("utf-8"))
    if pins.get("source_id") != "VISUALWEBARENA" or not pins.get("revision"):
        raise ValueError("pinned original VisualWebArena profile required")
    files = pins["files"]
    if len({row["path"] for row in files}) != len(files):
        raise ValueError("distinct original source paths required")
    root = Path(author_root).resolve(strict=True)
    for row in files:
        verify_git_blob(bound_path(root, "", row), row)
    required = {*TASK_FILES, "run.py", "environment_docker/README.md"}
    if not required.issubset({row["path"] for row in files}):
        raise ValueError("original tasks, launcher and site deployment pins required")
    launcher = (root / "run.py").read_text("utf-8")
    if (
        "if DATASET == 'visualwebarena':" not in launcher
        or "eval_caption_image_fn = image_utils.get_captioning_fn(" not in launcher
        or 'choices=["cpu", "cuda"]' not in launcher
    ):
        raise ValueError("review changed original eager captioner/device contract")
    groups = {}
    for name in TASK_FILES:
        tasks = json.loads((root / name).read_text("utf-8"))
        if not isinstance(tasks, list) or not tasks:
            raise ValueError("nonempty original task lists required")
        ids = []
        for task in tasks:
            if (
                not isinstance(task, dict)
                or type(task.get("task_id")) is not int
                or not isinstance(task.get("eval"), dict)
                or not isinstance(task.get("sites"), list)
            ):
                raise ValueError("original task/site/evaluator objects required")
            ids.append(task["task_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("distinct original task IDs within each split required")
        groups[name] = tasks
    fixed = groups[TASK_FILES[0]][:3]
    if [row["task_id"] for row in fixed] != [0, 1, 2]:
        raise ValueError("fixed original first three Classifieds tasks required")
    return {
        "source_id": "VISUALWEBARENA",
        "status": "ORIGINAL_PREREQUISITES_INSPECTED_NOT_NATIVE_READY",
        "publisher_revision": pins["revision"],
        "verified_original_files": len(files),
        "host_architecture": platform.machine(),
        "original_task_definitions": sum(len(rows) for rows in groups.values()),
        "original_split_counts": {name: len(rows) for name, rows in groups.items()},
        "fixed_original_tasks": [
            {
                "task_id": row["task_id"],
                "sites": row["sites"],
                "eval_types": row["eval"].get("eval_types", []),
                "has_original_goal_image": bool(row.get("image")),
                "require_reset": row.get("require_reset"),
            }
            for row in fixed
        ],
        "original_launcher_initializes_eval_captioner_even_for_url_match": True,
        "original_eval_captioner": "Salesforce/blip2-flan-t5-xl",
        "original_cli_eval_captioner_devices": ["cpu", "cuda"],
        "npu_captioner_execution_qualified": False,
        "url_only_metric_does_not_prove_captioner_free_original_launcher": True,
        "site_runtime_and_login_qualified": False,
        "actor_tokenizer_and_vision_protocol_qualified": False,
        "model_calls": 0,
        "original_task_or_scoring_executed": False,
        "source_checkbox_eligible": False,
        "remaining": [
            "Provision the original Classifieds site, complete author initialization and native login",
            "Qualify NPU-only captioner and actor compatibility without changing original tasks or metrics",
            "Preserve selected native Agent workflow and explicit budgets; obtain actual original scores",
            "Verify installed public capability and unchanged research-side original input/reference relations",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--prepare-npu-launcher", type=Path, metavar="NEW_OUTPUT")
    args = parser.parse_args()
    if args.prepare_npu_launcher and args.author_root is None:
        parser.error("--prepare-npu-launcher requires --author-root")
    if args.author_root is None:
        result = {
            "source_id": "VISUALWEBARENA",
            "status": "PLAN_ONLY",
            "source_profile": str(PINS),
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    elif args.prepare_npu_launcher:
        result = prepare_npu_launcher(args.author_root, args.prepare_npu_launcher)
    else:
        result = inspect_original(args.author_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
