"""Verify pinned VisualWebArena prerequisites, without tasks, models or installs."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from urllib.parse import urlsplit
from zipfile import is_zipfile

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
LOCAL_ACTOR_FILES = ("agent/agent.py", "llms/tokenizers.py")


def prepare_fixed_classifieds(author_root, output, origin):
    """Materialize fixed tasks without altering the author tree or protocol."""
    report = inspect_original(author_root)
    root = Path(author_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the pinned author tree")
    parsed = urlsplit(origin)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not re.fullmatch(
            r"[A-Za-z0-9.:-]+|\[[A-Fa-f0-9:]+\](?::[0-9]+)?", parsed.netloc
        )
        or origin != f"{parsed.scheme}://{parsed.netloc}"
        or parsed.port == 0
    ):
        raise ValueError(
            "Classifieds requires a bare HTTP(S) origin without credentials"
        )
    tasks = json.loads((root / TASK_FILES[0]).read_text("utf-8"))[:3]
    output_files = {}
    for task_id, task in enumerate(tasks):
        if (
            task["task_id"] != task_id
            or task["sites"] != ["classifieds"]
            or task.get("require_login") is not True
            or task.get("require_reset") is not False
            or task.get("storage_state") != "./.auth/classifieds_state.json"
            or task.get("start_url") != "__CLASSIFIEDS__"
            or task["eval"].get("eval_types") != ["url_match"]
            or task["eval"].get("url_note") != "EXACT"
            or not re.fullmatch(
                r"__CLASSIFIEDS__/index\.php\?page=item&id=\d+",
                task["eval"].get("reference_url", ""),
            )
        ):
            raise ValueError("Fixed original Classifieds protocol changed")
        serialized = json.dumps(task, indent=2).replace("__CLASSIFIEDS__", origin)
        if re.search(r"__[A-Z_]+__", serialized):
            raise ValueError("Unresolved author site placeholder in fixed task")
        output_files[f"{task_id}.json"] = serialized.encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in output_files.items():
        (output / name).write_bytes(raw)
    report.update(
        status="FIXED_CLASSIFIEDS_CONFIGS_PREPARED_NOT_EXECUTED",
        prepared_configs={
            name: hashlib.sha256(raw).hexdigest() for name, raw in output_files.items()
        },
        requires_original_site_login=True,
    )
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def patch_local_actor(sources):
    original = (
        '("gemini" in lm_config.model or "gpt-4" in lm_config.model and '
        '"vision" in lm_config.model)'
    )
    actor = sources["agent/agent.py"]
    if actor.count(original) != 1:
        raise ValueError("Pinned original multimodal actor check changed")
    actor = actor.replace(
        original,
        "(\n"
        "            (lm_config.provider == 'openai' and lm_config.mode == 'chat'\n"
        "             and lm_config.model == 'Qwen/Qwen3.5-9B')\n"
        f"            or {original}\n"
        "        )",
        1,
    )
    tokenizer = sources["llms/tokenizers.py"]
    edits = (
        (
            "from typing import Any\n",
            "from typing import Any\nimport json\nimport os\nfrom pathlib import Path\n",
        ),
        (
            '        if provider == "openai":\n',
            """        self._local_qwen35 = provider == "openai" and model_name == "Qwen/Qwen3.5-9B"
        if self._local_qwen35:
            from transformers import AutoTokenizer

            root = Path(os.environ["VWA_QWEN35_TOKENIZER_ROOT"]).resolve(strict=True)
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen3_5":
                raise ValueError("Expected local Qwen3.5 tokenizer metadata")
            self.tokenizer = AutoTokenizer.from_pretrained(
                root, local_files_only=True, trust_remote_code=False
            )
        elif provider == "openai":
""",
        ),
        (
            "        return self.tokenizer.encode(text)\n\n    def decode",
            "        if self._local_qwen35:\n"
            "            return self.tokenizer.encode(text, add_special_tokens=False)\n"
            "        return self.tokenizer.encode(text)\n\n    def decode",
        ),
        (
            "    def __call__(self, text: str) -> list[int]:\n"
            "        return self.tokenizer.encode(text)",
            "    def __call__(self, text: str) -> list[int]:\n"
            "        return self.encode(text)",
        ),
    )
    for old, new in edits:
        if tokenizer.count(old) != 1:
            raise ValueError("Pinned original tokenizer binding changed")
        tokenizer = tokenizer.replace(old, new, 1)
    patched = dict(zip(LOCAL_ACTOR_FILES, (actor, tokenizer)))
    for name, source in patched.items():
        compile(source, name, "exec")
    return patched


def prepare_qwen35_actor(author_root, output):
    report = inspect_original(author_root)
    root = Path(author_root).resolve(strict=True)
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the pinned author tree")
    pins = json.loads(PINS.read_text("utf-8"))
    if not set(LOCAL_ACTOR_FILES).issubset(row["path"] for row in pins["files"]):
        raise ValueError("Original actor and tokenizer must be verified by source pins")
    sources = {
        name: (root / name).read_bytes().decode("utf-8") for name in LOCAL_ACTOR_FILES
    }
    patched = patch_local_actor(sources)
    output.mkdir(parents=True, exist_ok=False)
    for name, source in patched.items():
        path = output / name
        path.parent.mkdir()
        path.write_bytes(source.encode("utf-8"))
    diff = "".join(
        line
        for name in LOCAL_ACTOR_FILES
        for line in difflib.unified_diff(
            sources[name].splitlines(keepends=True),
            patched[name].splitlines(keepends=True),
            fromfile=f"author/{name}",
            tofile=f"prepared/{name}",
        )
    )
    (output / "local-qwen35-actor.patch").write_bytes(diff.encode("utf-8"))
    report.update(
        status="LOCAL_QWEN35_SOM_ACTOR_PREPARED_NOT_EXECUTED",
        prepared_actor_model="Qwen/Qwen3.5-9B",
        prepared_actor_files={
            name: hashlib.sha256(source.encode("utf-8")).hexdigest()
            for name, source in patched.items()
        },
    )
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


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


def audit_classifieds_results(author_log, result_dir):
    """Check fixed-task result coverage, not model provenance or site setup."""
    log_format = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - "
    result_line = re.compile(log_format + r"\[Result\] \((PASS|FAIL)\) (.+)")
    rows = []
    for line in author_log.read_text("utf-8").splitlines():
        match = result_line.fullmatch(line)
        if match:
            score, config_file = match.groups()
            rows.append((Path(config_file).name, score == "PASS"))
        elif re.match(log_format + r"\[(?:OpenAI|Unhandled) Error\]", line):
            raise ValueError("original launcher logged a per-task error")
    if [name for name, _ in rows] != ["0.json", "1.json", "2.json"]:
        raise ValueError("expected exactly one ordered original result for IDs 0, 1, 2")
    error_file = result_dir / "error.txt"
    if error_file.exists() and error_file.stat().st_size:
        raise ValueError("original launcher wrote per-task errors")
    for task_id in range(3):
        trace = result_dir / "traces" / f"{task_id}.zip"
        if trace.is_symlink() or not trace.is_file() or not is_zipfile(trace):
            raise ValueError("missing original browser trace archive")
    return {
        "source_id": "VISUALWEBARENA",
        "status": "FIXED_CLASSIFIEDS_RESULT_COVERAGE_AUDITED",
        "tasks": [
            {"task_id": task_id, "original_binary_score": int(passed)}
            for task_id, (_, passed) in enumerate(rows)
        ],
        "manual_model_and_site_audit_required": True,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--prepare-npu-launcher", type=Path, metavar="NEW_OUTPUT")
    output.add_argument("--prepare-qwen35-actor", type=Path, metavar="NEW_OUTPUT")
    output.add_argument("--prepare-fixed-classifieds", type=Path, metavar="NEW_OUTPUT")
    output.add_argument(
        "--audit-classifieds-results",
        nargs=2,
        type=Path,
        metavar=("AUTHOR_LOG", "RESULT_DIR"),
    )
    parser.add_argument("--classifieds-origin")
    args = parser.parse_args()
    if (
        args.prepare_npu_launcher
        or args.prepare_qwen35_actor
        or args.prepare_fixed_classifieds
    ) and args.author_root is None:
        parser.error("Preparation requires --author-root")
    if bool(args.prepare_fixed_classifieds) != bool(args.classifieds_origin):
        parser.error(
            "--prepare-fixed-classifieds and --classifieds-origin are required together"
        )
    if args.audit_classifieds_results:
        if args.author_root is not None:
            parser.error("result audit does not take an author source directory")
        result = audit_classifieds_results(*args.audit_classifieds_results)
    elif args.author_root is None:
        result = {
            "source_id": "VISUALWEBARENA",
            "status": "PLAN_ONLY",
            "source_profile": str(PINS),
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    elif args.prepare_npu_launcher:
        result = prepare_npu_launcher(args.author_root, args.prepare_npu_launcher)
    elif args.prepare_qwen35_actor:
        result = prepare_qwen35_actor(args.author_root, args.prepare_qwen35_actor)
    elif args.prepare_fixed_classifieds:
        result = prepare_fixed_classifieds(
            args.author_root, args.prepare_fixed_classifieds, args.classifieds_origin
        )
    else:
        result = inspect_original(args.author_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
