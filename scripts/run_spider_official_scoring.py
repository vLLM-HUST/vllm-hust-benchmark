"""Run pinned original Test Suite Accuracy in an owned CPU Docker sandbox.

Default is a read-only plan. No image pull, SQL rewriting, host SQL execution,
reference-value substitution, model calls or automatic source-completion claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

from prepare_spider_original_dev import digest

AUTHOR_REVISION = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"  # pragma: allowlist secret - public scorer revision
AUTHOR_FILES = {
    "evaluation.py": "7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779",  # pragma: allowlist secret
    "exec_eval.py": "29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7",  # pragma: allowlist secret
    "exec_subprocess.py": "1366694f8ad4d80cdd8fb45eb8c34f48f101fa88c3e0792bd3519f6bbe8530d9",  # pragma: allowlist secret
    "parse.py": "ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5",  # pragma: allowlist secret
    "process_sql.py": "927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1",  # pragma: allowlist secret
}


def regular(root, relative):
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
    ):
        raise ValueError("regular asset inside the explicit private root required")
    return path


def plan(assets, predictions, image):
    assets, predictions = Path(assets).resolve(), Path(predictions).resolve()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("immutable locally installed image ID required")
    receipt = json.loads(regular(assets, "inputs/preparation-public.json").read_bytes())
    indices = receipt.get("selected_original_dev_indices")
    if (
        receipt.get("source_id") != "SPIDER"
        or not isinstance(indices, list)
        or not indices
    ):
        raise ValueError("original Spider preparation receipt required")
    if (
        indices != list(range(len(indices)))
        or receipt.get("reference_sql_in_model_inputs") is not False
    ):
        raise ValueError("fixed original prefix with reference isolation required")
    author = []
    for name, expected in AUTHOR_FILES.items():
        path = regular(assets, "author/" + name)
        if digest(path) != expected:
            raise ValueError("pinned original scorer changed; no compatibility rewrite")
        author.append("author/" + name)
    refs = [
        json.loads(line)
        for line in regular(assets, "inputs/references.private.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    inputs = [
        json.loads(line)
        for line in regular(assets, "inputs/inputs.private.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    if len(refs) != len(indices) or len(inputs) != len(indices):
        raise ValueError("original task alignment differs")
    gold = []
    for index, ref, item in zip(indices, refs, inputs, strict=True):
        row = ref["original_record"]
        if (
            ref["original_dev_index"] != index
            or item["original_dev_index"] != index
            or set(item) != {"original_dev_index", "db_id", "question", "schema"}
            or item["db_id"] != row["db_id"]
            or item["question"] != row["question"]
            or item["schema"]["db_id"] != row["db_id"]
        ):
            raise ValueError("unaltered question/schema/reference binding required")
        gold.append(row["query"] + "\t" + row["db_id"] + "\n")
    gold_path = regular(assets, "inputs/gold.private.txt")
    if gold_path.read_bytes() != "".join(gold).encode("utf-8"):
        raise ValueError("original gold SQL/values changed")
    suites = receipt.get("suite_files", [])
    if not suites or len({s["archive_member"] for s in suites}) != len(suites):
        raise ValueError("complete original suite member inventory required")
    for item in suites:
        relative = item["archive_member"]
        if len(Path(relative).parts) != 3 or Path(relative).parts[0] != "database":
            raise ValueError("original suite namespace required")
        if digest(regular(assets, relative)) != item["sha256"]:
            raise ValueError("original suite member changed")
    selected_db_ids = {Path(item["archive_member"]).parts[1] for item in suites}
    if selected_db_ids != {ref["original_record"]["db_id"] for ref in refs}:
        raise ValueError("selected original task/database suite IDs differ")
    actual = {
        str(p.relative_to(assets)).replace(os.sep, "/")
        for db_id in selected_db_ids
        for p in (assets / "database" / db_id).glob("*.sqlite")
    }
    if actual != {item["archive_member"] for item in suites}:
        raise ValueError("missing or extra suite databases; never silently subset")
    raw = predictions.read_bytes()
    if len(raw) > 16 * 1024**2 or len(raw.decode("utf-8").splitlines()) != len(indices):
        raise ValueError(
            "one original prediction line per original task required; no SQL cleanup"
        )
    staged_files = (
        author
        + ["inputs/gold.private.txt", "inputs/preparation-public.json"]
        + [item["archive_member"] for item in suites]
    )
    return {
        "source_id": "SPIDER",
        "status": "PLAN_ONLY_NO_SQL_EXECUTION",
        "author_revision": AUTHOR_REVISION,
        "image": image,
        "original_dev_indices": indices,
        "prediction_sha256": hashlib.sha256(raw).hexdigest(),
        "suite_files": len(suites),
        "staged_files": staged_files,
        "staged_sha256": {name: digest(regular(assets, name)) for name in staged_files},
        "keep_distinct": True,
        "plug_value": False,
        "model_calls": 0,
        "sql_executed_on_host": False,
        "source_checkbox_eligible": False,
    }


def worker_command(image, name, volume):
    return [
        "docker",
        "run",
        "--name",
        name,
        "--label",
        "hust.spider.owner=" + name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--cpus=1",
        "--memory=2g",
        "--memory-swap=2g",
        "--pids-limit=64",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=64m",
        "--mount",
        "type=volume,source=" + volume + ",target=/assets,readonly",
        "--workdir=/assets/author",
        "--entrypoint=/opt/venv/bin/python",
        image,
        "/assets/author/evaluation.py",
        "--gold",
        "/assets/inputs/gold.private.txt",
        "--pred",
        "/assets/inputs/predictions.private.txt",
        "--db",
        "/assets/database",
        "--etype",
        "exec",
        "--keep_distinct",
    ]


def parse_metric(stdout, count):
    rows = [line.split() for line in stdout.splitlines()]
    totals = [r for r in rows if r and r[0] == "count"]
    metrics = [r for r in rows if r and r[0] == "execution"]
    if len(totals) != 1 or len(metrics) != 1 or int(totals[0][-1]) != count:
        raise ValueError("official scorer task count/output format differs")
    score = float(metrics[0][-1])
    if not 0 <= score <= 1:
        raise ValueError("official Test Suite Accuracy out of range")
    return score


def run(*, assets, predictions, image, output=None, execute=False, timeout=600):
    report = plan(assets, predictions, image)
    if not execute:
        return report
    if os.name != "posix" or os.geteuid() != 0:
        raise ValueError(
            "owned Linux Docker staging requires root; no host SQL fallback"
        )
    if type(timeout) is not int or not 1 <= timeout <= 7200 or output is None:
        raise ValueError("bounded controller timeout and fresh private output required")
    assets, predictions, output = (
        Path(p).resolve() for p in (assets, predictions, output)
    )
    if output.exists() or any(
        output.is_relative_to(p) or p.is_relative_to(output)
        for p in (assets, predictions)
    ):
        raise ValueError(
            "fresh output separate from original assets and predictions required"
        )
    output.mkdir(mode=0o700, parents=True)
    staging = output / "staging"
    staging.mkdir()
    for relative in report["staged_files"]:
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(regular(assets, relative), target)
        if digest(target) != report["staged_sha256"][relative]:
            raise ValueError("asset changed between plan and staging; preserve output")
    shutil.copyfile(predictions, staging / "inputs/predictions.private.txt")
    if (
        digest(staging / "inputs/predictions.private.txt")
        != report["prediction_sha256"]
    ):
        raise ValueError(
            "predictions changed between plan and staging; preserve output"
        )
    name = "hust-spider-" + uuid.uuid4().hex
    volume, stage = name + "-assets", name + "-stage"

    def docker(*args):
        return subprocess.check_output(["docker", *args], text=True, timeout=60).strip()

    docker("image", "inspect", image)
    docker("volume", "create", "--label", "hust.spider.owner=" + name, volume)
    docker(
        "create",
        "--name",
        stage,
        "--network=none",
        "--mount",
        "type=volume,source=" + volume + ",target=/assets",
        "--entrypoint=/bin/true",
        image,
    )
    docker("cp", str(staging) + "/.", stage + ":/assets")
    docker("rm", stage)
    # Trusted access-bit adjustment on this owned copy, never SQL/source edits.
    docker(
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--user=0:0",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--mount",
        "type=volume,source=" + volume + ",target=/assets",
        "--entrypoint=/bin/chmod",
        image,
        "-R",
        "a+rX",
        "/assets",
    )
    command = worker_command(image, name, volume)
    report.update(status="OFFICIAL_SCORER_RUNNING", container=name, volume=volume)
    receipt_path = output / "receipt-public.json"
    receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        owned = json.loads(docker("inspect", name))[0]
        if owned["Config"]["Labels"].get("hust.spider.owner") == name:
            docker("stop", "--time=5", name)
        report.update(status="RESOURCE_TIMEOUT_RETAINED_NOT_MODEL_SCORE")
        receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    (output / "stdout-private.log").write_bytes(result.stdout)
    (output / "stderr-private.log").write_bytes(result.stderr)
    inspection = json.loads(docker("inspect", name))[0]
    report.update(
        returncode=result.returncode,
        elapsed_seconds=time.monotonic() - started,
        container_state=inspection["State"],
        stdout_sha256=hashlib.sha256(result.stdout).hexdigest(),
    )
    report["status"] = "OFFICIAL_SCORER_FAILED_RETAINED_NOT_MODEL_SCORE"
    if result.returncode == 0:
        try:
            report["test_suite_accuracy"] = parse_metric(
                result.stdout.decode("utf-8"), len(report["original_dev_indices"])
            )
        except (ValueError, UnicodeDecodeError):
            report["status"] = (
                "OFFICIAL_OUTPUT_CONTRACT_FAILED_RETAINED_NOT_MODEL_SCORE"
            )
        else:
            report["status"] = (
                "ORIGINAL_TEST_SUITE_SCORING_COMPLETE_PROVIDER_CONSUMER_GATES_SEPARATE"
            )
    receipt_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    options = vars(parser.parse_args())
    result = run(**options)
    print(json.dumps(result))
    if (
        options["execute"]
        and result["status"]
        != "ORIGINAL_TEST_SUITE_SCORING_COMPLETE_PROVIDER_CONSUMER_GATES_SEPARATE"
    ):
        raise SystemExit(2)
