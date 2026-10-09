"""Pinned MMLU-Pro task-quality adapter for the PyramidKV dataset program.

Prompt formatting and answer extraction are adapted from TIGER-AI-Lab/MMLU-Pro,
Apache-2.0, evaluate_from_local.py at f418b116db00b065c2aea046518d8fcf74d39872.
Invalid extractions count as incorrect; the reference runner's random guessing
fallback is deliberately disabled. This is a named subset contract, not its
full leaderboard configuration.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path

DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"  # pragma: allowlist secret -- public upstream revision/checksum
REFERENCE_REVISION = "f418b116db00b065c2aea046518d8fcf74d39872"  # pragma: allowlist secret -- public upstream revision/checksum
DATA_HASHES = {
    "test": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",  # pragma: allowlist secret -- public upstream revision/checksum
    "validation": "139423c23722e480c807ac4a191409a710cfce4eba744c1d641cf88e730e2078",  # pragma: allowlist secret -- public upstream revision/checksum
}
REFERENCE_HASH = "d0a4ae51b80efa8f99713a21253ad5542913ef2ac9915e95d40c8a62174e9401"  # pragma: allowlist secret -- public upstream revision/checksum
INITIAL_PROMPT_HASH = "d7337806b4d84eef7a86bcb20bf3d64b65e72736c1603e16d08cadebac436f0a"  # pragma: allowlist secret -- public upstream revision/checksum
THRESHOLD = 4096
MAX_OUTPUT = 2048


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def checked(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return data


def extract_answer(text: str) -> str | None:
    for pattern in (r"answer is \(?([A-J])\)?", r".*[aA]nswer:\s*([A-J])"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def format_example(row: dict, *, answered: bool) -> str:
    options = [option for option in row["options"] if option != "N/A"]
    index = row["answer_index"]
    if not 0 <= index < len(options) <= 10 or options[index] != row["options"][index]:
        raise ValueError("Option filtering would change the frozen answer index")
    if row["answer"] != chr(65 + index):
        raise ValueError("Answer letter/index mismatch")
    text = "Question:\n" + row["question"] + "\nOptions:\n"
    text += "".join(f"{chr(65 + i)}. {option}\n" for i, option in enumerate(options))
    if answered:
        text += (
            row["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            + "\n\n"
        )
    else:
        text += "Answer: Let's think step by step."
    return text


def encode_prompt(tokenizer, text: str) -> list[int]:
    # Recent Transformers may return a BatchEncoding for tokenize=True. Count
    # actual input IDs, never the number of keys in that container.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    if (
        not isinstance(ids, list)
        or len(ids) < 16
        or not all(type(x) is int for x in ids)
    ):
        raise ValueError("Expected actual prompt token IDs")
    return ids


def choose_tasks(inventory: list[dict], per_category: int) -> set[int]:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    selected = set()
    for category in sorted({x["category"] for x in inventory}):
        members = sorted(
            (x for x in inventory if x["category"] == category),
            key=lambda x: x["question_id"],
        )
        selected.update(x["question_id"] for x in members[:per_category])
        eligible = [x for x in members if x["prompt_tokens"] > THRESHOLD]
        if eligible:
            selected.add(eligible[0]["question_id"])
    return selected


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args) -> None:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tables = {}
    for split, digest in DATA_HASHES.items():
        path = args.assets / f"{split}-00000-of-00001.parquet"
        checked(path, digest)
        tables[split] = pq.read_table(path).to_pylist()
    if len(tables["test"]) != 12032 or len(tables["validation"]) != 70:
        raise ValueError("Unexpected pinned dataset cardinality")
    checked(args.reference / "evaluate_from_local.py", REFERENCE_HASH)
    instruction = checked(
        args.reference / "cot_prompt_lib/initial_prompt.txt", INITIAL_PROMPT_HASH
    ).decode()
    runtime = json.loads(args.runtime_contract.read_text())
    for key in (
        "model_revision",
        "container_image_digest",
        "sources",
        "packages",
        "command",
        "hardware",
        "method_config_sha256",
        "environment",
    ):
        if not runtime.get(key):
            raise ValueError(f"Missing runtime provenance: {key}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    subjects = sorted({row["category"] for row in tables["test"]})
    prefixes = {}
    for subject in subjects:
        examples = [row for row in tables["validation"] if row["category"] == subject]
        if len(examples) != 5:
            raise ValueError(
                "The pinned contract requires all five validation examples"
            )
        prefixes[subject] = (
            instruction.replace("{$}", subject)
            + "\n"
            + "".join(format_example(row, answered=True) for row in examples)
        )
    inventory = []
    by_id = {}
    for row in tables["test"]:
        question_id = row["question_id"]
        if question_id in by_id:
            raise ValueError("Duplicate immutable task ID")
        by_id[question_id] = row
        prompt = encode_prompt(
            tokenizer, prefixes[row["category"]] + format_example(row, answered=False)
        )
        if len(prompt) + MAX_OUTPUT > 32768:
            raise ValueError(
                "Prompt exceeds the frozen context; do not silently truncate"
            )
        inventory.append(
            {
                "question_id": question_id,
                "category": row["category"],
                "source_row_sha256": sha(canonical(row)),
                "prompt_tokens": len(prompt),
                "prompt_sha256": sha(
                    json.dumps(prompt, separators=(",", ":")).encode()
                ),
            }
        )
    selected = choose_tasks(inventory, args.per_category)
    cases = []
    for metadata in sorted(inventory, key=lambda x: (x["category"], x["question_id"])):
        if metadata["question_id"] not in selected:
            continue
        row = by_id[metadata["question_id"]]
        prompt = encode_prompt(
            tokenizer, prefixes[row["category"]] + format_example(row, answered=False)
        )
        cases.append(
            dict(
                metadata,
                case_id=f"mmlu-pro-{row['question_id']:05d}",
                kind="mmlu-pro",
                prompt=prompt,
                answer=row["answer"],
                option_count=len([x for x in row["options"] if x != "N/A"]),
                max_tokens=MAX_OUTPUT,
                ignore_eos=False,
            )
        )
    save(args.output / "cases.json", cases)
    save(
        args.output / "tasks.json",
        [{k: v for k, v in row.items() if k != "prompt"} for row in cases],
    )
    with gzip.GzipFile(
        filename=str(args.output / "inventory.jsonl.gz"), mode="wb", mtime=0
    ) as f:
        f.write(b"".join(canonical(row) + b"\n" for row in inventory))
    contract = {
        "contract_version": "pyramidkv-mmlu-pro-subset-v1",
        "designation": "pujiang-specified-dataset-scope",
        "dataset": "mmlu-pro",
        "dataset_revision": DATASET_REVISION,
        "data_sha256": DATA_HASHES,
        "dataset_license": "MIT per pinned dataset card; raw questions are not republished",
        "reference_revision": REFERENCE_REVISION,
        "reference_source_sha256": REFERENCE_HASH,
        "prompt_template_sha256": INITIAL_PROMPT_HASH,
        "scorer": "reference three-stage A-J extraction; invalid/out-of-range/HTTP failures count incorrect; no random guessing",
        "scorer_source_sha256": sha(Path(__file__).read_bytes()),
        "transport_source_sha256": sha(
            (args.pyramidkv_repo / "scripts/qwen35/evaluate.py").read_bytes()
        ),
        "analyzer_source_sha256": sha(
            (args.pyramidkv_repo / "scripts/qwen35/analyze_evaluation.py").read_bytes()
        ),
        "full_test_count": len(inventory),
        "validation_count": 70,
        "selection": f"First {args.per_category} test IDs per category, plus the first compression-eligible ID per category if not already selected; fixed before outputs",
        "tasks": len(cases),
        "eligible_tasks": sum(x["prompt_tokens"] > THRESHOLD for x in cases),
        "full_eligible_tasks": sum(x["prompt_tokens"] > THRESHOLD for x in inventory),
        "generation": {
            "temperature": 0,
            "seed": 17,
            "thinking": False,
            "max_tokens": MAX_OUTPUT,
            "stop": None,
            "retries": 0,
            "concurrency": 1,
            "tools": [],
            "truncation": "reject",
        },
        "prompt_contract": "Reference five-shot CoT wrapped in Qwen chat template; thinking disabled; no reference 4096-context shot reduction or Question: stop",
        "runtime": runtime,
        "model_files_sha256": {
            name: sha((args.model_path / name).read_bytes())
            for name in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ]
            if (args.model_path / name).is_file()
        },
        "files_sha256": {
            name: sha((args.output / name).read_bytes())
            for name in ["cases.json", "tasks.json", "inventory.jsonl.gz"]
        },
        "publication_status": "candidate contract and bounded subset evidence; review required, no full-dataset score or optimization claim",
    }
    save(args.output / "contract.json", contract)
    print(
        json.dumps(
            {k: contract[k] for k in ["tasks", "eligible_tasks", "full_eligible_tasks"]}
        ),
        flush=True,
    )


def load_plan(path: Path) -> tuple[dict, list[dict]]:
    contract = json.loads((path / "contract.json").read_text())
    if contract["scorer_source_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("Scorer/runner changed after contract freeze")
    for name, digest in contract["files_sha256"].items():
        checked(path / name, digest)
    cases = json.loads((path / "cases.json").read_text())
    if len(cases) != contract["tasks"] or len({x["case_id"] for x in cases}) != len(
        cases
    ):
        raise ValueError("Task manifest mismatch")
    for row in cases:
        if (
            len(row["prompt"]) != row["prompt_tokens"]
            or sha(json.dumps(row["prompt"], separators=(",", ":")).encode())
            != row["prompt_sha256"]
        ):
            raise ValueError("Prompt IDs differ from frozen manifest")
    return contract, cases


def process_identity(pid: int) -> dict:
    process = Path(f"/proc/{pid}")
    env = dict(
        x.split("=", 1)
        for x in (process / "environ").read_text().split("\0")
        if "=" in x
    )
    keys = [
        "VLLM_ASCEND_KVCOMPRESS_ENABLED",
        "ASCEND_RT_VISIBLE_DEVICES",
        "VLLM_PLUGINS",
        "VLLM_VERSION",
        "VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2",
        "VLLMHUST_EXT_ENABLED_BUNDLES",
    ]
    return {
        "pid": pid,
        "command": (process / "cmdline").read_text().split("\0")[:-1],
        "environment": {k: env.get(k) for k in keys},
        "method_config_sha256": sha(
            Path(env["VLLM_ASCEND_KVCOMPRESS_CONFIG"]).read_bytes()
        ),
    }


def validate_activation(identity: dict, contract: dict, arm: str) -> None:
    if identity["environment"]["VLLM_ASCEND_KVCOMPRESS_ENABLED"] != (
        "0" if arm == "B0" else "1"
    ):
        raise ValueError("Arm label disagrees with actual server activation")
    if identity["environment"]["VLLMHUST_EXT_ENABLED_BUNDLES"]:
        raise ValueError(
            "Direct-activation contract requires empty Manager bundle list"
        )
    for key, value in contract["runtime"]["environment"].items():
        if (
            key != "VLLM_ASCEND_KVCOMPRESS_ENABLED"
            and identity["environment"].get(key) != value
        ):
            raise ValueError(f"Actual environment differs from frozen runtime: {key}")
    for key in ["command", "method_config_sha256"]:
        if identity[key] != contract["runtime"][key]:
            raise ValueError(f"Actual server differs from frozen runtime: {key}")


def score_results(results: list[dict], cases: list[dict]) -> dict:
    if not cases or len({x["case_id"] for x in cases}) != len(cases):
        raise ValueError("Expected nonempty unique task manifest")
    lookup = {x["case_id"]: x for x in cases}
    if len(results) != len(cases) or {x["case_id"] for x in results} != set(lookup):
        raise ValueError("Missing or duplicate task outcomes")
    scored = []
    for result in results:
        case = lookup[result["case_id"]]
        for key in ("prompt_sha256", "prompt_tokens", "max_tokens", "ignore_eos"):
            if result[key] != case[key]:
                raise ValueError(f"Unmatched request contract: {key}")
        prediction = (
            extract_answer(result.get("output", "")) if result["success"] else None
        )
        valid = (
            prediction is not None
            and prediction in "ABCDEFGHIJ"[: case["option_count"]]
        )
        scored.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "eligible": case["prompt_tokens"] > THRESHOLD,
                "prediction": prediction,
                "valid": valid,
                "correct": valid and prediction == case["answer"],
                "transport_success": result["success"],
                "length_limited": result.get("finish_reason") == "length",
            }
        )

    def summary(rows):
        return {
            "count": len(rows),
            "correct": sum(x["correct"] for x in rows),
            "accuracy": sum(x["correct"] for x in rows) / len(rows) if rows else None,
        }

    return {
        **summary(scored),
        "invalid_answers": sum(not x["valid"] for x in scored),
        "transport_failures": sum(not x["transport_success"] for x in scored),
        "length_limited": sum(x["length_limited"] for x in scored),
        "eligible": summary([x for x in scored if x["eligible"]]),
        "ineligible": summary([x for x in scored if not x["eligible"]]),
        "categories": {
            c: summary([x for x in scored if x["category"] == c])
            for c in sorted({x["category"] for x in scored})
        },
        "outcomes": scored,
    }


def run(args) -> None:
    contract, cases = load_plan(args.plan)
    transport_path = args.pyramidkv_repo / "scripts/qwen35/evaluate.py"
    checked(transport_path, contract["transport_source_sha256"])
    client = load_module(transport_path, "pyramidkv_pinned_transport")
    identity = process_identity(args.server_pid)
    validate_activation(identity, contract, args.arm)
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.output / "requests"
    raw.mkdir()
    save(args.output / "launch.json", identity)
    save(args.output / "contract.json", contract)
    (args.output / "metrics-before.txt").write_text(
        client.fetch(args.base_url, "/metrics")
    )
    stop = threading.Event()

    def sample():
        last_npu = 0.0
        with (args.output / "telemetry.jsonl").open("w") as output:
            while not stop.is_set():
                row = {"time_unix": time.time(), "phase": "mmlu-pro"}
                try:
                    row["metrics"] = client.metric_values(
                        client.fetch(args.base_url, "/metrics")
                    )
                    if time.monotonic() - last_npu >= 2:
                        row["npu_smi"] = subprocess.check_output(
                            ["npu-smi", "info"], text=True, timeout=10
                        )
                        last_npu = time.monotonic()
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    row["error"] = str(exc)
                output.write(json.dumps(row) + "\n")
                output.flush()
                stop.wait(0.5)

    sampler = threading.Thread(target=sample, daemon=True)
    results = []
    sampler.start()
    try:
        for case in cases:
            results.append(client.stream_request(args.base_url, case, raw))
            save(args.output / "results.json", results)
            print(
                f"{args.arm}: {len(results)}/{len(cases)} {case['case_id']} success={results[-1]['success']}",
                flush=True,
            )
    finally:
        stop.set()
        sampler.join(timeout=20)
        save(args.output / "results.json", results)
        (args.output / "metrics-after.txt").write_text(
            client.fetch(args.base_url, "/metrics")
        )
        shutil.copyfile(args.server_log, args.output / "server.log")
    if process_identity(args.server_pid) != identity:
        raise ValueError("Server identity changed during evaluation")
    scores = score_results(results, cases)
    save(args.output / "score.json", scores)
    if scores["transport_failures"]:
        raise RuntimeError(
            "Failed requests retained and scored incorrect; inspect raw evidence"
        )


def analyze(args) -> None:
    contract, cases = load_plan(args.plan)
    path = args.pyramidkv_repo / "scripts/qwen35/analyze_evaluation.py"
    checked(path, contract["analyzer_source_sha256"])
    shared = load_module(path, "pyramidkv_pinned_analysis")
    arms = {}
    identities = []
    for arm in ["B0", "B1"]:
        root = args.root / arm
        if json.loads((root / "contract.json").read_text()) != contract:
            raise ValueError("Mismatched B0/B1 frozen contract")
        identity = json.loads((root / "launch.json").read_text())
        validate_activation(identity, contract, arm)
        identities.append(identity)
        rows = json.loads((root / "results.json").read_text())
        compression = shared.transactions(
            (root / "server.log").read_text(), rows, arm == "B1"
        )
        successful = [x for x in rows if x["success"]]
        arms[arm] = {
            "quality": score_results(rows, cases),
            "compression": compression,
            "resources": shared.resources(root / "telemetry.jsonl"),
            "supporting_telemetry": {
                key: statistics.mean(x[key] for x in successful) if successful else None
                for key in ["ttft_seconds", "tpot_seconds", "e2e_seconds"]
            },
        }
    for key in identities[0]["environment"]:
        if (
            key != "VLLM_ASCEND_KVCOMPRESS_ENABLED"
            and identities[0]["environment"][key] != identities[1]["environment"][key]
        ):
            raise ValueError("Non-treatment launch environment differs")
    verified = all(x["compression"]["verified"] for x in arms.values())
    result = {
        "contract_version": contract["contract_version"],
        "scope": contract["publication_status"],
        "contract_sha256": sha((args.plan / "contract.json").read_bytes()),
        "applicability": "applicable"
        if contract["eligible_tasks"]
        else "not-exercised",
        "control_path_verified": verified,
        "accuracy_change_percentage_points": 100
        * (arms["B1"]["quality"]["accuracy"] - arms["B0"]["quality"]["accuracy"]),
        "arms": arms,
        "artifact_sha256": {
            str(p.relative_to(args.root)): sha(p.read_bytes())
            for arm in ["B0", "B1"]
            for p in sorted((args.root / arm).rglob("*"))
            if p.is_file()
        },
    }
    save(args.output, result)
    if not verified:
        raise RuntimeError(
            "Control-path verification failed; do not claim optimization"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in [
        "assets",
        "model-path",
        "reference",
        "runtime-contract",
        "pyramidkv-repo",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--per-category", type=int, default=5)
    r = sub.add_parser("run")
    for name in ["plan", "pyramidkv-repo", "output", "server-log"]:
        r.add_argument("--" + name, type=Path, required=True)
    r.add_argument("--server-pid", type=int, required=True)
    r.add_argument("--arm", choices=["B0", "B1"], required=True)
    r.add_argument("--base-url", default="http://127.0.0.1:8000")
    a = sub.add_parser("analyze")
    for name in ["plan", "pyramidkv-repo", "root", "output"]:
        a.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
