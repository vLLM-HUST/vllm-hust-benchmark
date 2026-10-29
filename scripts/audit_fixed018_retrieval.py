"""Reconstruct a fixed018/corrected-B1 retrieval pair from immutable raw capsules."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/long-context-fixed018-paired-20261029"


def reconstruct(raw):
    content, reasoning, finish, usage, done = "", "", None, None, False
    for line in gzip.decompress(raw).splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            done = True
            continue
        event = json.loads(payload)
        if event.get("error"):
            raise ValueError("Recorded stream error")
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            content += delta.get("content") or ""
            variants = [
                delta[k]
                for k in ("reasoning", "reasoning_content")
                if delta.get(k) is not None
            ]
            if variants:
                if not all(isinstance(v, str) and v == variants[0] for v in variants):
                    raise ValueError("Conflicting reasoning fields")
                reasoning += variants[0]
            finish = choice.get("finish_reason") or finish
    if not done or not finish or not usage:
        raise ValueError("Incomplete raw stream")
    return content, reasoning, finish, usage


def arm(files, directory, role):
    def raw(name):
        return files[directory + "/" + name]

    def read(name):
        return json.loads(raw(name))

    for line in raw("SHA256SUMS").decode().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256(raw(name)).hexdigest() != digest:
            raise ValueError("Raw member changed: " + directory + "/" + name)
    receipt = read("receipt.json")
    identity = read("runtime-identity.json")
    if identity["role"] != role:
        raise ValueError("Role mismatch")
    if role == "B0" and not (
        identity["vllm_version"] == identity["ascend_version"] == "0.18.0"
    ):
        raise ValueError("Not the fixed018 B0")
    result = dict(
        role=role,
        runtime=identity,
        status=receipt["status"],
        all_reserved_devices_released=receipt["all_reserved_devices_released"],
        physical_cards=receipt["reserved_physical_cards"],
        request_plan_sha256=None,
        cases=None,
        completed=None,
        failed=None,
        correct=None,
        total=None,
        continuous_stability_pass=None,
    )
    if receipt["status"] != "retrieval_complete":
        result["failure"] = receipt.get("error")
        return result
    inputs = read("request-plan.json")
    result["request_plan_sha256"] = hashlib.sha256(raw("request-plan.json")).hexdigest()
    qualification = read("parser-qualification.json")
    text, thought, finish, usage = reconstruct(raw("parser-qualification-sse.gz"))
    if not (
        qualification["passed"]
        and qualification["content"] == text
        and qualification["reasoning"] == thought
        and text.strip() == "4"
        and thought.strip()
        and finish == "stop"
        and qualification["usage"] == usage
    ):
        raise ValueError("Parser qualification mismatch")
    tokenize = read("tokenize-preflight.json")
    cases = []
    for index, case in enumerate(inputs["cases"]):
        pre = tokenize[index]
        if (pre["count"], pre["token_ids_sha256"]) != (
            case["input_tokens"],
            case["token_ids_sha256"],
        ):
            raise ValueError("Actual server input changed")
        recorded = read(f"long-case-{index:03d}.json")
        if not recorded["success"]:
            cases.append(dict(case_id=case["case_id"], success=False, correct=False))
            continue
        text, thought, finish, usage = reconstruct(raw(f"long-sse-{index:03d}.gz"))
        if (
            recorded["output"] != text
            or recorded["reasoning"] != thought
            or recorded["usage"] != usage
            or recorded["finish_reason"] != finish
            or usage["prompt_tokens"] != case["input_tokens"]
            or usage.get("prompt_tokens_details", {}).get("cached_tokens")
            not in (None, 0)
        ):
            raise ValueError("Stream/record/input/cache mismatch")
        correct = text.strip() == case["expected"]
        if recorded["correct"] != correct:
            raise ValueError("Original strict score mismatch")
        cases.append(
            dict(
                case_id=case["case_id"],
                input_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                finish_reason=finish,
                success=True,
                correct=correct,
                reasoning_separated=bool(thought.strip()),
                content_has_think_delimiter="<think>" in text or "</think>" in text,
            )
        )
    summary = read("long-context.json")
    counts = dict(
        completed=sum(r["success"] for r in cases),
        failed=sum(not r["success"] for r in cases),
        correct=sum(r["correct"] for r in cases),
        total=len(cases),
    )
    if any(summary[k] != value for k, value in counts.items()) or len(cases) != 6:
        raise ValueError("Original aggregate mismatch")
    result.update(cases=cases, **counts)
    return result


def audit(directory):
    capsule = directory / "raw.tar.gz"
    manifest = json.loads((directory / "capsule.json").read_text())
    payload = capsule.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("Capsule SHA mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            if (
                not member.isfile()
                or member.name in files
                or ".." in Path(member.name).parts
                or Path(member.name).is_absolute()
            ):
                raise ValueError("Unexpected capsule member")
            files[member.name] = archive.extractfile(member).read()
    prior = arm(files, "retrieval-b0-r1", "B0")
    if prior["status"] != "failed":
        raise ValueError("Original failed startup was changed")
    arms = [arm(files, "retrieval-b0-r2", "B0"), arm(files, "retrieval-b1-r1", "B1")]
    complete = all(a["status"] == "retrieval_complete" for a in arms)
    paired = (
        complete
        and arms[0]["request_plan_sha256"] == arms[1]["request_plan_sha256"]
        and arms[0]["physical_cards"] == arms[1]["physical_cards"]
    )
    activation = audit_activation(files) if complete else None
    return dict(
        schema="fixed018-retrieval-pair/v1",
        arms=arms,
        same_request_and_resource_pair=paired,
        corrected_b1_activation=activation,
        prior_failed_startup=prior,
        scope="Six synthetic16K/32K retrieval cases at10/50/90% positions, not full dataset quality or controlled continuous4K/16K stability",
        continuous_stability_pass=None,
        capsule_sha256=manifest["sha256"],
    )


def audit_activation(files):
    prefix = "retrieval-b1-r1/"

    def rows(name):
        return [
            json.loads(line)
            for line in files[prefix + name].splitlines()
            if line.strip()
        ]

    policy = rows("kvmat-policy.jsonl")
    engine = rows("kvmat-engine.jsonl")
    if len(policy) != 7 or len(engine) != 7:
        raise ValueError(
            "Expected six retrieval admissions plus one parser qualification"
        )
    salts = {}
    for row in policy:
        if (
            row.get("runtime_effective_decision") != "recompute"
            or row.get("runtime_control_path") != "request_scoped_prefix_cache_bypass"
            or row.get("reused_tokens") != 0
            or row.get("runtime_decision_supported") is not True
        ):
            raise ValueError("Unexpected KV action")
        digest = hashlib.sha256(row["runtime_cache_salt"].encode()).hexdigest()
        if digest in salts:
            raise ValueError("Duplicate request salt")
        salts[digest] = row["prompt_tokens"]
    admitted = {
        r["cache_salt_sha256"]: r["prompt_tokens"]
        for r in engine
        if r["event"] == "school028_request_salt_admitted"
    }
    if len(admitted) != 7 or admitted != salts:
        raise ValueError("Policy/engine linkage mismatch")
    ranks = {}
    for path in files:
        if not path.startswith(prefix + "microflow-") or not path.endswith(".jsonl"):
            continue
        events = [json.loads(line) for line in files[path].splitlines() if line.strip()]
        loaded = [r for r in events if r.get("event") == "model_loaded"]
        if len(loaded) != 1 or loaded[0].get("active") is not True:
            raise ValueError("Inactive Microflow rank")
        rank = loaded[0]["rank"]
        if rank not in (0, 1) or rank in ranks:
            raise ValueError("Duplicate or unexpected rank")
        equivalence = [r for r in events if r.get("event") == "partition_equivalence"]
        if {r["tokens"] for r in equivalence} != {1024, 2048, 4096} or any(
            r.get("exact") != [True] * 5 or r.get("rank") != rank for r in equivalence
        ):
            raise ValueError("Microflow partition qualification failed")
        split = sum(
            r.get("event") == "prefix" and r.get("split") is True for r in events
        )
        replay = sum(
            r.get("event") == "graph_replay"
            and r.get("active") is True
            and r.get("target") is True
            and r.get("speculative_enabled") is True
            for r in events
        )
        if not split or not replay:
            raise ValueError("Microflow not exercised")
        ranks[str(rank)] = dict(split_events=split, target_mtp_graph_replays=replay)
    if set(ranks) != {"0", "1"}:
        raise ValueError("Both ranks required")
    identity = json.loads(files[prefix + "runtime-identity.json"])
    if identity.get("worker") != "corrected_serving_runtime.CorrectedWorker":
        raise ValueError("Serving corrections missing")
    return dict(
        request_scoped_recompute_admissions=7,
        reused_tokens=0,
        microflow=ranks,
        corrected_worker=identity["worker"],
        scope="Actual activation and request admission, not general state-isolation proof",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args()
    result = audit(args.directory)
    path = args.directory / "summary.json"
    if args.write:
        path.write_text(json.dumps(result, indent=2) + "\n")
    if json.loads(path.read_text()) != result:
        raise ValueError("Derived summary is stale")
    print(
        "Raw retrieval streams and fixed018 identity audited; continuous stability remains separate"
    )


if __name__ == "__main__":
    main()
