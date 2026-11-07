"""Execute publisher CodeJudge-Eval inputs and declared paper-metric implementation.

The publisher has not released its regex extractor. This explicit strict-v1
extractor is independent code, not author-harness or paper-score reproduction.
No original input, option, code or few-shot example is reconstructed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from prepare_reused_research_assets import IO, selection
from run_docvqa_original import idle, local_endpoint


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def labels_for(name):
    for setting, labels in (
        ("easy", "ABC"),
        ("middle", "ABCDEF"),
        ("hard", "ABCDEFGHI"),
    ):
        if name.endswith("_" + setting + ".json"):
            return list(labels)
    raise ValueError("unknown publisher judging configuration")


def extract_choice(text, labels):
    """Independent strict-v1: bare option or explicit final answer; ambiguity fails."""
    text = text.strip()
    match = re.fullmatch(r"[\(\[]?([A-I])[\)\]]?[.!]?", text)
    if not match:
        match = re.search(
            r"(?i:the\s+answer\s+is|answer)\s*[:：]?\s*[\(\[]?([A-I])[\)\]]?[.!]?\s*$",
            text,
        )
    return match.group(1) if match and match.group(1) in labels else None


def metrics(gold, predictions, labels):
    if (
        not gold
        or len(gold) != len(predictions)
        or any(value not in labels for value in gold)
    ):
        raise ValueError("original gold labels and equal nonempty predictions required")
    f1 = []
    for label in labels:
        tp = sum(a == b == label for a, b in zip(gold, predictions))
        fp = sum(a != label and b == label for a, b in zip(gold, predictions))
        fn = sum(a == label and b != label for a, b in zip(gold, predictions))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {
        "accuracy": sum(a == b for a, b in zip(gold, predictions)) / len(gold),
        "macro_f1": sum(f1) / len(labels),
        "unparsed_answers": sum(value is None for value in predictions),
        "samples": len(gold),
        "labels": labels,
    }


def request_body(row, profile):
    body = {
        "model": profile["model"],
        "messages": [{"role": "user", "content": row["input"]}],
        "temperature": 0.0,
        "n": 1,
        "max_tokens": profile["max_output_tokens"],
    }
    if profile["disable_thinking"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def run(args):
    endpoint = local_endpoint(args.endpoint)
    source, files = selection("CODEJUDGE_EVAL", args.files)
    data = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if (
        output.is_relative_to(data)
        or data.is_relative_to(output)
        or output.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("fresh external output independent of original data required")
    if args.max_output_tokens <= 0:
        raise ValueError(
            "explicit positive fallback generation budget required; no author default established"
        )
    profile = {
        "dataset_revision": source["revision"],
        "model": args.model,
        "endpoint": endpoint,
        "max_output_tokens": args.max_output_tokens,
        "temperature": 0.0,
        "disable_thinking": args.disable_thinking,
        "full": args.full,
        "files": [file["path"] for file in files],
        "extractor": "independent-strict-v1-not-author-regex",
    }
    result = {
        "source_id": "CODEJUDGE_EVAL",
        "status": "PLAN_ONLY",
        "source_checkbox_eligible": False,
        "profile": profile,
        "profile_sha256": digest(profile),
        "model_request_attempts_this_invocation": 0,
        "configuration_metrics": {},
        "boundary": "Original full published inputs, separate references and independently implemented Accuracy/Macro-F1; not author regex/model/score reproduction. Sampling is not full1860-per-setting quality evaluation. No code execution or research observations.",
    }
    if not args.execute:
        return result
    if output.exists() and not args.resume:
        raise ValueError("fresh output or explicit compatible resume required")
    for file in files:
        IO.verify(IO.bound_path(data, "", file), file)
    selected = []
    for file in files:
        rows = json.loads((data / file["path"]).read_text("utf-8"))
        if len(rows) != file["rows"]:
            raise ValueError("original row count differs")
        for index, row in enumerate(rows if args.full else rows[:3]):
            selected.append((file["path"], index, row))
    completed = {}
    status_path = output / "native-public.json"
    private_path = output / "cases-private.jsonl"
    if output.exists():
        previous = json.loads(status_path.read_text("utf-8"))
        if previous["profile_sha256"] != result["profile_sha256"]:
            raise ValueError("resume profile changed; preserve original receipts")
        snapshot = output / f"previous-status-{time.time_ns()}.json"
        with snapshot.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(previous, indent=2) + "\n")
        if private_path.exists():
            for line in private_path.read_text("utf-8").splitlines():
                record = json.loads(line)
                key = (record["file"], record["row_index"])
                if key in completed:
                    raise ValueError("duplicate saved original case")
                completed[key] = record
    else:
        output.mkdir(mode=0o700, parents=True)
    expected = {(name, index): row for name, index, row in selected}
    for key, record in completed.items():
        if key not in expected:
            raise ValueError("saved case not in original selection")
        row = expected[key]
        if (
            record["original_prompt_sha256"] != digest(row["input"])
            or record["reference_answer"] != row["answer"]
            or record["request"] != request_body(row, profile)
            or any(record[field] != row[field] for field in ("data_id", "task_id"))
        ):
            raise ValueError("saved original input/reference/profile mismatch")
        if "response" in record:
            raw = record["response"]
            parsed = extract_choice(
                raw["choices"][0]["message"].get("content") or "", labels_for(key[0])
            )
            if raw.get("model") != args.model or record["prediction"] != parsed:
                raise ValueError("saved response/model/extraction mismatch")
    result["reused_cases"] = len(completed)

    def save():
        status_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    result["status"] = "RUNNING_ORIGINAL_PUBLISHED_TASKS"
    save()
    try:
        with urlopen(endpoint + "/models", timeout=10) as response:
            models = json.load(response)
        if args.model not in {row["id"] for row in models["data"]}:
            raise ValueError("declared fallback model not served")
        for name, index, row in selected:
            if (name, index) in completed:
                continue
            if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
                raise RuntimeError(
                    "research/device priority guard not idle; saved cases reusable"
                )
            body = request_body(row, profile)
            record = {
                "file": name,
                "row_index": index,
                "data_id": row["data_id"],
                "task_id": row["task_id"],
                "original_prompt_sha256": digest(row["input"]),
                "reference_answer": row["answer"],
                "request": body,
            }
            result["model_request_attempts_this_invocation"] += 1
            save()
            request = Request(
                endpoint + "/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urlopen(request, timeout=300) as response:
                    raw = json.load(response)
                if raw.get("model") != args.model:
                    raise ValueError("returned model identity differs")
                record["response"] = raw
                record["prediction"] = extract_choice(
                    raw["choices"][0]["message"].get("content") or "", labels_for(name)
                )
            except Exception as error:
                record["error_type"] = type(error).__name__
                record["error"] = str(error)
            with private_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed[(name, index)] = record
        for file in files:
            records = [
                completed[(name, index)]
                for name, index, row in selected
                if name == file["path"]
            ]
            if any("error_type" in record for record in records):
                raise RuntimeError(
                    "saved transport failures retained; no replacement questions or auto-retry"
                )
            result["configuration_metrics"][file["path"]] = metrics(
                [record["reference_answer"] for record in records],
                [record["prediction"] for record in records],
                labels_for(file["path"]),
            )
        result.update(
            status="ORIGINAL_PUBLISHED_TASKS_EXECUTED_AND_PAPER_METRICS_IMPLEMENTED_SAVED_AUDIT_PENDING",
            saved_cases=len(completed),
        )
    except Exception as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "output", "endpoint", "model"):
        parser.add_argument("--" + name, required=True)
    for name in ("api-server-pid", "npu-engine-pid", "max-output-tokens"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--files", nargs="+")
    for name in ("full", "execute", "resume", "disable-thinking"):
        parser.add_argument("--" + name, action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
