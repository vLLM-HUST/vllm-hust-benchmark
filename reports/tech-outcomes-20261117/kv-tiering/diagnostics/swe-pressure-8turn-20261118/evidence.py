"""Allowlisted SWE numeric export and offline recomputation; never run a model."""

import argparse
import gzip
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("B0", "B1")
ROW_FIELDS = {
    "case_id",
    "task_id",
    "source_index",
    "phase",
    "input_sha256",
    "prompt_sha256",
    "prompt_tokens",
    "output_tokens",
    "status",
    "http_status",
    "finish_reason",
    "start",
    "first",
    "end",
    "cached_tokens",
    "stream_sha256",
    "output_sha256",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def close(a, b, label):
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), label)


def project(row, identity, phase):
    require(row["input_identity"] == identity["request_sha256"], "input hash")
    require(canonical(row["request"]) == identity["request_sha256"], "actual body hash")
    return {
        "case_id": row["case_id"],
        "task_id": row["task_id"],
        "source_index": row["source_index"],
        "phase": phase,
        "input_sha256": identity["request_sha256"],
        "prompt_sha256": identity["prompt_sha256"],
        "prompt_tokens": row["usage"]["prompt_tokens"],
        "output_tokens": row["usage"]["completion_tokens"],
        "status": row["status"],
        "http_status": row["http_status"],
        "finish_reason": row["finish_reason"],
        "start": row["start"]["monotonic"],
        "first": row["first_token"]["monotonic"],
        "end": row["end"]["monotonic"],
        "cached_tokens": row["cached_tokens_reported"],
        "stream_sha256": row["stream_sha256"],
        "output_sha256": canonical(
            {"content": row["content"], "reasoning": row.get("reasoning")}
        ),
    }


def export(source):
    require(
        not (ROOT / "manifest.json").exists(), "Refuse to overwrite a frozen export"
    )
    identities = read(source / "input-manifest.json")
    identity = {row["case_id"]: row for row in identities}
    save(ROOT / "inputs.json", identities)
    original_files = {}
    (ROOT / "numeric").mkdir(exist_ok=True)
    (ROOT / "provenance").mkdir(exist_ok=True)
    for name in (
        "plan.json",
        "cpu-fs-config.json",
        "audit.json",
        "comparison.json",
        "status.json",
    ):
        shutil.copyfile(source / name, ROOT / "provenance" / name)
        original_files[name] = sha(source / name)
    for arm in ARMS:
        rows, rounds = [], []
        for folder in sorted((source / arm).glob("swe-turn-*")):
            summary = read(folder / "summary.json")
            rounds.append({"phase": folder.name, "summary": summary})
            for path in sorted((folder / "requests").glob("*.json")):
                row = read(path)
                require(
                    sha(Path(row["stream_capture"])) == row["stream_sha256"], "SSE hash"
                )
                rows.append(project(row, identity[row["case_id"]], folder.name))
                original_files[str(path.relative_to(source))] = sha(path)
        raw = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        (ROOT / "numeric" / (arm + "-requests.jsonl.gz")).write_bytes(
            gzip.compress(raw, mtime=0)
        )
        save(ROOT / "numeric" / (arm + "-rounds.json"), rounds)
        shutil.copyfile(
            source / arm / "summary.json", ROOT / "numeric" / (arm + "-summary.json")
        )
        for suffix in (
            "command.json",
            "startup.json",
            "cleanup.json",
            "npu-before.txt",
            "npu-after.txt",
        ):
            name = arm + "-" + suffix
            shutil.copyfile(source / name, ROOT / "provenance" / name)
            original_files[name] = sha(source / name)
        admission = read(source / (arm + "-input-admission.json"))
        require(
            admission["passed"] and len(admission["records"]) == 1016,
            "tokenizer admission",
        )
        save(
            ROOT / "provenance" / (arm + "-tokenizer-receipt.json"),
            {
                "passed": True,
                "checked": 1016,
                "original_file_sha256": sha(source / (arm + "-input-admission.json")),
            },
        )
        for stage in ("before", "after"):
            name = arm + "-whole-" + stage + "-metrics.txt"
            shutil.copyfile(source / name, ROOT / "numeric" / name)
            original_files[name] = sha(source / name)
    # Publish error categories and counts, not arbitrary server messages/prompts.
    errors = Counter()
    for line in (source / "B1-server.log").read_text().splitlines():
        if "block I/O failed:" in line:
            reason = line.split("block I/O failed:", 1)[1].strip()
            require(
                reason
                == "FS segment quota exhausted; delete keys and collect fully dead segments",
                "Unreviewed server error; stop export",
            )
            errors["fs_segment_quota_exhausted"] += 1
    save(
        ROOT / "numeric" / "server-error-categories.json",
        {
            "source_sha256": sha(source / "B1-server.log"),
            "block_io_error_log_counts": dict(errors),
            "limits": "Log line counts are not request failures and are not interchangeable with metric counts",
        },
    )
    save(ROOT / "original-members.json", original_files)
    payload = [
        p for folder in ("numeric", "provenance") for p in (ROOT / folder).iterdir()
    ]
    payload += [ROOT / "inputs.json", ROOT / "original-members.json"]
    save(
        ROOT / "manifest.json",
        {
            "schema": "kv-tiering-swe-eight-round-numeric/v1",
            "run": "swe-pressure-8turn-02",
            "files": {
                str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p)}
                for p in sorted(payload)
            },
            "public_scope": "Numeric derivatives, identities and original source/command receipts; no prompt or answer text",
            "limits": "Checks establish internal consistency, not an independent device run, model quality or disk durability",
        },
    )


def load_rows(arm):
    return [
        json.loads(line)
        for line in gzip.decompress(
            (ROOT / "numeric" / (arm + "-requests.jsonl.gz")).read_bytes()
        ).splitlines()
    ]


def summarize(rows):
    tokens = sum(row["output_tokens"] for row in rows)
    span = max(row["end"] for row in rows) - min(row["start"] for row in rows)
    return {
        "tokens": tokens,
        "span": span,
        "throughput": tokens / span,
        "ttft": statistics.mean(row["first"] - row["start"] for row in rows),
        "tpot": statistics.mean(
            (row["end"] - row["first"]) / (row["output_tokens"] - 1) for row in rows
        ),
    }


def check_rows(rows, identities):
    require(len(rows) == len(identities) == 1016, "request count")
    require(len({row["case_id"] for row in rows}) == 1016, "duplicate request")
    for row in rows:
        require(set(row) == ROW_FIELDS, "numeric allowlist")
        identity = identities[row["case_id"]]
        require(row["status"] == "ok" and row["http_status"] == 200, "HTTP result")
        require(
            row["output_tokens"] == 32 and row["finish_reason"] == "length",
            "fixed output",
        )
        require(row["prompt_tokens"] == identity["prompt_tokens"], "prompt token count")
        require(row["input_sha256"] == identity["request_sha256"], "request identity")
        require(row["prompt_sha256"] == identity["prompt_sha256"], "token identity")
        require(row["start"] <= row["first"] <= row["end"], "timestamp order")
        require(
            all(math.isfinite(row[k]) for k in ("start", "first", "end")),
            "finite clocks",
        )


def metric(raw):
    result = {}
    for line in raw.splitlines():
        if line and not line.startswith("#"):
            key, value = line.rsplit(None, 1)
            result[key] = float(value)
    return result


def check():
    manifest = read(ROOT / "manifest.json")
    for name, item in manifest["files"].items():
        path = ROOT / name
        require(
            path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
            "hash: " + name,
        )
    identities = {row["case_id"]: row for row in read(ROOT / "inputs.json")}
    require(len({x["task_id"] for x in identities.values()}) == 128, "task count")
    comparison = read(ROOT / "provenance/comparison.json")
    recomputed = {}
    for arm in ARMS:
        rows = load_rows(arm)
        check_rows(rows, identities)
        calculated = summarize(rows)
        summary = read(ROOT / "numeric" / (arm + "-summary.json"))
        require(summary == comparison["arms"][arm], "comparison summary identity")
        close(calculated["tokens"], summary["accepted_output_tokens"], "output tokens")
        close(calculated["span"], summary["solver_batch_wall_seconds"], "batch span")
        close(
            calculated["throughput"],
            summary["output_tokens_per_second"],
            "batch throughput",
        )
        close(calculated["ttft"], summary["ttft_seconds"]["mean"], "TTFT")
        close(calculated["tpot"], summary["tpot_seconds"]["mean"], "TPOT")
        rounds = read(ROOT / "numeric" / (arm + "-rounds.json"))
        require(len(rounds) == 8, "round count")
        spans = []
        for phase in rounds:
            group = [row for row in rows if row["phase"] == phase["phase"]]
            stats, s = summarize(group), phase["summary"]
            require(len(group) == s["planned"] == s["successful"], "round completeness")
            close(stats["span"], s["solver_batch_wall_seconds"], "round span")
            close(
                stats["throughput"], s["output_tokens_per_second"], "round throughput"
            )
            close(stats["ttft"], s["ttft_seconds"]["mean"], "round TTFT")
            close(stats["tpot"], s["tpot_seconds"]["mean"], "round TPOT")
            spans.append(stats["span"])
        close(sum(spans), summary["summed_round_wall_seconds"], "phase sum")
        close(
            calculated["tokens"] / sum(spans),
            summary["phase_wall_output_tokens_per_second"],
            "phase throughput",
        )
        close(
            calculated["tokens"] / summary["full_workflow_wall_seconds"],
            summary["full_workflow_output_tokens_per_second"],
            "whole throughput",
        )
        before, after = [
            metric(
                (
                    ROOT / "numeric" / (arm + "-whole-" + stage + "-metrics.txt")
                ).read_text()
            )
            for stage in ("before", "after")
        ]
        for key, evidence in summary["counters"].items():
            require(evidence["before"] == before.get(key), "counter before")
            close(evidence["after"], after[key], "counter after")
            require(evidence["baseline_known"] == (key in before), "baseline identity")
            if key in before:
                close(evidence["delta"], after[key] - before[key], "counter delta")
            else:
                require(evidence["delta"] is None, "missing baseline must stay null")
        recomputed[arm] = dict(requests=len(rows), **calculated)
    for key, expected in comparison["changes_mod_over_native"].items():
        a, b = [comparison["arms"][arm][key] for arm in ARMS]
        if isinstance(a, dict):
            a, b = a["mean"], b["mean"]
        close(b / a - 1, expected, "paired change: " + key)
    a, b = [read(ROOT / "provenance" / (arm + "-command.json"))["argv"] for arm in ARMS]
    require(
        b[: len(a)] == a and b[len(a)] == "--kv-transfer-config",
        "common service arguments",
    )
    require(
        not any(
            flag in a
            for flag in ("--kv-cache-memory-bytes", "--num-gpu-blocks-override")
        ),
        "HBM override",
    )
    require(
        read(ROOT / "provenance/audit.json")["passed"], "original SSE audit receipt"
    )
    print(
        json.dumps(
            {
                "passed": True,
                "arms": recomputed,
                "changes": comparison["changes_mod_over_native"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-from",
        type=Path,
        help="Controlled original run; explicit one-time export only",
    )
    parser.add_argument(
        "--check", action="store_true", help="CPU-only numeric/hash audit, no network"
    )
    args = parser.parse_args()
    if args.export_from:
        export(args.export_from)
    if args.check:
        check()
    require(
        args.export_from is not None or args.check, "Choose --check or --export-from"
    )
