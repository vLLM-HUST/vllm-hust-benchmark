"""Read immutable experiment evidence without extracting or executing its files."""

import hashlib
import io
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(content: bytes):
    return json.loads(content, object_pairs_hook=unique_object)


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
        and path.as_posix() == name
    )


def load_capsule(root: Path) -> dict[str, bytes]:
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/\\]+)", line)
        require(match is not None, "Checksum entry is malformed or unsafe")
        sha, name = match.groups()
        require(name not in (".", "..", "SHA256SUMS"), "Checksum filename is unsafe")
        require(name not in expected, f"Checksum duplicate entry: {name}")
        expected[name] = sha
    actual = {
        p.name
        for p in root.iterdir()
        if (p.is_file() or p.is_symlink()) and p.name != "SHA256SUMS"
    }
    require(set(expected) == actual, "File inventory differs from SHA256SUMS")
    files = {}
    for name, sha in expected.items():
        path = root / name
        require(path.is_file() and not path.is_symlink(), f"Nonregular file: {name}")
        files[name] = path.read_bytes()
        require(digest(files[name]) == sha, f"File digest mismatch: {name}")
    require("members.json" in files, "Member inventory is missing")
    inventory = read_json(files["members.json"])
    require(isinstance(inventory, dict), "Member inventory must be an object")
    parts = sorted(name for name in files if name.startswith("raw.tar.gz.part-"))
    if parts:
        require(
            "raw.tar.gz" not in files, "Archive parts mixed with a complete archive"
        )
        require(
            parts == [f"raw.tar.gz.part-{i:03d}" for i in range(len(parts))],
            "Archive part sequence is incomplete",
        )
        packed = b"".join(files[name] for name in parts)
    else:
        require("raw.tar.gz" in files, "Archive parts or complete archive are missing")
        packed = files["raw.tar.gz"]
    raw = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
            for member in archive:
                name = member.name
                require(member.isfile(), f"Nonregular member: {name}")
                require(safe_member(name), f"Unsafe member: {name}")
                require(name not in raw, f"Duplicate member: {name}")
                raw[name] = archive.extractfile(member).read()
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"Unreadable archive: {exc}") from exc
    require(set(raw) == set(inventory), "Member inventory differs from archive")
    for name, content in raw.items():
        record = inventory[name]
        require(
            isinstance(record, dict), f"Member inventory entry is malformed: {name}"
        )
        require(len(content) == record.get("bytes"), f"Member length mismatch: {name}")
        require(
            digest(content) == record.get("sha256"), f"Member digest mismatch: {name}"
        )
    return raw


def verify_case_order(cases: list[dict], rows: list[dict]) -> None:
    wanted = [case["case_id"] for case in cases]
    found = [row["case_id"] for row in rows]
    require(
        len(wanted) == len(set(wanted)) and found == wanted,
        "Case order or membership differs",
    )


def verify_response(case: dict, row: dict, payload: dict | None, stream: bytes) -> dict:
    for key, value in case.items():
        if key not in ("prompt", "answers"):
            require(row.get(key) == value, f"Result case metadata differs: {key}")
    require(
        row.get("success") is True and row.get("http_status") == 200,
        "Result request failed",
    )
    require(
        isinstance(row.get("request_id"), str) and row["request_id"],
        "Result request ID missing",
    )
    if payload is None:
        require(case["kind"] == "quality", "Payload missing for synthetic request")
    else:
        fixed = {
            "model": "qwen35-pyramidkv",
            "temperature": 0,
            "seed": 17,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        fixed.update({key: case[key] for key in ("max_tokens", "ignore_eos")})
        for key, value in fixed.items():
            require(payload.get(key) == value, f"Payload generation differs: {key}")
        require(
            payload.get("cache_salt") == case.get("cache_salt"),
            "Payload cache salt differs",
        )
        expected_keys = (
            set(fixed)
            | {"prompt"}
            | ({"cache_salt"} if "cache_salt" in case else set())
        )
        require(
            set(payload) == expected_keys, "Payload has unexpected generation fields"
        )
        prompt = payload["prompt"]
        require(
            isinstance(prompt, list) and all(type(x) is int and x >= 0 for x in prompt),
            "Payload prompt must contain token IDs",
        )
        require(len(prompt) == case["prompt_tokens"], "Payload prompt length differs")
        require(
            digest(json.dumps(prompt, separators=(",", ":")).encode())
            == case["prompt_sha256"],
            "Payload prompt digest differs",
        )
    lines = [line for line in stream.decode().splitlines() if line]
    require(
        lines and lines[-1] == "data: [DONE]" and lines.count("data: [DONE]") == 1,
        "SSE must end with exactly one DONE",
    )
    require(
        all(line.startswith("data: ") for line in lines),
        "SSE contains an unexpected line",
    )
    chunks = [read_json(line[6:].encode()) for line in lines[:-1]]
    require(
        chunks and all(chunk.get("id") == row["request_id"] for chunk in chunks),
        "SSE request ID differs",
    )
    choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
    require(
        choices and all(choice.get("index") == 0 for choice in choices),
        "SSE choice index differs",
    )
    require(
        "".join(choice.get("text", "") for choice in choices) == row["output"],
        "SSE output differs",
    )
    finishes = [
        choice["finish_reason"] for choice in choices if choice.get("finish_reason")
    ]
    require(
        finishes == [row.get("finish_reason")] and finishes[0] in ("length", "stop"),
        "SSE finish differs",
    )
    usages = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
    require(usages == [row.get("usage")], "SSE usage differs")
    usage = usages[0]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(
            type(usage.get(key)) is int and usage[key] >= 0, f"Usage invalid: {key}"
        )
    require(
        usage["prompt_tokens"] == case["prompt_tokens"], "Usage prompt length differs"
    )
    require(
        0 < usage["completion_tokens"] <= case["max_tokens"],
        "Usage output length differs",
    )
    require(
        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
        "Usage total differs",
    )
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    require(
        type(cached) is int and 0 <= cached <= usage["prompt_tokens"],
        "Usage cached count missing or invalid",
    )
    if case.get("require_uncached"):
        require(cached == 0, "Usage unexpected cached prompt tokens")
    if case["ignore_eos"]:
        require(
            usage["completion_tokens"] == case["max_tokens"],
            "Usage forced output incomplete",
        )
        require(finishes == ["length"], "SSE forced output finish differs")
    for key in ("ttft_seconds", "e2e_seconds", "tpot_seconds"):
        value = row.get(key)
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Result invalid timing: {key}",
        )
    require(
        row["e2e_seconds"] > 0 and row["ttft_seconds"] <= row["e2e_seconds"],
        "Result invalid timing order",
    )
    require(
        row["ttft_seconds"]
        + row["tpot_seconds"] * max(usage["completion_tokens"] - 1, 1)
        <= row["e2e_seconds"] + 1e-8,
        "Result TPOT exceeds request timing",
    )
    return {"payload_verified": payload is not None}


def recover_worker_pairs(log: str) -> tuple[str, list[dict]]:
    """Conserve explicit distinct prefixes and two byte-identical ACK bodies."""
    pattern = re.compile(
        r"^(?P<first>\(Worker_TP(?P<rank0>[01]) pid=\d+\) )"
        r"(?P<second>\(Worker_TP(?P<rank1>[01]) pid=\d+\) )"
        r"(?P<body>INFO [^\n]*worker commit acknowledged request_id=[^\n]+)\n"
        r"(?P=body)(?=\n|$)",
        re.MULTILINE,
    )
    recovered = []

    def replace(match):
        if match["rank0"] == match["rank1"]:
            return match[0]
        recovered.append(
            {
                "kind": "two-prefix-identical-pair",
                "raw_sha256": digest(match[0].encode()),
                "explicit_ranks": [int(match["rank0"]), int(match["rank1"])],
            }
        )
        return match["first"] + match["body"] + "\n" + match["second"] + match["body"]

    return pattern.sub(replace, log), recovered


def verify_transactions(
    log: str, rows: list[dict], enabled: bool, *, pressure: bool
) -> dict:
    log, recoveries = recover_worker_pairs(log)
    commit_pattern = (
        r"scheduler commit request_id=(\S+) semantic_tokens=(\d+) "
        r"physical_tokens=(\d+) source_blocks=(\d+) destination_blocks=(\d+) released_blocks=(\d+)"
    )
    ack_pattern = (
        r"Worker_TP(\d+) [^\n]*?worker commit acknowledged request_id=(\S+) "
        r"semantic_tokens=(\d+) physical_tokens=(\d+)"
    )
    commits = re.findall(commit_pattern, log)
    acks = re.findall(ack_pattern, log)
    require(
        len(commits) == log.count("scheduler commit request_id="),
        "Transaction malformed scheduler entry",
    )
    require(
        len(acks) == log.count("worker commit acknowledged request_id="),
        "Transaction malformed or unranked acknowledgement",
    )
    require(
        ("PyramidKV plugin activated" in log) == enabled,
        "Transaction activation differs",
    )
    by_request = {row["request_id"]: row for row in rows}
    require(len(by_request) == len(rows), "Transaction duplicate request ID")
    owners = {}
    for engine in {row[0] for row in commits} | {row[1] for row in acks}:
        matches = [
            request
            for request in by_request
            if re.fullmatch(re.escape(request) + r"(?:-0)?(?:-[a-fA-F0-9]{8})?", engine)
        ]
        require(
            len(matches) == 1,
            f"Transaction orphan or ambiguous engine request: {engine}",
        )
        owners[engine] = matches[0]
    require(
        all(rank in ("0", "1") for rank, *_ in acks), "Transaction unexpected TP rank"
    )
    output, repeated = [], []
    for request, row in by_request.items():
        selected = [item for item in commits if owners[item[0]] == request]
        expected = enabled and row["prompt_tokens"] > 4096
        require(
            bool(selected) == expected,
            f"Transaction admission differs: {row['case_id']}",
        )
        require(
            pressure or len(selected) <= 1,
            "Transaction unexpected recompression outside pressure",
        )
        if len(selected) > 1:
            repeated.append({"case_id": row["case_id"], "commits": len(selected)})
        engines = [engine for engine, owner in owners.items() if owner == request]
        for engine in engines:
            wanted = [
                (semantic, physical)
                for name, semantic, physical, *_ in selected
                if name == engine
            ]
            for rank in ("0", "1"):
                found = [
                    (semantic, physical)
                    for r, name, semantic, physical in acks
                    if r == rank and name == engine
                ]
                require(
                    found == wanted,
                    f"Transaction TP{rank} acknowledgement sequence differs: {engine}",
                )
        for ordinal, (
            engine,
            semantic,
            physical,
            source,
            destination,
            released,
        ) in enumerate(selected, 1):
            semantic, physical, source, destination, released = map(
                int, (semantic, physical, source, destination, released)
            )
            require(
                semantic == row["prompt_tokens"]
                and physical == 2048
                and physical < semantic,
                f"Transaction compression lengths differ: {engine}",
            )
            require(
                source > destination > 0 and released == source,
                f"Transaction block conservation differs: {engine}",
            )
            output.append(
                {
                    "case_id": row["case_id"],
                    "engine_request_id": engine,
                    "lifecycle_ordinal": ordinal,
                    "semantic_tokens": semantic,
                    "physical_tokens": physical,
                    "source_blocks": source,
                    "destination_blocks": destination,
                    "released_blocks": released,
                    "acknowledged_tp_ranks": [0, 1],
                }
            )
    return {
        "transactions": output,
        "repeated_commit_cases": repeated,
        "requires_recomputation_review": bool(repeated),
        "log_recoveries": recoveries,
    }
