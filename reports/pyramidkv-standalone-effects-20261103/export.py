"""Build an explicit public evidence capsule while retaining private-input hashes."""

import gzip
import io
import json
import re
import tarfile
from pathlib import Path

from evidence import digest, read_json, require, safe_member
from private_inputs import CASE_METADATA, public_plan

RUN_FILES = (
    "process.json",
    "ready.json",
    "completion.json",
    "shutdown.json",
    "failure.json",
    "observed-runtime.json",
    "serve.sh",
    "client-command.json",
    "client.log",
    "server.log",
    "npu-before.txt",
    "npu-after.txt",
    "startup-cache-events.json",
)
EVALUATION_FILES = (
    "results.json",
    "groups.json",
    "summary.json",
    "execution.json",
    "telemetry.jsonl",
    "metrics-before.txt",
    "metrics-after.txt",
)
RESULT_FIELDS = CASE_METADATA | {
    "http_status",
    "finish_reason",
    "success",
    "request_id",
    "output",
    "usage",
    "qa_f1",
    "ttft_seconds",
    "e2e_seconds",
    "tpot_seconds",
    "error",
}


def file_record(path: Path, content: bytes) -> dict:
    return {
        "bytes": len(content),
        "sha256": digest(content),
        "source_mtime_ns": path.stat().st_mtime_ns,
    }


def collect_run(root: Path, name: str, cases: dict) -> tuple[dict, dict, dict]:
    require(re.fullmatch(r"[A-Za-z0-9_-]+", name), "Export unsafe lifecycle name")
    public, private, origins = {}, {}, {}
    folder = root / "runs" / name

    def retain(path, confidential=False):
        require(
            path.is_file() and not path.is_symlink(),
            f"Export nonregular file: {path.name}",
        )
        key = path.relative_to(root).as_posix()
        content = path.read_bytes()
        record = file_record(path, content)
        if confidential:
            private[key] = record
        else:
            public[key] = content
            origins[key] = record

    for relative in (*RUN_FILES, *("evaluation/" + name for name in EVALUATION_FILES)):
        path = folder / relative
        if path.exists():
            retain(path)
    result_path = folder / "evaluation/results.json"
    if not result_path.exists() and (folder / "failure.json").is_file():
        require(
            not (folder / "evaluation/requests").exists(),
            "Export startup failure has unclassified requests",
        )
        return public, private, origins
    require(result_path.is_file(), "Export lacks retained results")
    rows = read_json(result_path.read_bytes())
    request_folder = folder / "evaluation/requests"
    expected = set()
    seen = set()
    for row in rows:
        key = row["case_id"]
        require(
            key in cases and key not in seen and re.fullmatch(r"[A-Za-z0-9_-]+", key),
            "Export unexpected request identity",
        )
        seen.add(key)
        case = cases[key]
        require(
            row.get("kind") == case["kind"],
            "Export result kind differs from frozen plan",
        )
        require(
            not set(row) - RESULT_FIELDS, "Export private or unrecognized result fields"
        )
        result = request_folder / f"{key}.json"
        require(
            result.is_file() and read_json(result.read_bytes()) == row,
            "Export per-request result differs",
        )
        for suffix in (".json", ".sse", ".request.json"):
            filename = key + suffix
            expected.add(filename)
            path = request_folder / filename
            if row["success"]:
                require(
                    path.is_file(),
                    f"Export missing successful request original: {filename}",
                )
            if path.exists():
                retain(
                    path,
                    confidential=case["kind"] == "quality"
                    and suffix == ".request.json",
                )
    require(
        {path.name for path in request_folder.iterdir()} <= expected,
        "Export unclassified request original",
    )
    return public, private, origins


TOP_FILES = (
    "contract.json",
    "contract-v1.json",
    "contract-v2.json",
    "series-state.json",
    "series-controller.log",
    "continue_series.py",
    "capture_runtime.py",
    "env.sh.source",
    "preregistration.json",
    "pressure-policy-preregistration.json",
    "pressure-policy-amendment.json",
    "amendment-timing-check.json",
    "timeout-clarification.json",
    "hf-metadata.json",
    "wheel-npu-tests.txt",
    "npu-before.txt",
    "npu-final.txt",
    "final-identity.json",
    "vllm_ascend_pyramidkv_hust-0.4.0rc1-py3-none-any.whl",
)


def collect_experiment(root: Path) -> tuple[dict, dict, dict]:
    state = read_json((root / "series-state.json").read_bytes())
    require(
        state["phase"] in ("all_requests_executed", "stopped"),
        "Export refuses a live measurement series",
    )
    contract = read_json((root / "contract.json").read_bytes())
    plans = {row["plan"] for row in contract["lifecycles"]}
    private_paths = {name + "/cases.json" for name in plans}
    public, private, origins = {}, {}, {}

    def retain(name, confidential=False):
        require(safe_member(name), f"Export unsafe source path: {name}")
        path = root / name
        require(
            path.is_file() and not path.is_symlink(),
            f"Export missing or nonregular source: {name}",
        )
        content = path.read_bytes()
        record = file_record(path, content)
        if confidential:
            private[name] = record
        else:
            public[name], origins[name] = content, record
        return content

    for name, expected in contract["files_sha256"].items():
        content = retain(name, confidential=name in private_paths)
        require(
            digest(content) == expected, f"Export frozen source digest differs: {name}"
        )
    for name in TOP_FILES:
        if (root / name).is_file():
            retain(name)
    plan_cases, data_hashes = {}, {}
    for name in sorted(plans):
        require(safe_member(name) and "/" not in name, "Export unsafe plan name")
        cases = read_json((root / name / "cases.json").read_bytes())
        manifest = read_json((root / name / "manifest.json").read_bytes())
        require(
            digest((root / name / "cases.json").read_bytes())
            == manifest["cases_sha256"],
            "Export case manifest differs",
        )
        require(
            digest((root / name / "groups.json").read_bytes())
            == manifest["groups_sha256"],
            "Export group manifest differs",
        )
        index, synthetic = public_plan(cases)
        public[name + "/case-index.json"] = json_bytes(index)
        public[name + "/synthetic-cases.json"] = json_bytes(synthetic)
        plan_cases[name] = {case["case_id"]: case for case in cases}
        for task, sha in manifest.get("dataset_files_sha256", {}).items():
            require(
                task in ("narrativeqa", "qasper", "2wikimqa"),
                "Export unknown private dataset",
            )
            require(
                task not in data_hashes or data_hashes[task] == sha,
                "Export inconsistent dataset identities",
            )
            data_hashes[task] = sha
    for task, sha in data_hashes.items():
        content = retain(f"data/{task}.jsonl", confidential=True)
        require(digest(content) == sha, "Export private dataset digest differs")
    names = {row["name"] for row in contract["lifecycles"]}
    require(
        {path.name for path in (root / "runs").iterdir()} <= names,
        "Export unclassified lifecycle directory",
    )
    for row in contract["lifecycles"]:
        if (root / "runs" / row["name"]).exists():
            run_public, run_private, run_origins = collect_run(
                root, row["name"], plan_cases[row["plan"]]
            )
            public.update(run_public)
            private.update(run_private)
            origins.update(run_origins)
    public["private-inputs.json"] = json_bytes(private)
    return public, private, origins


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def write_checksums(report: Path) -> None:
    files = sorted(
        path
        for path in report.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    require(
        all(not path.is_symlink() for path in files), "Export cannot checksum symlinks"
    )
    (report / "SHA256SUMS").write_text(
        "".join(f"{digest(path.read_bytes())}  {path.name}\n" for path in files)
    )


def write_capsule(
    report: Path, members: dict[str, bytes], origins: dict, *, part_bytes: int = 480000
) -> None:
    require(
        not list(report.glob("raw.tar.gz*")),
        "Export refuses existing archive originals",
    )
    require(
        type(part_bytes) is int and part_bytes > 0,
        "Export requires a positive archive part size",
    )
    require(all(safe_member(name) for name in members), "Export unsafe archive member")
    buffer = io.BytesIO()
    inventory = {}
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, content in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(content))
            record = {"bytes": len(content), "sha256": digest(content)}
            if name in origins:
                record["source_mtime_ns"] = origins[name]["source_mtime_ns"]
            else:
                record["derived"] = True
            inventory[name] = record
    packed = gzip.compress(buffer.getvalue(), mtime=0)
    if len(packed) <= part_bytes:
        (report / "raw.tar.gz").write_bytes(packed)
    else:
        for ordinal, offset in enumerate(range(0, len(packed), part_bytes)):
            (report / f"raw.tar.gz.part-{ordinal:03d}").write_bytes(
                packed[offset : offset + part_bytes]
            )
    encoded = json_bytes(inventory)
    if len(encoded) > 480000:
        (report / "members.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    else:
        (report / "members.json").write_bytes(encoded)
    write_checksums(report)
