"""Stable consumer contract for the benchmark-owned preparation distribution."""

from __future__ import annotations

from importlib import metadata
import json
import hashlib
from pathlib import Path
import zipfile

from . import PROVIDER_CONTRACT_VERSION, __version__
from .registry import all_adapters, read_catalog, source_spec
from .taxonomy import delivery_assets, delivery_classes, research_classes

TAU2_ORIGINAL_REVISION = "f8de30c298689cbe0117d76a378e7315a17e5bd8"  # pragma: allowlist secret - public upstream revision
TAU2_ORIGINAL_ARCHIVE_SHA256 = "65f1d59b3783bfb38664142fb166aa40b369e4b4083170723059d8f1883ccd64"  # pragma: allowlist secret - official archive checksum

AGENTBENCH_REVISION = "d1e4a10db08c87075c78972e48ecc182be03e2d5"  # pragma: allowlist secret - public upstream revision
AGENTBENCH_ARCHIVE_SHA256 = "426478d0442a542117d4cf517ccc6c8a04c2bdf44a39983ff2f041ed436051d7"  # pragma: allowlist secret - official archive checksum
AGENTBENCH_GAME_ARCHIVES = {
    "json": "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc",  # pragma: allowlist secret - official game archive checksum
    "pddl": "913942ebed06659ea0da2f8122512d98bc6add30d84961ca803132d8fbcad585",  # pragma: allowlist secret - official game archive checksum
    "tw-pddl": "d376c4bbb097c068528a2a9805e358e4cd14bc75c4d68a24e5141aebd234ef67",  # pragma: allowlist secret - official game archive checksum
}


LONGBENCH_V1_DATA_ARCHIVE_SHA256 = "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64"  # pragma: allowlist secret - official archive checksum
LONGBENCH_QASPER_SHA256 = "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854"  # pragma: allowlist secret - original data checksum


BFCL_ARCHIVE_SHA256 = "23db73c7b67850696bf4316a92fe97ba54a5dfb46e822bc294e5158b16efffb0"  # pragma: allowlist secret - official source archive checksum
BFCL_SIMPLE_MEMBERS = {
    "BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",  # pragma: allowlist secret - official task checksum
    "possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",  # pragma: allowlist secret - official scorer asset checksum
}

QRECC_ARCHIVE_SHA256 = "6ed52a0b0e495f752424b15742119a5f8f33ca7a029c07122be1b849de19a308"  # pragma: allowlist secret - official archive checksum
QRECC_TEST_SHA256 = "a4d8ef686827feb00c90997e0d6ff135928bf6f4bcb26fb523b5fac025d58f84"  # pragma: allowlist secret - official test member checksum

BBH_ARCHIVE_SHA256 = "8406fd092f7050638d60e7560952a62f78eeea38ac5a2e5755dff1e12efe44fc"  # pragma: allowlist secret - official publisher archive checksum


def _validate_bbh_publisher_prompt(raw: bytes, expected_demos: int = 3) -> None:
    try:
        text = raw.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        raise ValueError("original BBH publisher prompt is not UTF-8") from None
    separator = "\n-----\n"
    if text.count(separator) != 1:
        raise ValueError("original BBH publisher prompt header boundary changed")
    prefix = text.split(separator, 1)[1]
    demonstrations = [item for item in prefix.split("\n\n") if item.startswith("Q: ")]
    if len(demonstrations) != expected_demos:
        raise ValueError("original BBH publisher prompt demonstration count mismatch")


def _prepare_bbh_assets(*, data_archive, output, task_ids):
    archive = Path(data_archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != BBH_ARCHIVE_SHA256:
        raise ValueError("original BBH archive identity mismatch")
    spec = source_spec("BBH")
    task_assets = [asset for asset in spec["files"] if asset["role"] == "data"]
    prompt_assets = [
        asset for asset in spec["files"] if asset["role"] == "publisher_prompt"
    ]
    task_ids_available = {Path(asset["path"]).stem for asset in task_assets}
    prompt_ids_available = {Path(asset["path"]).stem for asset in prompt_assets}
    if (
        len(task_assets) != 27
        or len(prompt_assets) != 27
        or task_ids_available != prompt_ids_available
        or len(task_ids_available) != 27
    ):
        raise ValueError("BBH manifest must register all 27 task/prompt pairs")
    if not task_ids or not task_ids.issubset(task_ids_available):
        raise ValueError("requested BBH task IDs are not registered in this revision")
    selected = [
        asset for asset in spec["files"] if Path(asset["path"]).stem in task_ids
    ]
    contents = {}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.namelist()
        for asset in selected:
            relative = asset["path"]
            member = asset.get("archive_member")
            if not member:
                member = (
                    f"bbh/{Path(relative).name}"
                    if asset["role"] == "data"
                    else relative
                )
            names = [
                name
                for name in members
                if name == member or name.endswith("/" + member)
            ]
            if len(names) != 1:
                raise ValueError("missing or ambiguous original BBH archive member")
            raw = bundle.read(names[0])
            if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("original BBH member identity mismatch")
            if asset.get("bytes") is not None and len(raw) != asset["bytes"]:
                raise ValueError("original BBH member size mismatch")
            if asset["role"] == "data":
                try:
                    rows = json.loads(raw).get("examples")
                except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                    raise ValueError("invalid original BBH task JSON") from None
                if not isinstance(rows, list) or len(rows) != asset["rows"]:
                    raise ValueError("original BBH example count mismatch")
                if any(
                    not isinstance(row, dict)
                    or not isinstance(row.get("input"), str)
                    or not row["input"].strip()
                    or not isinstance(row.get("target"), str)
                    or not row["target"].strip()
                    for row in rows
                ):
                    raise ValueError("original BBH task fields are incomplete")
            else:
                _validate_bbh_publisher_prompt(
                    raw, expected_demos=asset.get("publisher_demos", 3)
                )
            contents[relative] = raw
    for relative in contents:
        path = Path(relative)
        if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
            raise ValueError("unsafe BBH output path in pinned manifest")
    target = Path(output)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    for relative, raw in contents.items():
        path = Path(relative)
        destination = target / path
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes(raw)
        destination.chmod(0o600)
    data_assets = [asset for asset in selected if asset["role"] == "data"]
    return {
        "source_id": "BBH",
        "subtasks": sorted(task_ids),
        "task_file_count": len(data_assets),
        "publisher_prompt_file_count": len(selected) - len(data_assets),
        "original_rows": sum(asset["rows"] for asset in data_assets),
        "status": "ORIGINAL_FILES_PREPARED_NOT_EXECUTED",
        "revision": spec["revision"],
        "files": [
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in sorted(contents.items())
        ],
        "model_calls": 0,
        "scorer_runs": 0,
    }


def prepare_bbh_boolean(*, data_archive, output):
    """Backward-compatible preparation of the historically validated BBH task."""
    result = _prepare_bbh_assets(
        data_archive=data_archive,
        output=output,
        task_ids={"boolean_expressions"},
    )
    result["subsuite"] = "boolean_expressions"
    result["status"] = "ORIGINAL_FILES_PREPARED_NOT_EXECUTED"
    result.pop("task_file_count", None)
    result.pop("publisher_prompt_file_count", None)
    result.pop("original_rows", None)
    return result


def prepare_bbh_original_tasks(*, data_archive, output):
    """Prepare all original BBH task/prompt assets without model or scorer calls."""
    spec = source_spec("BBH")
    task_ids = {
        Path(asset["path"]).stem for asset in spec["files"] if asset["role"] == "data"
    }
    result = _prepare_bbh_assets(
        data_archive=data_archive,
        output=output,
        task_ids=task_ids,
    )
    result["status"] = "ALL_ORIGINAL_BBH_FILES_PREPARED_NOT_EXECUTED"
    return result


def prepare_qrecc_test(*, data_archive, output):
    """Copy unchanged original test tasks only, not retrieval passages or a runner."""
    archive = Path(data_archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != QRECC_ARCHIVE_SHA256:
        raise ValueError("original QReCC archive identity mismatch")
    name = "qrecc_test.json"
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist().count(name) != 1:
            raise ValueError("missing or ambiguous original QReCC member")
        raw = bundle.read(name)
    if hashlib.sha256(raw).hexdigest() != QRECC_TEST_SHA256:
        raise ValueError("original QReCC member identity mismatch")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 16451:
        raise ValueError("original QReCC test row count mismatch")
    target = Path(output)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = target / name
    path.write_bytes(raw)
    path.chmod(0o600)
    return {
        "source_id": "QRECC",
        "status": "ORIGINAL_TEST_FILE_PREPARED_NOT_EXECUTED",
        "files": [
            {
                "path": name,
                "sha256": QRECC_TEST_SHA256,
                "rows": len(rows),
                "role": "data",
            }
        ],
        "model_calls": 0,
        "retrieval_corpus_prepared": False,
        "scorer_runs": 0,
    }


def prepare_bfcl_simple_python(*, official_archive, output):
    """Prepare unchanged task and separate scorer assets; no model/tool execution."""
    archive = Path(official_archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != BFCL_ARCHIVE_SHA256:
        raise ValueError("original BFCL archive identity mismatch")
    raw_files = {}
    with zipfile.ZipFile(archive) as bundle:
        prefix = "ShishirPatil-gorilla-6ea5797/berkeley-function-call-leaderboard/bfcl_eval/data/"
        for relative, expected in BFCL_SIMPLE_MEMBERS.items():
            if bundle.namelist().count(prefix + relative) != 1:
                raise ValueError("missing or ambiguous original BFCL member")
            raw = bundle.read(prefix + relative)
            if (
                hashlib.sha256(raw).hexdigest() != expected
                or len(raw.splitlines()) != 400
            ):
                raise ValueError("original BFCL member identity/count mismatch")
            raw_files[relative] = raw
    target = Path(output)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    files = []
    for relative, raw in raw_files.items():
        path = target / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o600)
        files.append(
            {
                "path": relative,
                "sha256": BFCL_SIMPLE_MEMBERS[relative],
                "rows": 400,
                "role": "scorer_reference"
                if relative.startswith("possible_answer/")
                else "data",
            }
        )
    return {
        "source_id": "BFCL",
        "subsuite": "v4/simple_python",
        "status": "ORIGINAL_FILES_PREPARED_NOT_EXECUTED",
        "files": files,
        "model_calls": 0,
        "scorer_runs": 0,
    }


def prepare_longbench_qasper(*, data_archive, output):
    """Copy the one fixed publisher member, byte unchanged, to a fresh data root.

    No network, model calls, prompt construction, feature extraction or scoring.
    All other LongBench members are deliberately outside this preparation scope.
    """
    archive = Path(data_archive)
    if (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        != LONGBENCH_V1_DATA_ARCHIVE_SHA256
    ):
        raise ValueError("original LongBench v1 data archive identity mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.namelist().count("data/qasper.jsonl") != 1:
            raise ValueError("missing or ambiguous original qasper member")
        raw = bundle.read("data/qasper.jsonl")
    if hashlib.sha256(raw).hexdigest() != LONGBENCH_QASPER_SHA256:
        raise ValueError("original qasper file identity mismatch")
    if len(raw.splitlines()) != 200:
        raise ValueError("original qasper row count mismatch")
    target = Path(output)
    target.mkdir(mode=0o700, parents=True, exist_ok=False)
    path = target / "qasper.jsonl"
    path.write_bytes(raw)
    path.chmod(0o600)
    return {
        "source_id": "LONGBENCH",
        "subsuite": "qasper",
        "status": "ORIGINAL_FILE_PREPARED_NOT_EXECUTED",
        "file": "qasper.jsonl",
        "sha256": LONGBENCH_QASPER_SHA256,
        "rows": 200,
        "model_calls": 0,
        "scorer_runs": 0,
    }


def agentbench_task_references(*, official_archive, game_archives, data_root, indices):
    """Original ALFWorld FC indices and verified file references, no task payload.

    This does not start an environment, select actions, call models or score.
    Only the fixed new_std loader order is supported, not other AgentBench tasks.
    """
    source = Path(official_archive)
    if hashlib.sha256(source.read_bytes()).hexdigest() != AGENTBENCH_ARCHIVE_SHA256:
        raise ValueError("official AgentBench archive identity mismatch")
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("explicit unique original integer indices required")
    with zipfile.ZipFile(source) as bundle:
        names = [
            n for n in bundle.namelist() if n.endswith("/data/alfworld/new_std.json")
        ]
        if len(names) != 1:
            raise ValueError("ambiguous original AgentBench index")
        raw = bundle.read(names[0])
    rows = [value for group in json.loads(raw).values() for value in group]
    if any(i >= len(rows) for i in indices):
        raise ValueError("original AgentBench index outside selected split")
    roles = {}
    for path in game_archives:
        path = Path(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = [
            key
            for key, expected in AGENTBENCH_GAME_ARCHIVES.items()
            if expected == digest
        ]
        if len(matches) != 1 or matches[0] in roles:
            raise ValueError("unknown or repeated original game archive")
        roles[matches[0]] = path
    if set(roles) != set(AGENTBENCH_GAME_ARCHIVES):
        raise ValueError("all three original game release archives required")
    base = Path(data_root).resolve()
    results = []
    for index in indices:
        relative = rows[index]
        game = (base / relative).resolve()
        if not game.is_relative_to(base) or not relative.endswith("/game.tw-pddl"):
            raise ValueError("unsafe or unsupported original game reference")
        # Verify all publisher files in this episode directory, including task
        # description/state files. Return hashes only, never their JSON payloads.
        assets = []
        directory = str(Path(relative).parent).replace("\\", "/") + "/"
        for role, archive_path in roles.items():
            with zipfile.ZipFile(archive_path) as bundle:
                for name in bundle.namelist():
                    if not name.startswith(directory) or name.endswith("/"):
                        continue
                    target = (base / name).resolve()
                    if not target.is_relative_to(base):
                        raise ValueError("original asset path escapes data root")
                    original = bundle.read(name)
                    if target.read_bytes() != original:
                        raise ValueError(
                            "original game asset differs from publisher release"
                        )
                    assets.append(
                        {
                            "file": name,
                            "sha256": hashlib.sha256(original).hexdigest(),
                            "release_role": role,
                        }
                    )
        if relative not in {asset["file"] for asset in assets}:
            raise ValueError("original game absent from release archives")
        assets.sort(key=lambda asset: (asset["file"], asset["release_role"]))
        results.append(
            {
                "schema_version": "native-task-reference/v1",
                "source_id": "AGENTBENCH",
                "revision": AGENTBENCH_REVISION,
                "domain": "alfworld",
                "variant": "FC",
                "task_set": "new_std",
                "native_task_id": str(index),
                "input_ref": {
                    "file": "data/alfworld/new_std.json",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "row_index": index,
                    "game_file": relative,
                    "game_sha256": hashlib.sha256(game.read_bytes()).hexdigest(),
                    "assets": assets,
                },
                "runtime_kind": "official_native_task",
                "scoring": "OFFICIAL_RUNNER_REQUIRED",
                "preparation_only_not_execution": True,
            }
        )
    return results


def tau2_task_references(*, official_archive, data_root, task_ids):
    """Return original telecom task references, never user/scoring payloads.

    This is preparation only. The official runner owns task loading, user
    simulation, environment operations and scoring. Caller fixes IDs first.
    """
    archive_path = Path(official_archive)
    if (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != TAU2_ORIGINAL_ARCHIVE_SHA256
    ):
        raise ValueError("official tau2 archive identity mismatch")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(value, str) or not value for value in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("explicit unique original task IDs required")
    relative = "tau2/domains/telecom/tasks_full.json"
    path = Path(data_root) / relative
    raw = path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            name for name in archive.namelist() if name.endswith("/data/" + relative)
        ]
        if len(names) != 1 or raw != archive.read(names[0]):
            raise ValueError("native task file differs from official archive")
    rows = json.loads(raw)
    positions = {row["id"]: index for index, row in enumerate(rows)}
    if len(positions) != len(rows) or not set(task_ids) <= positions.keys():
        raise ValueError("unknown or ambiguous original task ID")
    digest = hashlib.sha256(raw).hexdigest()
    return [
        {
            "schema_version": "native-task-reference/v1",
            "source_id": "TAUBENCH",
            "revision": TAU2_ORIGINAL_REVISION,
            "domain": "telecom",
            "task_set": "telecom_full",
            "native_task_id": identifier,
            "input_ref": {
                "file": relative,
                "sha256": digest,
                "row_index": positions[identifier],
            },
            "runtime_kind": "official_native_task",
            "scoring": "OFFICIAL_RUNNER_REQUIRED",
            "preparation_only_not_execution": True,
        }
        for identifier in task_ids
    ]


def provider_identity() -> dict:
    """Check the imported version against distribution metadata, never auto-install."""
    try:
        installed_version = metadata.version("vllm-hust-workload-data")
    except metadata.PackageNotFoundError:
        installed_version = None
    if installed_version is not None and installed_version != __version__:
        raise RuntimeError(
            "a different vllm-hust-workload-data installation shadows this provider"
        )
    try:
        provenance = read_catalog("migration_provenance.json")
    except FileNotFoundError:
        provenance = json.loads(
            (
                Path(__file__).resolve().parents[2] / "MIGRATION_PROVENANCE.json"
            ).read_text(encoding="utf-8")
        )
    return {
        "distribution": "vllm-hust-workload-data",
        "version": __version__,
        "installed_version": installed_version,
        "contract": PROVIDER_CONTRACT_VERSION,
        "module_root": str(Path(__file__).resolve().parent),
        "installation": "INSTALLED"
        if installed_version
        else "SOURCE_ONLY_NOT_INSTALL_VERIFIED",
        "provenance": provenance,
    }


def class_plan(class_id: str) -> dict:
    """Describe a v2 leaf without silently forwarding it as a v1 family filter."""
    classes = {entry["family_id"]: entry for entry in research_classes()}
    if class_id not in classes:
        raise ValueError("unknown workload class")
    definition = classes[class_id]
    rules = [
        rule
        for rule in read_catalog("workload_taxonomy_v2.json")["record_rules"]
        if rule["family_id"] == class_id
    ]
    return {
        "class_id": class_id,
        "compatibility_family": definition["parent_family_id"],
        "candidate_sources": definition["candidate_source_ids"],
        "reviewed_native_subtypes": rules,
        "selection_scope": "CANDIDATE_SOURCES_NOT_RECORD_MEMBERSHIP",
        "instruction": "Select an explicit source and its reviewed native subtype; do not run the whole source as this class.",
    }


def delivery_class_plan(class_id: str) -> dict:
    """Lookup a plan ID or stable family ID, never claim dataset readiness."""
    for row in delivery_classes():
        if class_id in {row["workload_id"], row["family_id"]}:
            return row
    raise ValueError("unknown delivery-plan workload class")


def delivery_source_plan(source_id: str) -> dict:
    """Exact source lookup in the 118-source plan; no reader or readiness inference."""
    for row in delivery_assets():
        if row["source_id"] == source_id:
            return row
    raise ValueError("unknown delivery-plan source")


def list_sources(*, class_id: str | None = None) -> list[dict]:
    manifests = read_catalog("source_manifests_v1.json")["sources"]
    adapters = all_adapters()
    allowed = (
        set(class_plan(class_id)["candidate_sources"]) if class_id else set(manifests)
    )
    rows = []
    for source_id in sorted(allowed):
        source = source_spec(source_id)
        spec = adapters.get(source_id)
        rows.append(
            {
                "source_id": source_id,
                "revision": source.get("revision"),
                "kind": spec["kind"] if spec else "NOT_IMPLEMENTED",
                "adapter_definition": spec.get("status", "DEFINED")
                if spec
                else "NOT_IMPLEMENTED",
                "pinned_data_files": sum(
                    f.get("role") == "data" for f in source.get("files", [])
                ),
                "pinned_native_archives": sum(
                    f.get("role") in {"source_archive", "native_game_archive"}
                    for f in source.get("files", [])
                ),
                "native_reference_command": source.get("native_reference_command"),
                "local_availability": "NOT_CHECKED_ON_THIS_MACHINE",
                "runtime_validation": "NOT_ESTABLISHED_BY_CATALOG",
                "class_membership": "CANDIDATE_ONLY" if class_id else "NOT_SELECTED",
            }
        )
    return rows
