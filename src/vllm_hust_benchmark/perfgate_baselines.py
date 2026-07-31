from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from vllm_hust_benchmark.perfgate_measurement import validate_measurement_block
from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash

BASELINE_SCHEMA_VERSION = "perfgate-baseline/v1"
REVOCATION_SCHEMA_VERSION = "perfgate-baseline-revocation/v1"
REVOCATION_STATUSES = frozenset({"quarantined", "withdrawn"})
SUPPORTED_SCENARIOS = frozenset({"random-online"})
SUPPORTED_TARGET_REPOSITORIES = {
    "vllm-hust/vllm-hust": "vLLM-HUST/vllm-hust",
    "vllm-hust/vllm-ascend-hust": "vLLM-HUST/vllm-ascend-hust",
}
SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
REQUIRED_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms", "error_rate")
DEFAULT_BASELINE_BRANCH = "benchmark-baselines"
DEFAULT_WRITER_NAME = "vLLM-HUST Baseline Writer"
DEFAULT_WRITER_EMAIL = "baseline-writer@vllm-hust.local"


@dataclass(frozen=True)
class BaselineIdentity:
    target_repository: str
    target_sha: str
    scenario: str
    spec_id: str
    spec_hash: str


@dataclass(frozen=True)
class BaselineProvenance:
    vllm_hust_sha: str
    vllm_ascend_hust_sha: str
    benchmark_runner_sha: str
    hardware_chip_model: str
    cann_version: str
    torch_version: str
    torch_npu_version: str


def _canonical_repository(value: str) -> str:
    normalized = str(value or "").strip()
    canonical = SUPPORTED_TARGET_REPOSITORIES.get(normalized.lower())
    if canonical is None:
        supported = ", ".join(sorted(SUPPORTED_TARGET_REPOSITORIES.values()))
        raise ValueError(
            f"unsupported target repository: {normalized!r}; supported: {supported}"
        )
    return canonical


def _validate_sha(value: str, *, field: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a full 40-character Git SHA")
    return normalized


def _validate_component(value: str, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or not SAFE_COMPONENT_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must contain only letters, digits, '.', '_' or '-'")
    return normalized


def _validate_metadata_value(value: str, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(character in normalized for character in "\0\r\n"):
        raise ValueError(f"{field} must be a non-empty single-line value")
    return normalized


def _validate_evidence_url(value: str, *, field: str) -> str:
    normalized = _validate_metadata_value(value, field=field)
    if not normalized.startswith("https://"):
        raise ValueError(f"{field} must be an HTTPS URL")
    return normalized


def normalize_identity(identity: BaselineIdentity) -> BaselineIdentity:
    scenario = str(identity.scenario or "").strip()
    if scenario not in SUPPORTED_SCENARIOS:
        supported = ", ".join(sorted(SUPPORTED_SCENARIOS))
        raise ValueError(
            f"unsupported required perfgate scenario: {scenario!r}; supported: {supported}"
        )
    spec_hash = str(identity.spec_hash or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", spec_hash):
        raise ValueError("spec_hash must be a 64-character lowercase SHA-256 digest")
    return BaselineIdentity(
        target_repository=_canonical_repository(identity.target_repository),
        target_sha=_validate_sha(identity.target_sha, field="target_sha"),
        scenario=scenario,
        spec_id=_validate_component(identity.spec_id, field="spec_id"),
        spec_hash=spec_hash,
    )


def normalize_provenance(provenance: BaselineProvenance) -> BaselineProvenance:
    return BaselineProvenance(
        vllm_hust_sha=_validate_sha(provenance.vllm_hust_sha, field="vllm_hust_sha"),
        vllm_ascend_hust_sha=_validate_sha(
            provenance.vllm_ascend_hust_sha, field="vllm_ascend_hust_sha"
        ),
        benchmark_runner_sha=_validate_sha(
            provenance.benchmark_runner_sha, field="benchmark_runner_sha"
        ),
        hardware_chip_model=_validate_metadata_value(
            provenance.hardware_chip_model, field="hardware_chip_model"
        ),
        cann_version=_validate_metadata_value(
            provenance.cann_version, field="cann_version"
        ),
        torch_version=_validate_metadata_value(
            provenance.torch_version, field="torch_version"
        ),
        torch_npu_version=_validate_metadata_value(
            provenance.torch_npu_version, field="torch_npu_version"
        ),
    )


def baseline_relative_dir(identity: BaselineIdentity) -> PurePosixPath:
    identity = normalize_identity(identity)
    owner, repository = identity.target_repository.split("/", maxsplit=1)
    return PurePosixPath(
        "baselines",
        owner,
        repository,
        identity.target_sha,
        identity.scenario,
        identity.spec_id,
        identity.spec_hash,
    )


def latest_pointer_relative_path(identity: BaselineIdentity) -> PurePosixPath:
    identity = normalize_identity(identity)
    owner, repository = identity.target_repository.split("/", maxsplit=1)
    return PurePosixPath(
        "pointers",
        owner,
        repository,
        identity.scenario,
        identity.spec_id,
        "latest-main.json",
    )


def revocation_relative_path(
    identity: BaselineIdentity, status: str | None = None
) -> PurePosixPath:
    identity = normalize_identity(identity)
    owner, repository = identity.target_repository.split("/", maxsplit=1)
    root = PurePosixPath(
        "revoked",
        owner,
        repository,
        identity.target_sha,
        identity.scenario,
        identity.spec_id,
        identity.spec_hash,
    )
    if status is None:
        return root
    if status not in REVOCATION_STATUSES:
        raise ValueError(f"invalid revocation status: {status!r}")
    return root / f"{status}.json"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validate_metrics(payload: dict[str, Any], *, source: Path) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{source}: missing object key metrics")
    invalid: list[str] = []
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if (
            not math.isfinite(number)
            or number < 0
            or (name == "error_rate" and number != 0)
        ):
            invalid.append(name)
    if invalid:
        raise ValueError(f"{source}: invalid required metrics: {', '.join(invalid)}")


def _artifact_target_component(
    payload: dict[str, Any], identity: BaselineIdentity
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("baseline artifact is missing object key metadata")
    artifact_repository = str(metadata.get("github_repository") or "").strip()
    artifact_sha = str(metadata.get("git_commit") or "").strip().lower()
    if artifact_repository.lower() != identity.target_repository.lower():
        raise ValueError(
            "baseline target repository mismatch: "
            f"expected {identity.target_repository}, got {artifact_repository or 'unset'}"
        )
    if artifact_sha != identity.target_sha:
        raise ValueError(
            "baseline target SHA mismatch: "
            f"expected {identity.target_sha}, got {artifact_sha or 'unset'}"
        )

    runtime = metadata.get("runtime_provenance")
    if not isinstance(runtime, dict):
        raise ValueError("baseline artifact is missing metadata.runtime_provenance")
    key = (
        "plugin"
        if identity.target_repository.endswith("/vllm-ascend-hust")
        else "engine"
    )
    component = runtime.get(key)
    if not isinstance(component, dict):
        raise ValueError(
            f"baseline artifact is missing target provenance component: {key}"
        )
    component_repository = str(component.get("repository") or "").strip()
    component_sha = str(component.get("commit") or "").strip().lower()
    if component_repository.lower() != identity.target_repository.lower():
        raise ValueError(
            f"baseline {key} repository mismatch: expected "
            f"{identity.target_repository}, got {component_repository or 'unset'}"
        )
    if component_sha != identity.target_sha:
        raise ValueError(
            f"baseline {key} SHA mismatch: expected {identity.target_sha}, "
            f"got {component_sha or 'unset'}"
        )
    return runtime


def validate_artifact(
    source: Path,
    identity: BaselineIdentity,
    provenance: BaselineProvenance,
) -> dict[str, Any]:
    identity = normalize_identity(identity)
    provenance = normalize_provenance(provenance)
    payload = _load_json_object(source)
    _validate_metrics(payload, source=source)

    same_spec = payload.get("same_spec")
    if not isinstance(same_spec, dict):
        raise ValueError(f"{source}: missing object key same_spec")
    actual_scenario = str(same_spec.get("scenario") or "").strip()
    actual_spec_id = str(same_spec.get("spec_id") or "").strip()
    actual_spec_hash = str(same_spec.get("resolved_spec_hash") or "").strip().lower()
    try:
        computed_spec_hash = compute_resolved_spec_hash(same_spec)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{source}: invalid resolved same-spec payload: {error}"
        ) from error
    if actual_spec_hash != computed_spec_hash:
        raise ValueError(
            "baseline resolved spec hash mismatch: "
            f"declared {actual_spec_hash or 'unset'}, computed {computed_spec_hash}"
        )
    expected = (identity.scenario, identity.spec_id, identity.spec_hash)
    actual = (actual_scenario, actual_spec_id, actual_spec_hash)
    if actual != expected:
        raise ValueError(
            "baseline same-spec identity mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )

    runtime = _artifact_target_component(payload, identity)
    expected_components = {
        "engine": ("vLLM-HUST/vllm-hust", provenance.vllm_hust_sha),
        "plugin": (
            "vLLM-HUST/vllm-ascend-hust",
            provenance.vllm_ascend_hust_sha,
        ),
    }
    for name, (expected_repository, expected_sha) in expected_components.items():
        component = runtime.get(name)
        if not isinstance(component, dict):
            raise ValueError(
                f"baseline artifact is missing provenance component: {name}"
            )
        repository = str(component.get("repository") or "").strip()
        sha = str(component.get("commit") or "").strip().lower()
        if repository.lower() != expected_repository.lower() or sha != expected_sha:
            raise ValueError(
                f"baseline {name} provenance mismatch: expected "
                f"{expected_repository}@{expected_sha}, got "
                f"{repository or 'unset'}@{sha or 'unset'}"
            )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(
    identity: BaselineIdentity,
    provenance: BaselineProvenance,
    *,
    artifact_sha256: str,
    measurement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "identity": asdict(identity),
        "provenance": asdict(provenance),
        "artifact": {
            "path": "run_leaderboard.json",
            "sha256": artifact_sha256,
        },
    }
    if measurement is not None:
        payload["measurement"] = measurement
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_revocation_file(
    repository_root: Path, identity: BaselineIdentity, status: str
) -> dict[str, Any] | None:
    relative_path = revocation_relative_path(identity, status)
    _reject_symlink_components(repository_root, relative_path)
    path = repository_root / relative_path
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"baseline revocation record is not a file: {path}")
    payload = _load_json_object(path)
    if payload.get("schema_version") != REVOCATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported baseline revocation schema in {path}")
    if payload.get("identity") != asdict(identity):
        raise ValueError(f"baseline revocation identity mismatch: {path}")
    actual_status = payload.get("release_visibility_status")
    if actual_status != status:
        raise ValueError(
            f"baseline revocation status mismatch in {path}: {actual_status!r}"
        )
    expected_artifact_path = (
        baseline_relative_dir(identity).as_posix() + "/run_leaderboard.json"
    )
    if payload.get("artifact_path") != expected_artifact_path:
        raise ValueError(f"baseline revocation artifact path mismatch: {path}")
    for field in ("reason", "detected_at", "actor"):
        _validate_metadata_value(payload.get(field), field=field)
    for field in ("detection_run_url", "workflow_url"):
        _validate_evidence_url(payload.get(field), field=field)
    for field in ("publication_commit", "artifact_sha256"):
        value = _validate_metadata_value(payload.get(field), field=field)
        if field == "publication_commit":
            _validate_sha(value, field=field)
        elif field == "artifact_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"artifact_sha256 is invalid in {path}")
    return payload


def _load_revocation(
    repository_root: Path, identity: BaselineIdentity
) -> dict[str, Any] | None:
    # A later withdrawn event supersedes quarantine while both immutable records
    # remain in Git history and in the current tree for audit.
    for status in ("withdrawn", "quarantined"):
        payload = _load_revocation_file(repository_root, identity, status)
        if payload is not None:
            return payload
    return None


def record_revocation(
    repository_root: Path,
    identity: BaselineIdentity,
    *,
    status: str,
    reason: str,
    detected_at: str,
    detection_run_url: str,
    actor: str,
    workflow_url: str,
    publication_commit: str,
) -> Path:
    """Write an immutable revocation record for one exact central baseline."""

    identity = normalize_identity(identity)
    if status not in REVOCATION_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(REVOCATION_STATUSES)}, got {status!r}"
        )
    values = {
        "reason": _validate_metadata_value(reason, field="reason"),
        "detected_at": _validate_metadata_value(detected_at, field="detected_at"),
        "detection_run_url": _validate_evidence_url(
            detection_run_url, field="detection_run_url"
        ),
        "actor": _validate_metadata_value(actor, field="actor"),
        "workflow_url": _validate_evidence_url(workflow_url, field="workflow_url"),
        "publication_commit": _validate_sha(
            publication_commit, field="publication_commit"
        ),
    }
    relative_path = revocation_relative_path(identity, status)
    existing = _load_revocation_file(repository_root, identity, status)
    withdrawn = _load_revocation_file(repository_root, identity, "withdrawn")
    if status == "quarantined" and withdrawn is not None:
        raise ValueError(
            "cannot quarantine an exact baseline after it has been withdrawn: "
            f"{repository_root / revocation_relative_path(identity, 'withdrawn')}"
        )
    if _load_revocation(repository_root, identity) is None:
        # Validate the complete active baseline before recording its withdrawal.
        load_manifest(repository_root, identity, require_measurement=True)
    baseline_dir = repository_root / baseline_relative_dir(identity)
    manifest_path = baseline_dir / "baseline-metadata.json"
    artifact_path = baseline_dir / "run_leaderboard.json"
    if not manifest_path.is_file() or not artifact_path.is_file():
        raise ValueError(f"cannot revoke unavailable exact baseline: {baseline_dir}")
    manifest = _load_json_object(manifest_path)
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"baseline artifact metadata is missing: {manifest_path}")
    artifact_sha256 = str(artifact.get("sha256") or "").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
        raise ValueError(f"baseline artifact checksum is invalid: {manifest_path}")
    if _sha256(artifact_path) != artifact_sha256:
        raise ValueError(f"baseline artifact checksum mismatch: {artifact_path}")
    if (repository_root / ".git").exists():
        artifact_relative_path = (
            baseline_relative_dir(identity).as_posix() + "/run_leaderboard.json"
        )
        publication = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "rev-list",
                "-1",
                "HEAD",
                "--",
                artifact_relative_path,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        actual_publication_commit = publication.stdout.strip().lower()
        if (
            publication.returncode != 0
            or actual_publication_commit != values["publication_commit"]
        ):
            raise ValueError(
                "publication_commit does not match the commit that introduced "
                f"{artifact_relative_path}: expected "
                f"{actual_publication_commit or 'unavailable'}, got "
                f"{values['publication_commit']}"
            )
    payload = {
        "schema_version": REVOCATION_SCHEMA_VERSION,
        "release_visibility_status": status,
        "identity": asdict(identity),
        "artifact_path": (
            baseline_relative_dir(identity).as_posix() + "/run_leaderboard.json"
        ),
        "artifact_sha256": artifact_sha256,
        **values,
    }
    path = repository_root / relative_path
    if existing is not None:
        if existing != payload:
            raise ValueError(
                f"revocation record already exists with different content: {path}"
            )
        return path
    _reject_symlink_components(repository_root, relative_path)
    _write_json(path, payload)
    return path


def _reject_symlink_components(
    repository_root: Path, relative_path: PurePosixPath
) -> None:
    if repository_root.is_symlink():
        raise ValueError(
            f"central baseline repository root is a symlink: {repository_root}"
        )
    current = repository_root
    for component in relative_path.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"central baseline path contains a symlink: {current}")


def store_baseline(
    repository_root: Path,
    source: Path,
    identity: BaselineIdentity,
    provenance: BaselineProvenance,
    *,
    update_latest_pointer: bool = False,
    measurement: dict[str, Any] | None = None,
) -> Path:
    identity = normalize_identity(identity)
    provenance = normalize_provenance(provenance)
    revocation = _load_revocation(repository_root, identity)
    if revocation is not None:
        raise ValueError(
            "cannot store a revoked exact baseline: "
            f"{repository_root / revocation_relative_path(identity)}"
        )
    artifact_payload = validate_artifact(source, identity, provenance)
    if measurement is not None:
        validate_measurement_block(
            measurement,
            artifact_metrics=artifact_payload.get("metrics"),
            context=f"{source}: measurement",
            require_publication_policy=True,
        )

    relative_dir = baseline_relative_dir(identity)
    _reject_symlink_components(repository_root, relative_dir)
    destination_dir = repository_root / relative_dir
    destination = destination_dir / "run_leaderboard.json"
    manifest = destination_dir / "baseline-metadata.json"
    _reject_symlink_components(repository_root, relative_dir / "run_leaderboard.json")
    _reject_symlink_components(repository_root, relative_dir / "baseline-metadata.json")
    artifact_sha256 = _sha256(source)
    expected_manifest = _manifest_payload(
        identity, provenance, artifact_sha256=artifact_sha256, measurement=measurement
    )

    if destination_dir.exists():
        if not destination.is_file() or not manifest.is_file():
            raise ValueError(
                f"existing baseline key is incomplete and cannot be overwritten: {destination_dir}"
            )
        actual_manifest = _load_json_object(manifest)
        if (
            _sha256(destination) != artifact_sha256
            or actual_manifest != expected_manifest
        ):
            raise ValueError(
                f"baseline key already exists with different content: {destination_dir}"
            )
    else:
        destination_dir.mkdir(parents=True)
        shutil.copyfile(source, destination)
        _write_json(manifest, expected_manifest)

    if update_latest_pointer:
        pointer_relative_path = latest_pointer_relative_path(identity)
        _reject_symlink_components(repository_root, pointer_relative_path)
        pointer = repository_root / pointer_relative_path
        _write_json(
            pointer,
            {
                "schema_version": BASELINE_SCHEMA_VERSION,
                "identity": asdict(identity),
                "path": baseline_relative_dir(identity).as_posix()
                + "/run_leaderboard.json",
                "artifact_sha256": artifact_sha256,
            },
        )
    return destination


def load_manifest(
    repository_root: Path,
    identity: BaselineIdentity,
    *,
    require_measurement: bool = False,
) -> tuple[Path, BaselineProvenance]:
    identity = normalize_identity(identity)
    revocation = _load_revocation(repository_root, identity)
    if revocation is not None:
        status = revocation["release_visibility_status"]
        raise ValueError(
            f"exact central baseline is {status}: "
            f"{repository_root / revocation_relative_path(identity, status)}"
        )
    relative_dir = baseline_relative_dir(identity)
    _reject_symlink_components(repository_root, relative_dir / "run_leaderboard.json")
    _reject_symlink_components(repository_root, relative_dir / "baseline-metadata.json")
    baseline_dir = repository_root / relative_dir
    artifact = baseline_dir / "run_leaderboard.json"
    manifest_path = baseline_dir / "baseline-metadata.json"
    if not artifact.is_file() or not manifest_path.is_file():
        raise ValueError(f"exact central baseline is unavailable: {baseline_dir}")
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError(f"unsupported baseline schema in {manifest_path}")
    if manifest.get("identity") != asdict(identity):
        raise ValueError(f"baseline manifest identity mismatch: {manifest_path}")
    provenance_payload = manifest.get("provenance")
    if not isinstance(provenance_payload, dict):
        raise ValueError(f"baseline manifest provenance is missing: {manifest_path}")
    try:
        provenance = normalize_provenance(BaselineProvenance(**provenance_payload))
    except TypeError as error:
        raise ValueError(
            f"invalid baseline provenance in {manifest_path}: {error}"
        ) from error
    artifact_metadata = manifest.get("artifact")
    if not isinstance(artifact_metadata, dict):
        raise ValueError(f"baseline artifact metadata is missing: {manifest_path}")
    expected_digest = str(artifact_metadata.get("sha256") or "").strip()
    if _sha256(artifact) != expected_digest:
        raise ValueError(f"baseline artifact checksum mismatch: {artifact}")
    artifact_payload = validate_artifact(artifact, identity, provenance)
    if "measurement" in manifest:
        validate_measurement_block(
            manifest["measurement"],
            artifact_metrics=artifact_payload.get("metrics"),
            context=f"{manifest_path}: measurement",
            require_publication_policy=True,
        )
    elif require_measurement:
        raise ValueError(f"baseline measurement metadata is missing: {manifest_path}")
    return artifact, provenance


def fetch_baseline(
    repository_root: Path,
    output: Path,
    identity: BaselineIdentity,
    *,
    expected_provenance: BaselineProvenance | None = None,
    require_measurement: bool = False,
) -> Path:
    artifact, provenance = load_manifest(
        repository_root, identity, require_measurement=require_measurement
    )
    if expected_provenance is not None:
        expected = normalize_provenance(expected_provenance)
        if provenance != expected:
            raise ValueError(
                "exact central baseline provenance mismatch: "
                f"expected {expected!r}, got {provenance!r}"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact, output)
    return output


def verify_main_commit(
    repository: Path,
    target_sha: str,
    main_ref: str,
    *,
    require_tip: bool = False,
) -> None:
    target_sha = _validate_sha(target_sha, field="target_sha")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            target_sha,
            main_ref,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ValueError(
            f"target SHA {target_sha} is not an ancestor of {main_ref}{suffix}"
        )
    if require_tip:
        tip = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", main_ref],
            check=False,
            capture_output=True,
            text=True,
        )
        if tip.returncode != 0:
            detail = tip.stderr.strip()
            raise ValueError(f"unable to resolve main ref {main_ref}: {detail}")
        resolved_tip = tip.stdout.strip().lower()
        if resolved_tip != target_sha:
            raise ValueError(
                "latest-main pointer may only reference the current main tip: "
                f"expected {resolved_tip}, got {target_sha}"
            )


def _run_git(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def publish_baseline(
    remote: str,
    branch: str,
    source: Path,
    target_git_repository: Path,
    main_ref: str,
    identity: BaselineIdentity,
    provenance: BaselineProvenance,
    *,
    update_latest_pointer: bool = False,
    max_attempts: int = 5,
    writer_name: str = DEFAULT_WRITER_NAME,
    writer_email: str = DEFAULT_WRITER_EMAIL,
    measurement: dict[str, Any] | None = None,
) -> str:
    identity = normalize_identity(identity)
    provenance = normalize_provenance(provenance)
    branch = _validate_component(branch, field="branch")
    writer_name = _validate_metadata_value(writer_name, field="writer_name")
    writer_email = _validate_metadata_value(writer_email, field="writer_email")
    if not str(remote or "").strip():
        raise ValueError("remote must be non-empty")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if measurement is None:
        raise ValueError("measurement metadata is required for central publication")

    verify_main_commit(
        target_git_repository,
        identity.target_sha,
        main_ref,
        require_tip=update_latest_pointer,
    )
    validate_artifact(source, identity, provenance)

    last_error = ""
    with tempfile.TemporaryDirectory(prefix="perfgate-baseline-writer-") as temp:
        temp_root = Path(temp)
        for attempt in range(1, max_attempts + 1):
            checkout = temp_root / f"attempt-{attempt}"
            _run_git(["clone", "--no-checkout", remote, str(checkout)])
            remote_branch = _run_git(
                [
                    "ls-remote",
                    "--exit-code",
                    "--heads",
                    "origin",
                    branch,
                ],
                cwd=checkout,
                check=False,
            )
            if remote_branch.returncode == 0:
                _run_git(["fetch", "origin", branch], cwd=checkout)
                _run_git(["checkout", "-B", branch, "FETCH_HEAD"], cwd=checkout)
            elif remote_branch.returncode == 2:
                _run_git(["checkout", "--orphan", branch], cwd=checkout)
                _run_git(["rm", "-rf", "--ignore-unmatch", "."], cwd=checkout)
            else:
                detail = remote_branch.stderr.strip() or remote_branch.stdout.strip()
                raise ValueError(
                    f"unable to inspect central baseline branch {branch}: {detail}"
                )

            destination = store_baseline(
                checkout,
                source,
                identity,
                provenance,
                update_latest_pointer=update_latest_pointer,
                measurement=measurement,
            )
            paths_to_add = ["baselines"]
            if (checkout / "pointers").is_dir():
                paths_to_add.append("pointers")
            _run_git(["add", *paths_to_add], cwd=checkout)
            staged = _run_git(
                ["diff", "--cached", "--quiet"], cwd=checkout, check=False
            )
            if staged.returncode == 0:
                return f"unchanged:{destination.relative_to(checkout).as_posix()}"
            if staged.returncode != 1:
                detail = staged.stderr.strip() or staged.stdout.strip()
                raise ValueError(f"unable to inspect staged baseline changes: {detail}")

            _run_git(
                [
                    "-c",
                    f"user.name={writer_name}",
                    "-c",
                    f"user.email={writer_email}",
                    "commit",
                    "-m",
                    "chore(perfgate): store central baseline for "
                    f"{identity.target_repository}@{identity.target_sha[:12]}",
                ],
                cwd=checkout,
            )
            pushed = _run_git(
                ["push", "origin", f"HEAD:refs/heads/{branch}"],
                cwd=checkout,
                check=False,
            )
            if pushed.returncode == 0:
                return f"published:{destination.relative_to(checkout).as_posix()}"
            last_error = pushed.stderr.strip() or pushed.stdout.strip()

    raise ValueError(
        f"failed to publish central baseline after {max_attempts} attempts: {last_error}"
    )


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--spec-id", required=True)
    parser.add_argument("--spec-hash", required=True)


def _add_provenance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vllm-hust-sha", required=True)
    parser.add_argument("--vllm-ascend-hust-sha", required=True)
    parser.add_argument("--benchmark-runner-sha", required=True)
    parser.add_argument("--hardware-chip-model", required=True)
    parser.add_argument("--cann-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torch-npu-version", required=True)


def _identity_from_args(args: argparse.Namespace) -> BaselineIdentity:
    return BaselineIdentity(
        target_repository=args.target_repository,
        target_sha=args.target_sha,
        scenario=args.scenario,
        spec_id=args.spec_id,
        spec_hash=args.spec_hash,
    )


def _provenance_from_args(args: argparse.Namespace) -> BaselineProvenance:
    return BaselineProvenance(
        vllm_hust_sha=args.vllm_hust_sha,
        vllm_ascend_hust_sha=args.vllm_ascend_hust_sha,
        benchmark_runner_sha=args.benchmark_runner_sha,
        hardware_chip_model=args.hardware_chip_model,
        cann_version=args.cann_version,
        torch_version=args.torch_version,
        torch_npu_version=args.torch_npu_version,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage central perfgate baselines")
    commands = parser.add_subparsers(dest="command", required=True)

    store_parser = commands.add_parser("store")
    store_parser.add_argument("--repository-root", type=Path, required=True)
    store_parser.add_argument("--source", type=Path, required=True)
    store_parser.add_argument("--target-git-repository", type=Path, required=True)
    store_parser.add_argument("--main-ref", default="origin/main")
    store_parser.add_argument("--update-latest-pointer", action="store_true")
    store_parser.add_argument(
        "--measurement-file",
        type=Path,
        required=True,
        help="measurement.json produced by perfgate-aggregate-runs; embedded into "
        "the baseline manifest and cross-checked against the artifact metrics.",
    )
    _add_identity_arguments(store_parser)
    _add_provenance_arguments(store_parser)

    fetch_parser = commands.add_parser("fetch")
    fetch_parser.add_argument("--repository-root", type=Path, required=True)
    fetch_parser.add_argument("--output", type=Path, required=True)
    _add_identity_arguments(fetch_parser)
    _add_provenance_arguments(fetch_parser)

    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--repository-root", type=Path, required=True)
    _add_identity_arguments(validate_parser)
    _add_provenance_arguments(validate_parser)

    revoke_parser = commands.add_parser("revoke")
    revoke_parser.add_argument("--repository-root", type=Path, required=True)
    revoke_parser.add_argument(
        "--status", choices=sorted(REVOCATION_STATUSES), required=True
    )
    revoke_parser.add_argument("--reason", required=True)
    revoke_parser.add_argument("--detected-at", required=True)
    revoke_parser.add_argument("--detection-run-url", required=True)
    revoke_parser.add_argument("--actor", required=True)
    revoke_parser.add_argument("--workflow-url", required=True)
    revoke_parser.add_argument("--publication-commit", required=True)
    _add_identity_arguments(revoke_parser)

    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--remote", required=True)
    publish_parser.add_argument("--branch", default=DEFAULT_BASELINE_BRANCH)
    publish_parser.add_argument("--source", type=Path, required=True)
    publish_parser.add_argument("--target-git-repository", type=Path, required=True)
    publish_parser.add_argument("--main-ref", default="origin/main")
    publish_parser.add_argument("--update-latest-pointer", action="store_true")
    publish_parser.add_argument("--max-attempts", type=int, default=5)
    publish_parser.add_argument("--writer-name", default=DEFAULT_WRITER_NAME)
    publish_parser.add_argument("--writer-email", default=DEFAULT_WRITER_EMAIL)
    publish_parser.add_argument(
        "--measurement-file",
        type=Path,
        required=True,
        help="measurement.json produced by perfgate-aggregate-runs; embedded into "
        "the baseline manifest and cross-checked against the artifact metrics.",
    )
    _add_identity_arguments(publish_parser)
    _add_provenance_arguments(publish_parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        identity = _identity_from_args(args)
        if args.command == "revoke":
            print(
                record_revocation(
                    args.repository_root,
                    identity,
                    status=args.status,
                    reason=args.reason,
                    detected_at=args.detected_at,
                    detection_run_url=args.detection_run_url,
                    actor=args.actor,
                    workflow_url=args.workflow_url,
                    publication_commit=args.publication_commit,
                )
            )
            return 0
        provenance = _provenance_from_args(args)
        measurement: dict[str, Any] | None = None
        if getattr(args, "measurement_file", None) is not None:
            measurement = _load_json_object(args.measurement_file)
        if args.command == "store":
            verify_main_commit(
                args.target_git_repository,
                identity.target_sha,
                args.main_ref,
                require_tip=args.update_latest_pointer,
            )
            destination = store_baseline(
                args.repository_root,
                args.source,
                identity,
                provenance,
                update_latest_pointer=args.update_latest_pointer,
                measurement=measurement,
            )
            print(destination)
            return 0
        if args.command == "fetch":
            print(
                fetch_baseline(
                    args.repository_root,
                    args.output,
                    identity,
                    expected_provenance=provenance,
                    require_measurement=True,
                )
            )
            return 0
        if args.command == "validate":
            _artifact, actual_provenance = load_manifest(
                args.repository_root, identity, require_measurement=True
            )
            if actual_provenance != normalize_provenance(provenance):
                raise ValueError("exact central baseline provenance mismatch")
            print("Central perfgate baseline is valid.")
            return 0
        if args.command == "publish":
            print(
                publish_baseline(
                    args.remote,
                    args.branch,
                    args.source,
                    args.target_git_repository,
                    args.main_ref,
                    identity,
                    provenance,
                    update_latest_pointer=args.update_latest_pointer,
                    max_attempts=args.max_attempts,
                    writer_name=args.writer_name,
                    writer_email=args.writer_email,
                    measurement=measurement,
                )
            )
            return 0
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
