"""Evidence-backed registry projection for Issue #136 publication.

This module never edits the committed target registry or public snapshots.  It
validates a complete Issue #136 publication bundle, then emits an auditable
promotion patch and an in-memory/projected registry for review and binding
tests.  Applying the patch remains an explicit, versioned repository change.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from vllm_hust_benchmark.issue136_publication import (
    LOAD_PROFILES,
    SCHEMA_VERSION as BUNDLE_SCHEMA_VERSION,
    WORKLOADS,
    sha256,
)
from vllm_hust_benchmark.snapshot_target_binding import (
    OfficialTargetRegistry,
    official_target_binding_errors,
)


PROMOTION_SCHEMA_VERSION = "issue-136-target-promotion/v1"
BINDING_POLICY_SCHEMA_VERSION = "issue-136-binding-projection/v1"
EXPECTED_CANDIDATES = 24
EXPECTED_PROFILE_CELLS = 12
ENGINE_ALIASES = ({"observed": "vllm-hust", "target": "vllm"},)
EPHEMERAL_TRANSPORT_FIELDS = (
    "server_parameters.host",
    "server_parameters.port",
    "client_parameters.host",
    "client_parameters.port",
)


class PromotionError(ValueError):
    """A bundle or registry cannot be safely projected for promotion."""


def _load_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"invalid {label}: {path}") from exc


def _regular_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PromotionError(f"unsafe bundle path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise PromotionError(f"missing regular bundle file: {relative}")
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise PromotionError(f"bundle file escapes root: {relative}") from exc
    return path


def _validate_bundle_checksums(root: Path) -> str:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise PromotionError("publication bundle must not contain symbolic links")
    manifest = _regular_file(root, Path("SHA256SUMS"))
    covered: dict[Path, str] = {}
    for line_number, raw in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise PromotionError(f"malformed SHA256SUMS:{line_number}")
        try:
            int(parts[0], 16)
        except ValueError as exc:
            raise PromotionError(f"malformed SHA256SUMS:{line_number}") from exc
        posix = PurePosixPath(parts[1].removeprefix("*").removeprefix("./"))
        relative = Path(*posix.parts)
        target = _regular_file(root, relative)
        if relative in covered or sha256(target) != parts[0]:
            raise PromotionError(f"duplicate or mismatched bundle checksum: {relative}")
        covered[relative] = parts[0]
    present = {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(covered) != present:
        raise PromotionError("bundle SHA256SUMS does not cover exactly all files")
    return sha256(manifest)


def _load_candidates(root: Path) -> list[dict[str, Any]]:
    single = _load_json(
        root / "leaderboard_single.candidates.json", "single-chip candidates"
    )
    multi = _load_json(
        root / "leaderboard_multi.candidates.json", "multi-chip candidates"
    )
    if not isinstance(single, list) or not isinstance(multi, list):
        raise PromotionError("candidate files must contain JSON arrays")
    if len(single) != 8 or len(multi) != 16:
        raise PromotionError(
            "publication bundle must contain 8 single and 16 multi rows"
        )
    if any(not isinstance(entry, dict) for entry in single + multi):
        raise PromotionError("publication candidates must be JSON objects")
    return single + multi


def _validate_complete_bundle(
    bundle_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = bundle_root.resolve(strict=True)
    bundle_sha = _validate_bundle_checksums(root)
    manifest = _load_json(root / "promotion-manifest.json", "promotion manifest")
    if not isinstance(manifest, dict):
        raise PromotionError("promotion manifest must be an object")
    policy = manifest.get("promotion_policy")
    profiles = manifest.get("profiles")
    archives = manifest.get("archives")
    counts = manifest.get("snapshot_candidates")
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("status") != "promotion-ready"
        or not isinstance(policy, dict)
        or policy.get("requires_complete_fixed_and_scaled_archives") is not True
        or policy.get("expected_cells_per_profile") != EXPECTED_PROFILE_CELLS
        or policy.get("expected_repeats_per_cell") != 3
        or policy.get("specialty_observations_included") is not False
        or not isinstance(profiles, dict)
        or set(profiles) != set(LOAD_PROFILES)
        or not isinstance(archives, dict)
        or set(archives) != set(LOAD_PROFILES)
        or counts != {"single": 8, "multi": 16, "total": EXPECTED_CANDIDATES}
    ):
        raise PromotionError("publication bundle is not complete and promotion-ready")
    if any(
        not isinstance(archives[profile], str) or len(archives[profile]) != 64
        for profile in LOAD_PROFILES
    ):
        raise PromotionError("publication bundle has invalid archive identities")

    cells: dict[str, dict[str, Any]] = {}
    expected_cell_keys = {
        (profile, workload, tp)
        for profile in LOAD_PROFILES
        for workload in WORKLOADS
        for tp in (1, 2, 4)
    }
    observed_cell_keys: set[tuple[str, str, int]] = set()
    for profile in LOAD_PROFILES:
        profile_cells = profiles.get(profile)
        if not isinstance(profile_cells, list) or len(profile_cells) != 12:
            raise PromotionError(f"profile {profile} does not contain twelve cells")
        for cell in profile_cells:
            if not isinstance(cell, dict):
                raise PromotionError(f"profile {profile} contains an invalid cell")
            key = (profile, cell.get("workload"), cell.get("tensor_parallel_size"))
            spec_id = cell.get("spec_id")
            spec_sha = cell.get("spec_sha256")
            if (
                key not in expected_cell_keys
                or key in observed_cell_keys
                or not isinstance(spec_id, str)
                or not spec_id
                or not isinstance(spec_sha, str)
                or len(spec_sha) != 64
            ):
                raise PromotionError(f"invalid or duplicate promotion cell: {key}")
            observed_cell_keys.add(key)
            if spec_id in cells:
                raise PromotionError(f"target reused across promotion cells: {spec_id}")
            cells[spec_id] = cell
    if observed_cell_keys != expected_cell_keys:
        raise PromotionError("publication bundle does not cover the exact Dense matrix")

    candidates = _load_candidates(root)
    seen_signatures: set[str] = set()
    seen_targets: set[str] = set()
    for entry in candidates:
        same_spec = entry.get("same_spec")
        metadata = entry.get("metadata")
        publication = (
            metadata.get("issue_136_publication")
            if isinstance(metadata, Mapping)
            else None
        )
        if not isinstance(same_spec, Mapping) or not isinstance(publication, Mapping):
            raise PromotionError("candidate lacks same-spec publication provenance")
        target_id = same_spec.get("spec_id")
        profile = publication.get("load_profile")
        signature = publication.get("setting_signature")
        if (
            target_id not in cells
            or profile not in LOAD_PROFILES
            or publication.get("archive_sha256") != archives[profile]
            or not isinstance(signature, str)
            or not signature
            or signature in seen_signatures
            or target_id in seen_targets
        ):
            raise PromotionError("candidate provenance does not match promotion cells")
        seen_signatures.add(signature)
        seen_targets.add(str(target_id))
    if seen_targets != set(cells):
        raise PromotionError("candidate target set does not match promotion cell set")
    return manifest, candidates, bundle_sha


def _registry_payload(registry: OfficialTargetRegistry) -> dict[str, Any]:
    # Projection output preserves the registry's canonical target order.  Other
    # top-level generation metadata is intentionally minimal and review-only.
    return {
        "schema_version": "issue-136-projected-target-registry/v1",
        "status": "review-only-not-committable",
        "base_registry": {
            "version": registry.version,
            "sha256": registry.sha256,
        },
        "targets": [copy.deepcopy(target) for target in registry.targets.values()],
    }


def project_registry_promotion(
    bundle_root: Path, registry: OfficialTargetRegistry
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(patch, projected_registry)`` after complete evidence validation."""
    bundle, candidates, bundle_sha = _validate_complete_bundle(bundle_root)
    candidate_by_target = {
        str(entry["same_spec"]["spec_id"]): entry for entry in candidates
    }
    profiles = bundle["profiles"]
    cells = {
        str(cell["spec_id"]): cell
        for profile in LOAD_PROFILES
        for cell in profiles[profile]
    }
    missing = sorted(set(cells) - set(registry.targets))
    if missing:
        raise PromotionError(
            "registry is missing promotion targets: " + ", ".join(missing)
        )

    projected_payload = _registry_payload(registry)
    projected_by_id = {
        target["target_id"]: target for target in projected_payload["targets"]
    }
    operations = []
    for target_id in sorted(cells):
        target = projected_by_id[target_id]
        cell = cells[target_id]
        entry = candidate_by_target[target_id]
        if (
            target.get("status") != "provisional"
            or target.get("intended_use") != "specialty"
            or target.get("profile") != "dense-scaling"
            or target.get("source_spec", {}).get("sha256") != cell["spec_sha256"]
            or target.get("baseline_runtime", {}).get("engine") != "vllm"
            or entry.get("engine") != "vllm-hust"
        ):
            raise PromotionError(
                f"target is not an exact promotable Dense target: {target_id}"
            )
        binding_policy = {
            "schema_version": BINDING_POLICY_SCHEMA_VERSION,
            "evidence_bundle_sha256": bundle_sha,
            "engine_aliases": [dict(alias) for alias in ENGINE_ALIASES],
            "ephemeral_transport_fields": list(EPHEMERAL_TRANSPORT_FIELDS),
            "preserve_observed_values": True,
        }
        target["status"] = "active"
        target["intended_use"] = "public-leaderboard"
        compatibility = dict(target.get("compatibility_policy") or {})
        compatibility["evidence_backed_projection"] = binding_policy
        target["compatibility_policy"] = compatibility
        operations.append(
            {
                "target_id": target_id,
                "assert_before": {
                    "status": "provisional",
                    "intended_use": "specialty",
                    "profile": "dense-scaling",
                    "source_spec_sha256": cell["spec_sha256"],
                },
                "set": {
                    "status": "active",
                    "intended_use": "public-leaderboard",
                    "compatibility_policy.evidence_backed_projection": binding_policy,
                },
            }
        )
    for target_id, entry in candidate_by_target.items():
        target = projected_by_id[target_id]
        errors = official_target_binding_errors(entry, target)
        if (
            errors
            or target.get("status") != "active"
            or target.get("intended_use") != "public-leaderboard"
        ):
            raise PromotionError(
                f"projected target does not bind its candidate: {target_id}: {errors}"
            )
    patch = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "review-required",
        "source_registry": {
            "version": registry.version,
            "sha256": registry.sha256,
        },
        "evidence_bundle": {
            "sha256": bundle_sha,
            "archives": bundle["archives"],
            "candidate_count": EXPECTED_CANDIDATES,
        },
        "policy": {
            "raw_or_resolved_evidence_mutated": False,
            "engine_alias_requires_per_target_authorization": True,
            "ephemeral_transport_excluded_from_identity": list(
                EPHEMERAL_TRANSPORT_FIELDS
            ),
            "observed_transport_retained_for_audit": True,
            "requires_registry_version_and_history_update_before_commit": True,
            "does_not_modify_public_snapshots": True,
        },
        "operations": operations,
    }
    return patch, projected_payload


def write_promotion_projection(
    bundle_root: Path, registry: OfficialTargetRegistry, output_dir: Path
) -> Path:
    patch, projected = project_registry_promotion(bundle_root, registry)
    destination = output_dir.absolute()
    if destination.exists():
        raise PromotionError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        for name, payload in (
            ("registry-promotion.patch.json", patch),
            ("projected-official-targets.json", projected),
        ):
            (stage / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        files = sorted(stage.iterdir())
        (stage / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files),
            encoding="utf-8",
        )
        os.replace(stage, destination)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination
