"""Pin user-provided files without downloading or modifying repository catalogs."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from .assets import asset_path, sha256
from .adapters import iter_source_rows
from .registry import adapter_spec, source_spec


def bind_source(
    source_id: str,
    *,
    data_root: Path,
    files: list[str],
    revision: str,
    catalog_root=None,
    mapping=None,
):
    if mapping is not None:
        from .mapping_contract import mapping_spec

        spec = mapping_spec(mapping)
        if (
            mapping["source"]["source_id"] != source_id
            or mapping["source"]["revision"] != revision
        ):
            raise ValueError("source/revision must match mapping")
        if mapping["evidence"]["review_status"] != "field_contract_reviewed":
            raise ValueError("review publisher field semantics before binding")
        upstream = {**mapping["source"], "files": []}
        try:
            recorded = source_spec(source_id, catalog_root)
        except KeyError:
            recorded = None
        if recorded and recorded.get("revision") == revision:
            upstream["files"] = recorded["files"]
    else:
        spec = adapter_spec(source_id, catalog_root)
        upstream = source_spec(source_id, catalog_root)
    if spec["kind"] in {"external_runtime_reference", "official_task_suite"}:
        raise ValueError(
            "official task suites require their native runner; no replacement prompt manifest"
        )
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or not files
        or len(set(files)) != len(files)
    ):
        raise ValueError("explicit revision and distinct file selection required")
    known = (
        {f["path"]: f for f in upstream["files"]}
        if revision == upstream.get("revision")
        else {}
    )
    result = []
    for relative in files:
        path = asset_path(data_root, relative)
        digest = sha256(path)
        if relative in known and known[relative]["sha256"] != digest:
            raise ValueError(
                "file differs from pinned publisher/previous identity; do not overwrite its revision"
            )
        if (
            source_id in {"REPOBENCH", "AZURE_LMM_2025", "COPILOT_AGENT_2026", "DOCVQA"}
            and relative not in known
        ):
            raise ValueError("source requires registered original assets")
        if source_id == "RULER" and (
            relative not in known or not known[relative].get("generation")
        ):
            raise ValueError(
                "new RULER profile requires an explicit manifest with actual generator provenance; do not infer it"
            )
        count = 0
        for row in iter_source_rows(path, spec):
            if mapping is not None:
                from .adapters import adapt_row

                adapt_row(
                    source_id,
                    row,
                    source={**upstream, "file": relative, "file_sha256": digest},
                    spec=spec,
                    index=count,
                )
            if set(spec["required_fields"]) - row.keys():
                raise ValueError(
                    "source file does not match the reviewed field contract"
                )
            count += 1
        if sha256(path) != digest:
            raise ValueError("source file changed during binding")
        result.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "rows": count,
                "role": "data",
                **(
                    {"generation": deepcopy(known[relative]["generation"])}
                    if source_id == "RULER"
                    else {}
                ),
                **(
                    {
                        key: deepcopy(known[relative][key])
                        for key in ("language", "dataset_repo", "dataset_revision")
                    }
                    if source_id == "REPOBENCH"
                    else {}
                ),
                "identity_assurance": "matches_recorded_identity"
                if relative in known
                else "user_selected_local_snapshot_not_publisher_authenticated",
            }
        )
    return {
        **upstream,
        "revision": revision,
        "files": result,
        "binding_version": "source-binding/v1",
        "upstream_revision_verified": mapping is None
        and revision == upstream.get("revision")
        and all(
            file["identity_assurance"] == "matches_recorded_identity" for file in result
        ),
        "adapter_contract_status": spec["status"],
        "raw_data_copied": False,
        **({"mapping_sha256": spec["contract_sha256"]} if mapping is not None else {}),
    }
