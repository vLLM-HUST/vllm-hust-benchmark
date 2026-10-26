"""Portable metadata resources. No network, data discovery, or runtime imports."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


def read_catalog(name: str, root: Path | None = None):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("catalog must be a JSON basename")
    if root is not None:
        return json.loads(
            (Path(root) / "data" / "catalogs" / name).read_text(encoding="utf-8")
        )
    resource = files("vllm_hust_workload_data").joinpath("resources", name)
    if resource.is_file():
        return json.loads(resource.read_text(encoding="utf-8"))
    # Editable/source checkout; built wheels contain the same versioned resources.
    checkout = Path(__file__).resolve().parents[2]
    return json.loads(
        (checkout / "data" / "catalogs" / name).read_text(encoding="utf-8")
    )


def source_spec(source_id: str, root: Path | None = None) -> dict:
    specs = read_catalog("source_manifests_v1.json", root)["sources"]
    if source_id not in specs:
        raise KeyError(f"unknown source: {source_id}; no generator fallback")
    source = specs[source_id]
    if source.get("files_resource") and not source["files"]:
        manifest = read_catalog(source["files_resource"], root)
        if (
            manifest.get("schema_version") != "source-file-manifest/v1"
            or manifest.get("source_id") != source_id
            or manifest.get("revision") != source.get("revision")
            or manifest.get("repo_id") != source.get("repo_id")
            or not isinstance(manifest.get("files"), list)
            or not manifest["files"]
            or len({item["path"] for item in manifest["files"]})
            != len(manifest["files"])
        ):
            raise ValueError("source file resource identity or file set differs")
        return {**source, "files": manifest["files"]}
    return source


def adapter_spec(source_id: str, root: Path | None = None) -> dict:
    specs = all_adapters(root)
    if source_id not in specs:
        raise NotImplementedError(f"source adapter not implemented: {source_id}")
    return specs[source_id]


def all_adapters(root=None):
    specs = read_catalog("adapter_registry_v1.json", root)["adapters"]
    try:
        extension = read_catalog("adapter_extensions_v1.json", root)["adapters"]
    except FileNotFoundError:
        extension = {}
    if set(specs) & set(extension):
        raise ValueError("duplicate adapter definition across explicit registries")
    return {**specs, **extension}
