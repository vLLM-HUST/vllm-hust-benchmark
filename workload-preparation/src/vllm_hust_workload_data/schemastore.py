"""Original schema assets and native fixture references, never invented prompts."""

from copy import deepcopy
import hashlib
from pathlib import PurePosixPath
from zipfile import ZipFile

from .records import WorkloadRecord
from .assets import _json, asset_path, verify_file


def original_schema_rows(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        roots = {name.split("/")[0] for name in names}
        if len(roots) != 1 or "" in roots or len(names) != len(set(names)):
            raise ValueError("one original archive root and distinct members required")
        prefix = next(iter(roots)) + "/"
        members = {}
        for item in archive.infolist():
            relative = item.filename.removeprefix(prefix)
            candidate = PurePosixPath(relative)
            if (
                candidate.is_absolute()
                or ".." in candidate.parts
                or ((item.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("contained regular original archive members required")
            if not item.is_dir():
                members[relative] = item
        schemas = sorted(
            name
            for name in members
            if name.startswith("src/schemas/json/") and name.endswith(".json")
        )
        if not schemas:
            raise ValueError("original schema collection missing")
        for name in schemas:
            raw = archive.read(members[name])
            schema = _json(raw.decode("utf-8-sig"))
            if not isinstance(schema, (dict, bool)):
                raise ValueError("original JSON schema must be object or boolean")
            schema_id = PurePosixPath(name).name.replace(".json", "", 1)
            examples = {}
            for key, directory in [
                ("positive_examples", "test"),
                ("negative_examples", "negative_test"),
            ]:
                parent = f"src/{directory}/{schema_id}/"
                references = []
                for relative in sorted(
                    item for item in members if item.startswith(parent)
                ):
                    data = archive.read(members[relative])
                    references.append(
                        {
                            "archive_member": members[relative].filename,
                            "original_path": relative,
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                examples[key] = references
            yield {"schema_path": name, "schema": schema, **examples}


def adapt_schema(row, *, source, index):
    if not isinstance(row.get("schema_path"), str) or not isinstance(
        row.get("schema"), (dict, bool)
    ):
        raise ValueError("original schema asset required")
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["schema_path"],
            "identity_kind": "original_schema_path",
            "native_unit": "json_schema_asset",
            "input": {"schema": deepcopy(row["schema"])},
        },
        runtime={
            "kind": "task_dataset",
            "execution": "not_executed",
            "asset_role": "schema_collection_not_model_request",
            "requirements": [
                "Original SchemaStore cli.js check and original positive/negative fixtures; no fabricated prompts or model quality score"
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "schemastore_original_archive",
            "source_row_index": index,
            "order": "original_schema_path_lexicographic_not_request_arrival",
            "annotations": {"schema_path": row["schema_path"]},
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            key: deepcopy(row[key])
            for key in ["positive_examples", "negative_examples"]
        },
    )


def read_original_fixture(data_root, source, reference):
    """Return unchanged JSON/YAML/TOML fixture bytes; never reinterpret as a prompt."""
    if source["source_id"] != "SCHEMASTORE":
        raise ValueError("original SchemaStore source binding required")
    spec = {"path": source["file"], "sha256": source["file_sha256"]}
    verify_file(data_root, spec)
    with ZipFile(asset_path(data_root, source["file"])) as archive:
        data = archive.read(reference["archive_member"])
    if (
        len(data) != reference["bytes"]
        or hashlib.sha256(data).hexdigest() != reference["sha256"]
    ):
        raise ValueError("original validation fixture identity changed")
    verify_file(data_root, spec)
    return data
