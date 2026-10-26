"""Reference-only source preparation. No observation, features or research analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .adapters import iter_workload
from .assets import sha256
from .bindings import bind_source
from .identity import canonical, key
from .mapping_contract import example_mapping, load_mapping, mapping_spec
from .registry import adapter_spec, source_spec

SCHEMA = "workload-source-index/v1"
OPTIONS = {
    "source_id",
    "data_root",
    "manifest",
    "mapping",
    "files",
    "limit",
    "history_targets",
}
SQL = """
CREATE TABLE sources(source_id TEXT PRIMARY KEY,source_json TEXT NOT NULL,adapter_json TEXT NOT NULL);
CREATE TABLE records(record_id TEXT PRIMARY KEY,source_id TEXT NOT NULL REFERENCES sources(source_id),
 source_json TEXT NOT NULL,task_json TEXT NOT NULL,source_unit TEXT NOT NULL,runtime_kind TEXT NOT NULL);
"""


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    for entry in config.get("sources", []):
        for name in ("data_root", "manifest", "mapping"):
            if entry.get(name):
                entry[name] = str((path.parent / entry[name]).resolve())
    return config


def _binding(entry, catalog_root=None):
    sid = entry["source_id"]
    mapping = load_mapping(entry["mapping"]) if entry.get("mapping") else None
    manifest = (
        json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
        if entry.get("manifest")
        else source_spec(sid, catalog_root)
    )
    spec = mapping_spec(mapping) if mapping else adapter_spec(sid, catalog_root)
    if sid == "NARRATIVEQA" and mapping is None:
        from .narrativeqa import relation_spec

        spec = relation_spec(manifest, spec)
    from .conversations import validate_history_targets

    validate_history_targets(sid, entry.get("history_targets"), mapping)
    if entry.get("history_targets") is not None:
        spec = {**spec, "history_targets": entry["history_targets"]}
    if mapping and mapping["source"]["source_id"] != sid:
        raise ValueError("mapping source identity mismatch")
    return manifest, spec, mapping


def plan(config, *, catalog_root=None):
    if (
        not isinstance(config, dict)
        or set(config) != {"sources"}
        or not isinstance(config["sources"], list)
        or not config["sources"]
    ):
        raise ValueError("config requires a nonempty sources array only")
    seen, rows = set(), []
    for entry in config["sources"]:
        if (
            not isinstance(entry, dict)
            or set(entry) - OPTIONS
            or not isinstance(entry.get("source_id"), str)
        ):
            raise ValueError(
                "invalid source binding; QBI roots, history and research options are not supported"
            )
        sid = entry["source_id"]
        if sid in seen:
            raise ValueError("duplicate source binding")
        seen.add(sid)
        if entry.get("limit") is not None and (
            type(entry["limit"]) is not int or entry["limit"] <= 0
        ):
            raise ValueError("limit must be a positive integer")
        reason = None
        try:
            manifest, spec, mapping = _binding(entry, catalog_root)
            if spec["kind"] in {"official_task_suite", "external_runtime_reference"}:
                reason = "OFFICIAL_RUNNER_ADAPTER_REQUIRED_NO_QBI_FALLBACK"
            elif not entry.get("data_root") or not Path(entry["data_root"]).is_dir():
                reason = "DATA_ROOT_NOT_AVAILABLE"
            elif not spec.get("original_relation_source") and not any(
                f.get("role") == "data" for f in manifest["files"]
            ):
                reason = "LOCAL_FILE_BINDING_REQUIRED"
            elif (
                mapping
                and mapping["evidence"]["review_status"] != "field_contract_reviewed"
            ):
                reason = "MAPPING_FIELD_REVIEW_REQUIRED"
        except NotImplementedError:
            reason = "ADAPTER_NOT_IMPLEMENTED"
        except FileNotFoundError:
            reason = "BINDING_FILE_NOT_AVAILABLE"
        rows.append(
            {
                "source_id": sid,
                "ready_to_attempt": reason is None,
                "blocking_reason": reason,
                "runtime_validation": "NOT_RUN",
            }
        )
    return {"schema_version": SCHEMA, "sources": rows, "network_calls": 0}


def run(config, *, output, allow_partial=False, catalog_root=None):
    preview = plan(config, catalog_root=catalog_root)
    if not allow_partial and any(not r["ready_to_attempt"] for r in preview["sources"]):
        raise ValueError("selected source is unavailable; inspect plan, no fallback")
    output = Path(output).resolve()
    for entry in config["sources"]:
        if entry.get("data_root"):
            root = Path(entry["data_root"]).resolve()
            if (
                root == output
                or root.is_relative_to(output)
                or output.is_relative_to(root)
            ):
                raise ValueError("output must be separate from every source root")
    output.mkdir(parents=True, exist_ok=False)
    database = output / "workload.sqlite.partial"
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SQL)
    reports = []
    try:
        for entry, readiness in zip(config["sources"], preview["sources"]):
            sid = entry["source_id"]
            if not readiness["ready_to_attempt"]:
                reports.append({**readiness, "status": "NOT_PROCESSED", "records": 0})
                continue
            connection.execute("SAVEPOINT source_batch")
            try:
                manifest, spec, mapping = _binding(entry, catalog_root)
                connection.execute(
                    "INSERT INTO sources VALUES(?,?,?)",
                    (sid, canonical(manifest), canonical(spec)),
                )
                count = 0
                records = iter_workload(
                    sid,
                    data_root=Path(entry["data_root"]),
                    files=entry.get("files"),
                    limit=entry.get("limit"),
                    manifest=manifest,
                    mapping=mapping,
                    catalog_root=catalog_root,
                    history_targets=entry.get("history_targets"),
                )
                try:
                    for record in records:
                        reference = {
                            "file": record.source["file"],
                            "sha256": record.source["file_sha256"],
                            "row_index": record.replay["source_row_index"],
                        }
                        if sid == "FLORES200" and mapping:
                            # One original parallel row has distinct directed
                            # task views. Do not collide their record identities.
                            reference["mapping_sha256"] = spec["contract_sha256"]
                        if entry.get("history_targets") is not None:
                            reference["view"] = "original-parent-history/v1"
                        task = {
                            k: record.task[k]
                            for k in (
                                "task_id",
                                "parent_task_id",
                                "session_id",
                                "request_id",
                            )
                            if k in record.task
                        }
                        task["input_ref"] = reference
                        task["selection"] = record.replay.get("selection")
                        record_id = key(sid, record.source["revision"], reference)
                        connection.execute(
                            "INSERT INTO records VALUES(?,?,?,?,?,?)",
                            (
                                record_id,
                                sid,
                                canonical(record.source),
                                canonical(task),
                                record.source["unit"],
                                record.runtime["kind"],
                            ),
                        )
                        count += 1
                finally:
                    records.close()
                connection.execute("RELEASE source_batch")
                connection.commit()
                reports.append(
                    {
                        "source_id": sid,
                        "status": "PREPARED_REFERENCES_ONLY",
                        "records": count,
                        "scope": "BOUNDED_SLICE"
                        if entry.get("limit") or entry.get("history_targets")
                        else "SELECTED_FILES_FULLY_READ",
                    }
                )
            except Exception as error:
                connection.execute("ROLLBACK TO source_batch")
                connection.execute("RELEASE source_batch")
                connection.commit()
                reports.append(
                    {
                        "source_id": sid,
                        "status": "FAILED",
                        "records": 0,
                        "error_type": type(error).__name__,
                    }
                )
                if not allow_partial:
                    raise
    finally:
        connection.close()
    database.rename(output / "workload.sqlite")
    report = {
        "schema_version": SCHEMA,
        "status": "PARTIAL"
        if any(r["status"] != "PREPARED_REFERENCES_ONLY" for r in reports)
        else "PREPARED_REFERENCES_ONLY",
        "sources": reports,
        "model_calls": 0,
        "native_tasks_run": 0,
        "scoring": "NOT_RUN",
        "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    (output / "manifest.json").write_text(canonical(report) + "\n", encoding="utf-8")
    return report


def connect_readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def prepared_rows(path, *, source_id=None):
    connection = connect_readonly(path)
    connection.row_factory = sqlite3.Row
    try:
        query = "SELECT record_id,source_id,source_unit,runtime_kind FROM records"
        rows = (
            connection.execute(
                query + " WHERE source_id=? ORDER BY rowid", (source_id,)
            )
            if source_id
            else connection.execute(query + " ORDER BY rowid")
        )
        for row in rows:
            yield dict(row)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name == "run":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--allow-partial", action="store_true")
    command = commands.add_parser("bind")
    command.add_argument("source_id")
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--files", nargs="+", required=True)
    command.add_argument("--revision", required=True)
    command.add_argument("--mapping", type=Path)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("mapping-check")
    command.add_argument("mapping", type=Path)
    command = commands.add_parser("mapping-template")
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "plan":
            value = plan(load_config(args.config))
        elif args.command == "run":
            value = run(
                load_config(args.config),
                output=args.output,
                allow_partial=args.allow_partial,
            )
        elif args.command == "mapping-check":
            load_mapping(args.mapping)
            value = {"status": "MAPPING_VALID_NOT_EXECUTED"}
        else:
            value = (
                example_mapping()
                if args.command == "mapping-template"
                else bind_source(
                    args.source_id,
                    data_root=args.data_root,
                    files=args.files,
                    revision=args.revision,
                    mapping=load_mapping(args.mapping) if args.mapping else None,
                )
            )
            with args.output.open("x", encoding="utf-8") as output:
                output.write(canonical(value) + "\n")
    except Exception as error:
        parser.exit(
            1,
            f"source preparation failed ({type(error).__name__}); no payload logged\n",
        )
    print(json.dumps(value, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
