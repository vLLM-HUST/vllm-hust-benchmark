"""Explicit source adapters. No model calls, implicit acquisition, or synthetic rows."""

from __future__ import annotations

from copy import deepcopy
from itertools import islice
from pathlib import Path

from .assets import _json, asset_path, iter_rows, verify_file
from .records import WorkloadRecord
from .registry import adapter_spec, source_spec


def iter_source_rows(path, spec):
    """Use one source view for reading, binding and index resolution."""
    if spec["mapping"] == "schemastore_original_archive":
        from .schemastore import original_schema_rows

        yield from original_schema_rows(path)
        return
    if spec.get("original_relation_source") is not None:
        from .narrativeqa import joined_rows

        if Path(path).name != "qaps.csv":
            raise ValueError("original NarrativeQA view requires qaps.csv")
        yield from joined_rows(Path(path).parent, spec["original_relation_source"])
        return
    if spec["mapping"] == "crosscodeeval_original_archive":
        from .crosscodeeval import original_rows

        yield from original_rows(path)
        return
    if (
        spec["mapping"] == "source-mapping/v1"
        and spec["contract"]["source"]["source_id"] == "FLORES200"
    ):
        from .flores200 import parallel_rows

        yield from parallel_rows(path)
        return
    if spec["mapping"] == "multinews_original_pairs":
        from .multinews import original_rows

        yield from original_rows(path)
        return
    if spec.get("native_format") == "openslr_archives_or_hf_rows" and Path(
        path
    ).name.endswith(".tar.gz"):
        from .librispeech import original_archive_rows

        yield from original_archive_rows(path)
        return
    if spec["mapping"] == "stsb_original_pairs":
        from .stsb import original_rows

        yield from original_rows(path)
        return
    if spec["mapping"] == "beir_original_assets":
        from .beir import original_asset_rows

        yield from original_asset_rows(path)
        return
    if spec["mapping"] == "xlsum_original_summary":
        from .xlsum import original_rows

        yield from original_rows(path)
        return
    if spec["mapping"] == "qasper_original_questions":
        from .qasper import question_rows

        yield from question_rows(path)
        return
    if spec["mapping"] == "whole_json_schema":
        try:
            schema = _json(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError):
            raise ValueError(
                "invalid original JSON schema asset; payload suppressed"
            ) from None
        if not isinstance(schema, (dict, bool)):
            raise ValueError("JSON schema asset must be an object or boolean")
        # A schema is one constraint asset, not a request or a collection of rows.
        # Preserve all keywords, including unknown extensions and nested values.
        yield {"schema": schema}
        return
    rows = iter_rows(
        path,
        records_path=spec.get("records_path", []),
        row_format=spec.get("row_format"),
        root_fields=spec.get("root_fields", []),
    )
    try:
        if spec["mapping"] == "quality_questions":
            from .quality import question_rows

            yield from question_rows(rows)
        elif spec["mapping"] == "locomo_questions":
            from .locomo import question_rows

            yield from question_rows(rows)
        else:
            yield from rows
    finally:
        rows.close()


def adapt_row(
    source_id: str, row: dict, *, source: dict, spec: dict, index: int
) -> WorkloadRecord:
    """Projection is explicit/allowlisted; outcome fields never become task inputs."""
    if source_id == "DOCVQA":
        from .docvqa import verify_question

        verify_question(row)
    if source_id == "COPILOT_AGENT_2026":
        from .copilot import verify_session

        verify_session(row)
        if (
            spec["kind"] != "request_metadata"
            or spec["input_fields"]
            or spec.get("outcome_fields", [])
            or set(spec.get("annotation_fields", [])) != set(row)
            or spec.get("id_field")
        ):
            raise ValueError(
                "Copilot requires unchanged nested metadata and file/row identity, not requests"
            )
    if source_id == "SCHEMASTORE" and spec["mapping"] == "schemastore_original_archive":
        from .schemastore import adapt_schema

        return adapt_schema(row, source=source, index=index)
    if spec["mapping"] == "beir_original_assets":
        from .beir import adapt_asset

        return adapt_asset(row, source=source, index=index)
    if spec["kind"] in {"official_task_suite", "external_runtime_reference"}:
        raise NotImplementedError(
            "official native runner required; no QBI fallback or prompt flattening"
        )
    if spec["mapping"] == "source-mapping/v1":
        from .mapping_contract import adapt_mapping

        return adapt_mapping(row, source=source, spec=spec, index=index)
    missing = set(spec["required_fields"]) - row.keys()
    if missing:
        raise ValueError(
            f"{source_id}: missing required fields at row {index}; payload suppressed"
        )
    nullable = set(spec.get("nullable_fields", [])) | (
        {"parent_id"} if source_id == "OASST1" else set()
    )
    if any(row[key] is None for key in set(spec["required_fields"]) - nullable):
        raise ValueError(
            f"{source_id}: null required field at row {index}; payload suppressed"
        )
    text_fields = {
        "question",
        "instruction",
        "prompt",
        "task_inst",
        "problem_statement",
        "complete_prompt",
        "instruct_prompt",
        "query",
        "text",
    }
    for key in (text_fields - set(spec.get("native_structured_fields", []))) & set(
        spec["required_fields"]
    ):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(
                f"{source_id}: invalid text field at row {index}; payload suppressed"
            )
    if source_id == "QUALITY" and (
        any(
            not isinstance(row[key], str) or not row[key].strip()
            for key in ("article", "question", "question_unique_id")
        )
        or not isinstance(row["options"], list)
        or len(row["options"]) != 4
        or any(
            not isinstance(value, str) or not value.strip() for value in row["options"]
        )
        or type(row["gold_label"]) is not int
        or row["gold_label"] not in {1, 2, 3, 4}
    ):
        raise ValueError(
            "QuALITY requires original article/question/four options and 1-based gold"
        )
    if source_id == "BBH" and any(
        not isinstance(row[key], str) or not row[key].strip()
        for key in ("input", "target")
    ):
        raise ValueError("BBH requires original nonempty input and reference strings")
    if source_id == "RULER" and (
        type(row["index"]) is not int
        or not isinstance(row["input"], str)
        or not row["input"].strip()
        or not isinstance(row.get("answer_prefix", ""), str)
        or not isinstance(row["outputs"], list)
        or not row["outputs"]
        or any(not isinstance(value, str) for value in row["outputs"])
    ):
        raise ValueError(
            "RULER requires original generated input, index and reference list"
        )
    if source_id == "NATURAL_INSTRUCTIONS" and (
        not isinstance(row["Definition"], list)
        or not row["Definition"]
        or any(
            not isinstance(value, str) or not value.strip()
            for value in row["Definition"]
        )
        or not isinstance(row["input"], str)
        or not row["input"].strip()
        or not isinstance(row["output"], list)
        or not row["output"]
        or any(not isinstance(value, str) for value in row["output"])
        or not isinstance(row["id"], str)
        or not row["id"]
    ):
        raise ValueError(
            "Natural Instructions requires original definition/input/reference fields"
        )
    if source_id == "MATH500" and (
        not isinstance(row["unique_id"], str)
        or not row["unique_id"].strip()
        or not isinstance(row["problem"], str)
        or not row["problem"].strip()
        or not isinstance(row["solution"], str)
        or not row["solution"].strip()
        or not isinstance(row["answer"], str)
        or not row["answer"].strip()
        or not isinstance(row["subject"], str)
        or not row["subject"].strip()
        or type(row["level"]) is not int
        or not 1 <= row["level"] <= 5
    ):
        raise ValueError(
            "MATH500 requires original problem, reference, subject, level, and unique ID"
        )
    if source_id == "MMLU_PRO" and (
        not isinstance(row["options"], list) or not row["options"]
    ):
        raise ValueError("MMLU_PRO requires source options")
    if source_id == "LONGBENCH" and (
        row["dataset"] != "qasper"
        or any(
            not isinstance(row[key], str) or not row[key].strip()
            for key in ("context", "input")
        )
        or not isinstance(row["answers"], list)
        or not row["answers"]
        or not all(isinstance(answer, str) for answer in row["answers"])
        or (row["all_classes"] is not None and not isinstance(row["all_classes"], list))
    ):
        raise ValueError("LONGBENCH adapter currently requires original qasper fields")
    if source_id == "AI2_ARC":
        choices = row["choices"]
        if (
            not isinstance(choices, dict)
            or not isinstance(choices.get("text"), list)
            or not isinstance(choices.get("label"), list)
            or not choices["text"]
            or len(choices["text"]) != len(choices["label"])
        ):
            raise ValueError("AI2_ARC requires aligned source choice text and labels")
    if source_id == "MBPP" and (
        set(row)
        != {
            "task_id",
            "text",
            "code",
            "test_list",
            "test_setup_code",
            "challenge_test_list",
        }
        or type(row["task_id"]) is not int
        or row["task_id"] <= 0
        or any(
            not isinstance(row[key], str) for key in ("text", "code", "test_setup_code")
        )
        or any(
            not isinstance(row[key], list)
            or any(not isinstance(test, str) for test in row[key])
            for key in ("test_list", "challenge_test_list")
        )
        or len(row["test_list"]) != 3
    ):
        raise ValueError(
            "MBPP requires unchanged full-configuration fields and three tests"
        )
    if source_id == "BILLSUM" and (
        set(row) != {"text", "summary", "title"}
        or any(not isinstance(row[key], str) for key in row)
    ):
        raise ValueError("BILLSUM requires original text, summary and title strings")
    if source_id == "XSUM" and (
        set(row) != {"document", "summary", "id"}
        or any(not isinstance(row[key], str) for key in row)
        or not row["id"].strip()
    ):
        raise ValueError("XSUM requires original document, summary and ID strings")
    if source_id == "TOOLACE" and (
        not isinstance(row["conversations"], list) or not row["conversations"]
    ):
        raise ValueError("TOOLACE requires its original dialogue")
    if source_id == "TULU3" and (
        set(row) != {"id", "messages", "source"}
        or not isinstance(row["id"], str)
        or not isinstance(row["source"], str)
        or not isinstance(row["messages"], list)
        or not row["messages"]
    ):
        raise ValueError(
            "TULU3 requires original ID, messages and per-row mixture source"
        )
    kind = spec["kind"]
    if source_id == "BFCL":
        questions, functions = row["question"], row["function"]
        if (
            not isinstance(row["id"], str)
            or not row["id"].startswith("simple_python_")
            or not isinstance(questions, list)
            or len(questions) != 1
            or not isinstance(questions[0], list)
            or len(questions[0]) != 1
            or not isinstance(questions[0][0], dict)
            or questions[0][0].get("role") != "user"
            or not isinstance(questions[0][0].get("content"), str)
            or not questions[0][0]["content"].strip()
            or not isinstance(functions, list)
            or len(functions) != 1
            or not isinstance(functions[0], dict)
            or not isinstance(functions[0].get("name"), str)
            or not functions[0]["name"].strip()
            or not isinstance(functions[0].get("parameters"), dict)
        ):
            raise ValueError(
                "BFCL adapter only supports original single-turn simple_python tasks"
            )
    if source_id in {"AZURE_LLM", "AZURE_LMM_2025"}:
        if source_id == "AZURE_LMM_2025" and set(row) != set(spec["required_fields"]):
            raise ValueError(
                "Azure LMM original four-field schema required; re-review extra fields"
            )
        if not isinstance(row["TIMESTAMP"], str) or not row["TIMESTAMP"].strip():
            raise ValueError("Azure metadata requires a source timestamp string")
        count_fields = ["ContextTokens", "GeneratedTokens"]
        if source_id == "AZURE_LMM_2025":
            count_fields.append("NumImages")
        for field in count_fields:
            value = row[field]
            if not (
                type(value) is int
                and value >= 0
                or isinstance(value, str)
                and value.isascii()
                and value.isdecimal()
            ):
                raise ValueError("Azure metadata requires nonnegative integer counts")
    if source_id == "MOONCAKE":
        for field in ("timestamp", "input_length", "output_length"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("MOONCAKE requires nonnegative integer trace fields")
        if not isinstance(row["hash_ids"], list) or any(
            type(v) is not int or v < 0 for v in row["hash_ids"]
        ):
            raise ValueError("MOONCAKE requires an ordered list of block identifiers")
    task_input = {key: deepcopy(row[key]) for key in spec["input_fields"] if key in row}
    outcome = {
        key: deepcopy(row[key]) for key in spec.get("outcome_fields", []) if key in row
    }
    annotations = {
        key: deepcopy(row[key])
        for key in spec.get("annotation_fields", [])
        if key in row
    }
    if kind == "request_replay":
        if spec["mapping"] == "wrapped_request":
            body = row["request_body"]
            if isinstance(body, str):
                try:
                    body = _json(body)
                except ValueError:
                    raise ValueError(
                        f"{source_id}: invalid request body at row {index}"
                    ) from None
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            # Preserve every API field, including response_format, tools and media.
            task_input = {"request_body": deepcopy(body)}
        else:
            body = {
                key: deepcopy(row[key]) for key in spec["input_fields"] if key in row
            }
            if not isinstance(body.get("messages"), list) or not body["messages"]:
                raise ValueError("enterprise request needs its original messages")
            task_input = {"request_body": body}
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise ValueError("chat request must retain original messages")
        for message in body["messages"]:
            if not isinstance(message, dict) or not isinstance(
                message.get("role"), str
            ):
                raise ValueError("invalid original chat message")
    if spec["mapping"] == "supervised_messages":
        messages = row["messages"]
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(m, dict) for m in messages)
        ):
            raise ValueError("expected source messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError("expected final assistant reference")
        task_input = {"messages": deepcopy(messages[:-1])}
        if not task_input["messages"]:
            raise ValueError("reference has no task input")
        outcome = {"reference_message": deepcopy(messages[-1])}
    if spec["mapping"] == "gpqa_original_seed0_choices":
        from .gpqa import seed_zero_choices

        choices, correct_index = seed_zero_choices(row, index)
        task_input["choices"] = choices
        outcome["correct_index"] = correct_index
    if spec["mapping"] == "longmemeval_history":
        sessions = row["haystack_sessions"]
        if (
            not isinstance(sessions, list)
            or len(sessions) != len(row["haystack_dates"])
            or len(sessions) != len(row["haystack_session_ids"])
        ):
            raise ValueError("unaligned LongMemEval history/session/date arrays")
        clean_sessions, labels = [], []
        for si, session in enumerate(sessions):
            if not isinstance(session, list):
                raise ValueError("LongMemEval history session must contain turns")
            clean = []
            for ti, turn in enumerate(session):
                if (
                    not isinstance(turn, dict)
                    or not isinstance(turn.get("content"), str)
                    or "role" not in turn
                ):
                    raise ValueError("invalid LongMemEval history turn")
                clean.append(
                    {k: deepcopy(v) for k, v in turn.items() if k != "has_answer"}
                )
                if "has_answer" in turn:
                    labels.append(
                        {
                            "session_index": si,
                            "turn_index": ti,
                            "has_answer": turn["has_answer"],
                        }
                    )
            clean_sessions.append(clean)
        task_input["haystack_sessions"] = clean_sessions
        outcome["turn_evidence_labels"] = labels
    if source_id == "CHARTQA":
        media = row["image"]
        if not isinstance(media, dict) or not (media.get("bytes") or media.get("path")):
            raise ValueError(
                "ChartQA requires real image bytes or a resolvable source media reference"
            )
    identity = row.get(spec.get("id_field", ""))
    task_id = (
        str(identity) if identity is not None else f"{source['file_sha256']}:{index}"
    )
    # Preserve non-API source text and identifiers separately, not as invented API parameters.
    if source_id.startswith("PAIO-") and spec["mapping"] == "enterprise_messages":
        for key in ("prompt", "query_text"):
            if key in row:
                annotations[key] = deepcopy(row[key])
        if source_id == "PAIO-LONGTEXT-4000":
            # These are explicit source fields for input-side document analysis.
            # Keep them separate from the unchanged API request body: 781 released
            # rows have a nonempty query_text that is not embedded in messages.
            task_input["feature_inputs"] = {
                "query": deepcopy(row["query_text"]),
                "documents": [{"text": deepcopy(row["prompt"])}],
            }
    unknown = sorted(
        set(row)
        - set(spec["input_fields"])
        - set(spec.get("outcome_fields", []))
        - set(spec.get("annotation_fields", []))
    )
    if source_id.startswith("PAIO-"):
        unknown = [key for key in unknown if key not in annotations]
    task = {
        "task_id": task_id,
        "identity_kind": "source_id"
        if identity is not None
        else "file_sha256_row_index",
        "input": task_input,
    }
    replay = {
        "adapter_version": "source-adapter/v1",
        "mapping": spec["mapping"],
        "source_row_index": index,
        "order": "source_file_order_not_assumed_arrival_order",
        "annotations": annotations,
        "unmapped_field_names": unknown,
        "synthetic_fallback": False,
        "normalization": spec.get("normalization", {}),
    }
    if source_id == "OASST1":
        task["parent_task_id"] = row.get("parent_id")
        task["session_id"] = row.get("message_tree_id")
        replay["order"] = (
            "message_tree_edges; source_file_order_is_not_conversation_order"
        )
    if (
        source_id == "LIBRISPEECH"
        and isinstance(row.get("audio"), dict)
        and "archive_member" in row["audio"]
    ):
        replay["order"] = (
            "original_sorted_utterance_ids_within_archive_not_request_arrival"
        )
    return WorkloadRecord(
        source=deepcopy(source),
        task=task,
        runtime={
            "kind": kind,
            "execution": "not_executed",
            "requirements": spec["requirements"],
        },
        replay=replay,
        outcome=outcome or None,
    )


def iter_workload(
    source_id: str,
    *,
    data_root: Path,
    files: list[str] | None = None,
    limit: int | None = None,
    catalog_root: Path | None = None,
    manifest: dict | None = None,
    mapping: dict | None = None,
    history_targets: list[str] | None = None,
):
    """Read explicit pinned assets. limit is a diagnostic slice, never full coverage."""
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or omitted")
    from .conversations import history_record, validate_history_targets

    validate_history_targets(source_id, history_targets, mapping)
    if mapping is not None:
        from .mapping_contract import mapping_spec

        spec = mapping_spec(mapping)
        if (
            manifest is None
            or manifest.get("mapping_sha256") != spec["contract_sha256"]
        ):
            raise ValueError("bind files with this exact mapping before reading")
    else:
        spec = adapter_spec(source_id, catalog_root)
    if spec["kind"] in {"external_runtime_reference", "official_task_suite"}:
        raise NotImplementedError(
            "official native runner required; no QBI fallback or prompt flattening"
        )
    manifest = source_spec(source_id, catalog_root) if manifest is None else manifest
    if manifest.get("source_id", source_id) != source_id or not manifest.get(
        "revision"
    ):
        raise ValueError("explicit manifest requires matching source ID and revision")
    if source_id == "NARRATIVEQA" and mapping is None:
        from .narrativeqa import relation_spec

        spec = relation_spec(manifest, spec)
    available = {
        file["path"]: file
        for file in manifest["files"]
        if (
            file["path"] == "qaps.csv"
            if spec.get("original_relation_source") is not None
            else file.get("role") == "data"
        )
    }
    selected = list(available) if files is None else files
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in available for name in selected)
    ):
        raise ValueError(
            "select nonempty, distinct registered data files for this source revision"
        )
    emitted = 0
    if history_targets is not None and len(selected) != 1:
        raise ValueError(
            "historical continuation requires one complete original tree file"
        )
    original_repo_files = None
    if source_id == "REPOBENCH":
        original_repo = source_spec(source_id, catalog_root)
        if manifest["revision"] != original_repo["revision"]:
            raise ValueError("RepoBench author protocol revision mismatch")
        original_repo_files = {item["path"]: item for item in original_repo["files"]}
    for name in selected:
        file_spec = available[name]
        receipt = verify_file(Path(data_root), file_spec)
        source = {
            "source_id": source_id,
            "revision": manifest["revision"],
            "file": name,
            "file_sha256": receipt["sha256"],
            "unit": manifest["unit"],
            "provenance_kind": manifest["provenance_kind"],
        }
        if source_id == "REPOBENCH":
            expected = original_repo_files.get(name, {})
            keys = ("language", "dataset_repo", "dataset_revision")
            if (
                any(
                    not isinstance(file_spec.get(key), str)
                    or file_spec[key] != expected.get(key)
                    for key in keys
                )
                or expected.get("sha256") != file_spec["sha256"]
                or name.split("/")[0] != file_spec["language"]
            ):
                raise ValueError("RepoBench language/dataset provenance mismatch")
            source.update({key: file_spec[key] for key in keys})
        if source_id == "PAWS":
            configuration = file_spec.get("configuration", "unspecified")
            reference_kind = file_spec.get("reference_kind", "unspecified")
            expected_kind = {
                "labeled_final": "human",
                "labeled_swap": "human",
                "unlabeled_final": "noisy_auxiliary",
            }
            if configuration != "unspecified" and (
                configuration not in expected_kind
                or reference_kind != expected_kind[configuration]
                or name.split("/")[0] != configuration
            ):
                raise ValueError(
                    "PAWS original configuration/reference provenance mismatch"
                )
            source["dataset_configuration"] = configuration
            source["label_reference_kind"] = reference_kind
        if source_id == "XLSUM":
            language = file_spec.get("language_variant")
            if not isinstance(language, str) or not language:
                raise ValueError("XL-Sum original language variant must be preserved")
            source["language_variant"] = language
        if source_id == "RULER":
            generation = file_spec.get("generation")
            generation_text_fields = (
                "task",
                "model_template_type",
                "tokenizer_type",
                "tokenizer_model",
                "tokenizer_revision",
                "subset",
            )
            if (
                not isinstance(generation, dict)
                or any(
                    not isinstance(generation.get(key), str)
                    or not generation[key].strip()
                    for key in generation_text_fields
                )
                or any(
                    type(generation.get(key)) is not int or generation[key] <= 0
                    for key in ("max_seq_length", "num_samples")
                )
                or type(generation.get("random_seed")) is not int
                or generation["random_seed"] < 0
                or generation["num_samples"] != file_spec.get("rows")
            ):
                raise ValueError("RULER manifest must preserve generation provenance")
            source["generation"] = deepcopy(generation)
        rows = iter_source_rows(asset_path(Path(data_root), name), spec)
        remaining = None if limit is None else limit - emitted
        try:
            if history_targets is not None:
                original_rows = list(rows)
                if (
                    file_spec.get("rows") is not None
                    and len(original_rows) != file_spec["rows"]
                ):
                    raise ValueError("source row count mismatch")
                indices = {
                    row["message_id"]: index for index, row in enumerate(original_rows)
                }
                if any(target not in indices for target in history_targets):
                    raise ValueError("selected original history target missing")
                for target in history_targets[:remaining]:
                    record = history_record(
                        original_rows, target, source=source, index=indices[target]
                    )
                    record.replay["selection"] = {
                        "files": list(selected),
                        "limit": limit,
                        "history_targets": history_targets,
                    }
                    yield record
                verify_file(Path(data_root), file_spec)
                return
            count = 0
            for index, row in enumerate(islice(rows, remaining)):
                record = adapt_row(
                    source_id, row, source=source, spec=spec, index=index
                )
                record.replay["selection"] = {"files": list(selected), "limit": limit}
                yield record
                count += 1
                emitted += 1
            if (
                remaining is None
                and file_spec.get("rows") is not None
                and count != file_spec["rows"]
            ):
                raise ValueError(f"source row count mismatch: {source_id}/{name}")
        finally:
            rows.close()
        verify_file(Path(data_root), file_spec)
        if limit is not None and emitted >= limit:
            return
