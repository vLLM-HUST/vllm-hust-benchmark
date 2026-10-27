"""Bind existing publisher-generated RULER JSONL without copying or generating data.

Generation metadata is explicit caller provenance, not independent attestation.
Native execution and scoring remain the job of run_ruler_original_profile.py.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.assets import _json, asset_path, iter_rows, sha256
from vllm_hust_workload_data.registry import source_spec


TASKS = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
}
GENERATION_FIELDS = (
    "task",
    "max_seq_length",
    "random_seed",
    "model_template_type",
    "tokenizer_type",
    "tokenizer_model",
    "tokenizer_revision",
    "subset",
    "num_samples",
)


def bind(data_root, relative, generation):
    manifest = deepcopy(source_spec("RULER"))
    if not isinstance(generation, dict) or any(
        key not in generation for key in GENERATION_FIELDS
    ):
        raise ValueError("complete original generator provenance required")
    for key in (
        "task",
        "subset",
        "tokenizer_type",
        "tokenizer_model",
        "tokenizer_revision",
        "model_template_type",
    ):
        if not isinstance(generation[key], str) or not generation[key].strip():
            raise ValueError("generation text fields must be nonempty strings")
    if (
        generation["task"] not in TASKS
        or generation["subset"] not in {"validation", "test"}
        or generation["tokenizer_type"] not in {"hf", "nemo", "openai"}
        or generation.get("source_commit", manifest["revision"]) != manifest["revision"]
    ):
        raise ValueError("profile must use the declared pinned publisher protocol")
    path = asset_path(Path(data_root), relative)
    digest = sha256(path)
    count = sum(1 for _ in iter_rows(path, row_format="jsonl"))
    if type(generation["num_samples"]) is not int or generation["num_samples"] != count:
        raise ValueError("original generator num_samples must match complete file rows")
    known_identity = False
    for existing in manifest["files"]:
        if existing.get("role") == "data" and existing["sha256"] == digest:
            if any(
                existing["generation"].get(key) != generation[key]
                for key in GENERATION_FIELDS
            ):
                raise ValueError(
                    "generation declaration conflicts with recorded identical file"
                )
            known_identity = True
    declared = {key: deepcopy(generation[key]) for key in GENERATION_FIELDS}
    declared["origin"] = (
        "caller-declared unchanged publisher generator profile; binding does not reexecute generator"
    )
    compat = generation.get("generator_compatibility")
    if compat is not None:
        required = {
            "patch_id",
            "target_relative_path",
            "original_source_sha256",
            "runtime_source_sha256",
            "explicit_opt_in",
            "question_renderer_ast_preserved",
        }
        if (
            not isinstance(compat, dict)
            or not required <= compat.keys()
            or compat["patch_id"] != "ruler-qa-required-doc-floor/v1"
            or compat["target_relative_path"] != "scripts/data/synthetic/qa.py"
            or compat["explicit_opt_in"] is not True
            or compat["question_renderer_ast_preserved"] is not True
            or any(
                not isinstance(compat[key], str)
                or re.fullmatch(r"[0-9a-f]{64}", compat[key]) is None
                for key in ("original_source_sha256", "runtime_source_sha256")
            )
        ):
            raise ValueError("reviewed generator compatibility metadata required")
        declared["generator_compatibility"] = {
            key: deepcopy(compat[key]) for key in sorted(required)
        }
    manifest["files"] = [
        {
            "path": relative,
            "role": "data",
            "sha256": digest,
            "bytes": path.stat().st_size,
            "rows": count,
            "generation": declared,
            "identity_assurance": "matches_recorded_generated_profile"
            if known_identity
            else "caller_selected_local_generated_snapshot_not_publisher_file_attestation",
        }
    ]
    manifest.update(
        binding_version="ruler-generated-profile-binding/v1",
        raw_data_copied=False,
        generation_independently_reexecuted_by_binding=False,
        native_execution_or_scoring_verified_by_binding=False,
    )
    # Full reader validation checks fields, provenance, counts and digest. No task is run.
    validated_count = 0
    for record in iter_workload("RULER", data_root=Path(data_root), manifest=manifest):
        if record.replay["unmapped_field_names"]:
            raise ValueError(
                "undeclared original source fields; update matching data component contract before binding"
            )
        validated_count += 1
    if validated_count != count:
        raise ValueError("full generated reader count mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--file", required=True, help="relative original JSONL path")
    parser.add_argument(
        "--generation",
        type=Path,
        required=True,
        help="original generation metadata JSON, not raw data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="fresh manifest file; parent must exist",
    )
    args = parser.parse_args()
    manifest = bind(
        args.data_root, args.file, _json(args.generation.read_text(encoding="utf-8"))
    )
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
