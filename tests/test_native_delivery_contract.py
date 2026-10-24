"""Keep current delivery checkboxes distinct from historical reader evidence."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_science_verified_task_metadata_and_source_are_not_native_execution():
    receipt = json.loads(
        (DOCS / "validation/scienceagentbench-cpu-preparation-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["dataset_split"] == "verified"
    assert receipt["all_original_rows_fields_and_separation_equal"] == 102
    assert (
        receipt["prepared_original_tasks"]
        == receipt["prepare_resolve_roundtrips"]
        == 102
    )
    assert receipt["original_reference_files_kept_scorer_side"]
    assert receipt["native_source_original_blob_verified"]
    assert len(receipt["native_source_files"]) == 30
    assert receipt["legacy_validation_split_entrypoints"] == [
        "run_infer.py",
        "run_eval.py",
    ]
    assert not receipt["verified_benchmark_archive_acquired"]
    assert not receipt["source_requirements_installed_or_compatible_verified"]
    assert not receipt["sandbox_execution_verified"]
    assert not receipt["native_source_modified"]
    assert receipt["model_calls"] == receipt["native_tasks_executed_or_scored"] == 0
    assert not receipt["generated_or_reference_programs_executed"]
    assert not receipt["source_completion_checked"]


def test_tracelab_original_metadata_is_not_parent_tasks_or_full_index():
    receipt = json.loads(
        (DOCS / "validation/tracelab-cpu-audit-20261024.json").read_text("utf-8")
    )
    assert receipt["original_rows_fields_and_order_equal"] == 665453
    assert receipt["prepared_rows"] == receipt["prepare_resolve_roundtrips"] == 3
    assert not receipt["full_source_index_prepared"]
    assert receipt["all_session_round_timing_tool_metadata_preserved"]
    assert receipt["model_request_rendering_refused"]
    assert receipt["model_calls"] == receipt["native_parent_tasks_executed"] == 0
    assert not receipt["source_row_count_is_parent_task_count"]
    assert not receipt["source_completion_checked"]


def test_tracelab_official_replay_sources_are_not_build_or_runtime_evidence():
    receipt = json.loads(
        (
            DOCS / "validation/tracelab-official-replay-preparation-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["source_files_original_git_blob_verified"]
    assert len(receipt["files"]) == 20
    assert all(
        row["path"].startswith("replay/")
        or row["path"] in {"README.md", "LICENSE", "LICENSE-DATASET.md"}
        for row in receipt["files"]
    )
    assert not receipt["research_analysis_modules_copied"]
    assert receipt["cargo_path"] is receipt["rustc_path"] is None
    assert not receipt["native_build_executed"]
    assert not receipt["native_dry_run_executed"]
    assert not receipt["native_replay_executed"]
    assert not receipt["official_duckdb_runtime_asset"]["acquired_or_hash_verified"]
    assert receipt["official_replay_is_synthetic_session_driver"]
    assert receipt["dry_run_zero_exit_does_not_prove_context_fit"]
    assert (
        receipt["upstream_skip_flag_not_permitted_for_fixed_task_validation"]
        == "--fail-on-context-overflow"
    )


def test_toolace_full_dialogue_preparation_is_not_agent_execution():
    receipt = json.loads(
        (DOCS / "validation/toolace-corpus-cpu-audit-20261024.json").read_text("utf-8")
    )
    assert receipt["all_original_rows_fields_and_dialogue_order_equal"] == 11300
    assert receipt["prepared_original_rows"] == 11300
    assert receipt["prepare_resolve_roundtrips"] == 6
    assert receipt["system_tool_descriptions_preserved"]
    assert receipt["original_request_body_conversion_refused"]
    assert receipt["model_calls"] == receipt["native_agent_tasks_executed"] == 0
    assert not receipt["native_task_scorer_executed"]
    assert not receipt["publisher_card_download_verified"]
    assert not receipt["source_completion_checked"]


def test_schema_cpu_audit_preserves_failures_and_registered_scope():
    receipt = json.loads(
        (DOCS / "validation/jsonschemabench-cpu-audit-20261024.json").read_text("utf-8")
    )
    assert (
        receipt["prepared_original_assets"] == receipt["saved_native_tasks_reused"] == 3
    )
    assert receipt["prepare_resolve_roundtrips"] == 3
    assert receipt["audit_model_calls"] == receipt["audit_tokenizer_api_calls"] == 0
    assert [row["original_validator_valid"] for row in receipt["checks"]] == [
        True,
        False,
        False,
    ]
    assert receipt["original_conformance_failures_preserved"] == 2
    assert not receipt["schema_or_prompt_rewritten"]
    assert not receipt["full_schema_collection_validated"]


def test_vision_weight_headers_do_not_count_as_multimodal_execution():
    receipt = json.loads(
        (
            DOCS / "validation/multimodal-existing-model-preflight-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["vision_config_present"]
    assert (
        receipt["indexed_vision_tensor_count"]
        == receipt["actual_vision_safetensors_header_entries_matched"]
        == 333
    )
    assert receipt["current_model_service_language_model_only"]
    assert not receipt["visual_inference_validated"]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["new_weights_downloaded"]
    assert not receipt["model_service_stopped_or_changed"]
    assert receipt["model_generation_calls"] == 0


def test_code_generation_is_not_functional_execution_without_isolation():
    receipt = json.loads(
        (
            DOCS / "validation/code-execution-isolation-preflight-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["network_user_namespace_noop"]["returncode"] == 0
    assert receipt["mount_network_pid_namespace_noop"]["returncode"] != 0
    assert receipt["private_bubblewrap_trusted_noop"]["returncode"] != 0
    assert not receipt["generated_code_executed"]
    assert not receipt["functional_scores_computed"]
    assert not receipt["source_checkbox_eligible"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    for sid in receipt["affected_sources"]:
        assert not ledger["delivery_inventory"][sid]["checked"]
        assert ledger["delivery_inventory"][sid]["isolation_preflight"]


def test_original_image_processing_is_not_native_multimodal_execution():
    receipt = json.loads(
        (DOCS / "validation/chartqa-original-processor-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["status"] == "CPU_ORIGINAL_IMAGES_PROCESSOR_PASSED_NOT_INFERENCE"
    assert receipt["source_indices"] == [0, 1, 2]
    assert receipt["native_tasks_executed"] == receipt["model_calls"] == 0
    assert not receipt["weights_loaded"]
    assert not receipt["model_service_changed"]
    assert not receipt["full_benchmark_completed"]
    for task in receipt["tasks"]:
        assert task["original_question_used"]
        assert task["tensor_shapes"]["pixel_values"][0] > 0
        assert task["tensor_shapes"]["image_grid_thw"] == [1, 3]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    assert not ledger["delivery_inventory"]["CHARTQA"]["checked"]
    assert ledger["delivery_inventory"]["CHARTQA"]["original_processor_preflight"]


def test_new_installed_research_handoff_is_not_published_lock_or_native_run():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    status = ledger["research_consumer_latest"]
    receipt = json.loads((ROOT / status["receipt"]).read_text("utf-8"))
    assert receipt["software_tests_passed"] == 62
    assert receipt["software_tests_skipped"] == 0
    assert {item["source_id"] for item in receipt["checks"]} == set(
        status["actual_sources_verified"]
    )
    assert len(receipt["checks"]) == 4
    for item in receipt["checks"]:
        assert item["original_inputs_and_independent_references_equal"]
        assert (
            item["public_index_to_research_rows"]
            == item["direct_research_prepare_resolve_rows"]
            == 3
        )
    assert receipt["new_model_calls"] == 0
    assert not receipt["research_pr_lock_updated"]
    assert not receipt["all_37_sources_revalidated"]
    assert not receipt["observation_or_analysis_code_migrated_into_benchmark"]


def test_burstgpt_official_simulation_not_claimed_as_original_request_replay():
    receipt = json.loads(
        (DOCS / "validation/burstgpt-official-demo-review-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["external_prompt_pool_required"]
    assert not receipt["original_prompt_replay"]
    assert not receipt["source_checkbox_eligible"]
    assert (
        receipt["official_demo_execution"]
        == "NOT_EXECUTED_TRANSPORT_INCOMPATIBLE_WITH_CURRENT_SERVICE"
    )
    assert (
        receipt["current_service_openapi_probe"]["generate_endpoint_present"] is False
    )
    assert receipt["new_model_calls"] == 0


def test_locomo_supplement_does_not_close_unverified_category5_or_full_delivery():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["LOCOMO"]
    assert item["checked"] is False
    receipt = json.loads((ROOT / item["category4_receipt"]).read_text("utf-8"))
    assert receipt["original_question_indices"] == [82, 83, 84]
    assert receipt["new_model_calls"] == 3
    assert receipt["combined_original_question_count"] == 9
    assert not receipt["category5_validated"]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["raw_data_or_outputs_published"]
    assert all(
        sample["prompt_tokens"] == sample["preflight_tokens"]
        for sample in receipt["samples"]
    )
    assert all(sample["finish_reason"] == "stop" for sample in receipt["samples"])
    preflight = json.loads(
        (ROOT / item["category5_original_preflight"]).read_text("utf-8")
    )
    assert preflight["category5_original_tasks"] == 446
    assert preflight["category5_without_answer_key"] == 444
    assert preflight["model_api_attempts"] == preflight["model_calls"] == 0
    assert not preflight["task_fields_renamed_or_filled"]
    assert not preflight["source_checkbox_eligible"]


def test_private_request_delivery_reuses_native_responses_without_publishing_them():
    import hashlib

    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    private = {
        key: row
        for key, row in ledger["delivery_inventory"].items()
        if row["scope"] == "enterprise_private"
    }
    receipt = json.loads(
        (DOCS / "validation/enterprise-fixed-ci-handoff-20261024.json").read_text(
            "utf-8"
        )
    )
    assert len(private) == receipt["source_count"] == len(receipt["sources"]) == 8
    assert set(private) == {row["source_id"] for row in receipt["sources"]}
    assert all(
        row["status"] == "AVAILABLE" and row["checked"] for row in private.values()
    )
    assert receipt["delivery_kind"] == "private_request_corpus"
    assert receipt["source_checkbox_eligible"] is True
    assert (
        receipt["original_request_roundtrips"]
        == receipt["existing_native_responses_revalidated"]
        == 24
    )
    assert receipt["new_model_calls"] == 0
    assert not receipt["quality_claim"]
    assert not receipt["full_source_inference_validated"]
    assert not receipt["raw_enterprise_data_published"]
    assert not receipt["paths_or_request_ids_exported"]
    script = ROOT / receipt["public_runner"]
    assert (
        hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        == receipt["public_runner_sha256"]
    )
    for result in receipt["sources"]:
        assert result["original_file_sha_matches_native_receipt"]
        assert result["private_response_bundle_matches_native_receipt"]
        assert (
            result["fixed_ci_reader_roundtrips"]
            == result["private_responses_revalidated"]
            == 3
        )
        original = json.loads(
            (ROOT / private[result["source_id"]]["existing_receipt"]).read_text("utf-8")
        )
        assert (
            original["status"]
            == "REQUESTS_EXECUTED_AND_INTEGRITY_VALIDATED_NO_QUALITY_ORACLE"
        )
        assert original["quality_claim"] is False


def test_current_delivery_inventory_and_checklist_are_aligned():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    contract = ledger["delivery_contract"]
    inventory = ledger["delivery_inventory"]
    assert len(inventory) == contract["planned_sources"] == 118
    excluded = set(contract["classification_only_source_ids"])
    assert len(excluded) == 5
    assert len(inventory) - len(excluded) == contract["in_scope_sources"] == 113
    assert sum(v["scope"] == "enterprise_private" for v in inventory.values()) == 8
    lines = (DOCS / "DATASET_CHECKLIST.zh-CN.md").read_text("utf-8").splitlines()
    delivery_header = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("|")
        and len(line.split("|")) > 3
        and line.split("|")[1].strip() == "来源 ID"
        and line.split("|")[2].strip() == "本轮勾选"
    )
    history_header = next(
        i for i, line in enumerate(lines) if line.startswith("## 历史抽样记录")
    )
    delivery_table = lines[delivery_header + 2 : history_header]
    for source_id, item in inventory.items():
        assert item["checked"] == (item["status"] == "AVAILABLE")
        if source_id in excluded:
            assert item["status"] == "CLASSIFICATION_ONLY"
            assert not item["checked"]
        rows = [
            line
            for line in delivery_table
            if line.startswith("|") and line.split("|")[1].strip() == source_id
        ]
        assert len(rows) == 1
        assert ("[x]" in rows[0]) == item["checked"]
        if item["checked"]:
            assert (ROOT / item["delivery_receipt"]).is_file()


def test_existing_37_audit_checkmarks_do_not_mean_native_delivery():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    review = ledger["existing_37_review"]
    assert review["review_complete"] is True
    assert review["count"] == 37
    assert review["public_count"] == 29
    assert review["enterprise_private_count"] == 8
    assert all(item["review_complete"] for item in review["sources"].values())
    revalidation = review["local_revalidation"]
    assert revalidation["date"] == "2026-10-24"
    assert revalidation["test_result"] == "109 passed"
    assert revalidation["native_benchmark_reruns"] == 0
    assert revalidation["npu_requests"] == 0
    assert revalidation["enterprise_payloads_added"] is False
    snapshot = ledger["existing_37_review"]["comparison_snapshot"]
    assert (
        # pragma: allowlist nextline secret - public source revision
        snapshot["benchmark_main_sha"] == "140a046d34ecf50bb8de3d8b19a7d70a6b0490e0"
    )
    assert snapshot["benchmark_pr_stack_state"] == "All six PRs open Draft and unmerged"
    assert (
        snapshot["benchmark_pr_ci"]["head_sha"]
        == snapshot["benchmark_pr_stack_heads"]["432"]
    )
    assert snapshot["benchmark_pr_ci"]["ci_conclusion"] == "success"
    assert snapshot["benchmark_pr_ci"]["cpu_correctness_smoke_conclusion"] == "success"
    assert (
        snapshot["research_pr37_ci"]["head_sha"] == snapshot["research_pr37_head_sha"]
    )
    assert snapshot["research_pr37_ci"]["ci_conclusion"] == "success"
    assert snapshot["research_pr37_ci"]["consumer_model_free_conclusion"] == "success"
    assert snapshot["research_main_sha"] == snapshot["research_pr37_base_sha"]
    assert (
        snapshot["research_pr37_head_sha"]
        == ledger["existing_37_review"]["research_consumer_commit"]
    )
    assert set(snapshot["paths_changed_on_both_latest_main_and_pr_stack"]) == {
        "README.md",
        "docs/data-index.json",
    }
    assert (
        sum(
            item["consumer_evidence"] == "source_specific_receipt"
            for item in review["sources"].values()
        )
        == 23
    )
    assert (
        review["sources"]["STRUCTEVAL"]["consumer_evidence"]
        == "source_specific_receipt"
    )
    assert (
        review["sources"]["TAUBENCH"]["consumer_evidence"] == "source_specific_receipt"
    )
    consumer_audit = ledger["existing_37_review"]["research_consumer_reconciliation"]
    no_source_receipt = {
        source_id
        for source_id, item in review["sources"].items()
        if item["consumer_evidence"] != "source_specific_receipt"
    }
    missing_by_scope = {
        scope: sorted(
            source_id
            for source_id, item in review["sources"].items()
            if item["consumer_evidence"] != "source_specific_receipt"
            and item["scope"] == scope
        )
        for scope in ("public", "enterprise_private")
    }
    assert consumer_audit["source_specific_receipt_count"] == 23
    assert (
        set(consumer_audit["source_specific_receipt_missing_ids"]) == no_source_receipt
    )
    assert consumer_audit["source_specific_receipt_missing_scope"] == missing_by_scope
    assert sum(map(len, missing_by_scope.values())) == len(no_source_receipt) == 14
    supplemental = consumer_audit["local_supplemental_consumer_receipts"]
    private_ids = {
        source_id
        for source_id, item in review["sources"].items()
        if item["scope"] == "enterprise_private"
    }
    private_receipt = supplemental["enterprise_private"]
    public_receipt = supplemental["public"]
    assert set(private_receipt["source_ids"]) == private_ids
    assert private_receipt["sources_passed"] == len(private_ids) == 8
    assert private_receipt["rows_per_source_prepared_and_resolved"] == 3
    public_ids = set(public_receipt["source_ids"])
    assert public_ids == {
        "AI2_ARC",
        "DOLLY",
        "GSM8K",
        "LONGBENCH_V2",
        "LONGWRITER",
        "MMLU_PRO",
    }
    assert set(public_receipt["source_ids"]) == public_ids
    assert public_receipt["sources_passed"] == len(public_ids) == 6
    assert public_receipt["prepared_rows"] == public_receipt["resolved_rows"] == 24
    assert (
        consumer_audit["source_specific_consumer_evidence_total_after_local_supplement"]
        == 37
    )
    assert (
        consumer_audit[
            "remaining_source_specific_receipt_missing_ids_after_local_supplement"
        ]
        == []
    )
    for source_id in private_ids:
        entry = review["sources"][source_id]
        assert entry["local_supplemental_consumer_receipt"] == private_receipt["path"]
        assert entry["local_supplemental_consumer_rows"] == 3
        assert entry["local_supplemental_consumer_resolved_rows"] == 3
    for source_id in public_ids:
        entry = review["sources"][source_id]
        assert entry["local_supplemental_consumer_receipt"] == public_receipt["path"]
        assert entry["local_supplemental_consumer_rows"] >= 3
        assert (
            entry["local_supplemental_consumer_resolved_rows"]
            == entry["local_supplemental_consumer_rows"]
        )
    reconciled = consumer_audit["newly_reconciled_receipts"]
    assert reconciled["STRUCTEVAL"]["model_calls"] == 0
    assert reconciled["STRUCTEVAL"]["task_executions"] == 0
    assert reconciled["TAUBENCH"]["model_calls"] == 0
    assert "Three original references only" in reconciled["TAUBENCH"]["scope"]
    lines = (DOCS / "DATASET_CHECKLIST.zh-CN.md").read_text("utf-8").splitlines()
    review_header = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("|")
        and len(line.split("|")) > 3
        and line.split("|")[1].strip() == "来源 ID"
        and line.split("|")[2].strip() == "证据与能力复核"
    )
    delivery_header = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("|")
        and len(line.split("|")) > 3
        and line.split("|")[1].strip() == "来源 ID"
        and line.split("|")[2].strip() == "本轮勾选"
    )
    review_table = lines[review_header + 2 : delivery_header]
    for source_id in review["sources"]:
        rows = [
            line
            for line in review_table
            if line.startswith("|") and line.split("|")[1].strip() == source_id
        ]
        assert len(rows) == 1
        assert "[x] 已复核" in rows[0]
        if source_id in private_ids or source_id in public_ids:
            assert "本地补充消费" in rows[0]
    assert review["sources"]["QRECC"]["receipt_is_standalone_json"] is False


def test_existing_37_sources_are_registered_in_the_public_workload_package():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    review = ledger["existing_37_review"]
    catalogs = ROOT / "workload-preparation" / "data" / "catalogs"
    manifests = json.loads((catalogs / "source_manifests_v1.json").read_text("utf-8"))[
        "sources"
    ]
    public_catalog = json.loads(
        (catalogs / "authoritative_datasets_v1.json").read_text("utf-8")
    )["sources"]
    adapters = json.loads((catalogs / "adapter_registry_v1.json").read_text("utf-8"))[
        "adapters"
    ]
    adapters.update(
        json.loads((catalogs / "adapter_extensions_v1.json").read_text("utf-8"))[
            "adapters"
        ]
    )
    private_assets = json.loads(
        (catalogs / "dataset_assets_v1.json").read_text("utf-8")
    )["assets"]

    public_ids = {
        source_id
        for source_id, item in review["sources"].items()
        if item["scope"] == "public"
    }
    private_ids = {
        source_id
        for source_id, item in review["sources"].items()
        if item["scope"] == "enterprise_private"
    }
    assert len(public_ids) == review["public_count"] == 29
    assert len(private_ids) == review["enterprise_private_count"] == 8
    assert public_ids | private_ids == set(review["sources"])
    assert public_ids <= set(manifests)
    assert public_ids <= {item["source_id"] for item in public_catalog}
    assert public_ids <= set(adapters)
    assert private_ids <= set(manifests)
    assert private_ids <= set(adapters)
    assert private_ids <= {item["dataset_id"] for item in private_assets}


def test_local_private_consumer_receipt_is_source_complete_and_sanitized():
    receipt_path = DOCS / "validation/enterprise-private-source-consumer-20261024.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    expected = {
        "PAIO-CHAT-1000",
        "PAIO-CODE-EVAL-1000",
        "PAIO-JSON-500",
        "PAIO-LONG-PREFILL-5000",
        "PAIO-LONGTEXT-4000",
        "PAIO-PREFIX-SHARED-5000",
        "PAIO-REUSE-CONV-5000",
        "PAIO-SEMANTIC-SIMILAR-5000",
    }
    assert (
        receipt["status"]
        == "ORIGINAL_DATA_SOURCE_CONSUMER_ROUNDTRIP_PASSED_NOT_BENCHMARK_EXECUTION"
    )
    assert {row["source_id"] for row in receipt["results"]} == expected
    assert all(
        row["prepared_rows"] == row["original_input_resolved_rows"] == 3
        and row["status"] == "PASS"
        for row in receipt["results"]
    )
    assert receipt["totals"]["npu_requests"] == 0
    assert receipt["totals"]["model_calls"] == 0
    assert receipt["totals"]["native_task_executions"] == 0
    assert receipt["totals"]["scorer_calls"] == 0
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert "C:\\Users" not in serialized
    assert "派欧云" not in serialized
    assert "request_body" not in serialized
    assert receipt["selection"]["row_ids_recorded"] is False
    assert receipt["selection"]["request_or_answer_content_recorded"] is False


def test_local_public_consumer_receipt_is_source_complete_and_sanitized():
    receipt_path = DOCS / "validation/public-source-consumer-20261024.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    expected = {
        "AI2_ARC",
        "DOLLY",
        "GSM8K",
        "LONGBENCH_V2",
        "LONGWRITER",
        "MMLU_PRO",
    }
    assert (
        receipt["status"]
        == "ORIGINAL_PUBLIC_DATA_SOURCE_CONSUMER_ROUNDTRIP_PASSED_NOT_BENCHMARK_EXECUTION"
    )
    assert {row["source_id"] for row in receipt["results"]} == expected
    assert all(
        row["prepared_rows"] == row["original_input_resolved_rows"]
        and row["prepared_rows"] >= 3
        and row["status"] == "PASS"
        for row in receipt["results"]
    )
    assert receipt["results"][0]["prepared_rows"] == 9
    assert receipt["totals"]["prepared_rows"] == 24
    assert receipt["totals"]["original_input_resolved_rows"] == 24
    assert receipt["totals"]["npu_requests"] == 0
    assert receipt["totals"]["model_calls"] == 0
    assert receipt["totals"]["native_task_executions"] == 0
    assert receipt["totals"]["scorer_calls"] == 0
    serialized = json.dumps(receipt, ensure_ascii=False)
    assert "C:\\Users" not in serialized
    assert ".cache" not in serialized
    assert "request_body" not in serialized
    assert receipt["selection"]["row_ids_recorded"] is False
    assert receipt["selection"]["request_or_answer_content_recorded"] is False


def test_azure_is_explicitly_metadata_not_native_inference():
    receipt = json.loads(
        (DOCS / "validation/azure-serving-trace-delivery-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["status"] == "DATA_ASSET_AVAILABLE_NOT_NATIVE_MODEL_BENCHMARK"
    assert receipt["full_rows_read_and_indexed"] == 28185
    assert receipt["sampled_original_metadata_roundtrips"] == 6
    assert receipt["model_calls"] == receipt["scorer_calls"] == 0
    assert receipt["fabricated_prompts"] == 0
    assert not receipt["native_inference_replay_verified"]
    assert not receipt["raw_payloads_published"]
