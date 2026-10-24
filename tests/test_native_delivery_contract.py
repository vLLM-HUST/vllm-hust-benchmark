"""Keep current delivery checkboxes distinct from historical reader evidence."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_qasper_full_original_media_decode_is_not_vision_model_execution():
    receipt = json.loads(
        (DOCS / "validation/qasper-original-media-cpu-audit-20261025.json").read_text(
            "utf-8"
        )
    )
    assert receipt["unique_original_figure_table_references"] == 11364
    assert receipt["matched_reference_images_decoded"] == 11364
    assert receipt["missing_reference_images"] == 0
    assert receipt["previous_train_dev_cpu_decodes_reused"] == 8486
    assert receipt["new_test_cpu_decodes"] == 2878
    assert receipt["all_matched_images_fully_cpu_decoded"]
    assert receipt["image_formats"] == {"PNG": 11364}
    assert receipt["test_archive_root_explicitly_mapped_to_test_split"]
    assert not receipt["images_cropped_resized_or_replaced"]
    assert not receipt["vision_model_inference_verified"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["model_calls"] == 0


def test_qasper_fixed_ci_research_consumer_preserves_raw_tasks_not_scientific_acceptance():
    receipt = json.loads(
        (DOCS / "validation/qasper-ci-research-handoff-20261025.json").read_text(
            "utf-8"
        )
    )
    assert (
        receipt["data_commit"] == "39556148d18a30bb1f6d3ec04cba663ab8d67db6"
    )  # pragma: allowlist secret - public CI commit
    assert receipt["data_cpu_ci_run"] == 40119103024
    assert receipt["installed_provider_payload_files_equal"] == 33
    assert receipt["research_installed_source_files_equal"] == 40
    assert [row["original_positions"] for row in receipt["splits"]] == [[0, 1, 2]] * 3
    assert receipt["public_prepare_resolve_rows"] == 9
    assert receipt["research_source_index_import_rows"] == 9
    assert receipt["research_direct_prepare_resolve_rows"] == 9
    assert receipt[
        "original_answer_evidence_annotations_and_full_paper_input_preserved"
    ]
    assert receipt["new_model_calls"] == 0
    assert not receipt["research_default_dependency_lock_updated"]
    assert not receipt["research_observation_or_analysis_migrated_into_benchmark"]
    assert not receipt["research_feature_validity_or_observation_acceptance_claim"]
    assert not receipt["official_scoring_executed"]
    assert not receipt["source_checkbox_eligible"]


def test_qasper_full_original_provider_proof_is_not_native_or_ci_completion():
    receipt = json.loads(
        (
            DOCS / "validation/qasper-original-provider-cpu-audit-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["installed_payload_files_equal"] == 33
    assert (
        receipt["all_original_inputs_references_and_parent_question_indices_equal"]
        == 5049
    )
    assert receipt["original_papers"] == 1585
    assert receipt["full_index_rows"] == 5049
    assert receipt["prepare_resolve_roundtrips"] == 18
    assert [row["original_questions"] for row in receipt["splits"]] == [
        2593,
        1005,
        1451,
    ]
    assert receipt["all_answer_evidence_annotations_remain_reference_only"]
    assert not receipt["candidate_is_successful_ci_artifact"]
    assert not receipt["research_handoff_completed_by_this_audit"]
    assert not receipt["official_scoring_executed"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["model_calls"] == 0


def test_burstgpt_first_file_native_simulation_keeps_other_file_gaps():
    receipt = json.loads(
        (DOCS / "validation/burstgpt-release1-native-audit-20261025.json").read_text(
            "utf-8"
        )
    )
    assert receipt["original_indices"] == list(range(20))
    assert receipt["native_requests"] == 20
    assert receipt["terminal_counts"] == {"length": 12, "stop": 8}
    assert receipt["all_native_requests_equal_original_publisher_query_generator"]
    assert receipt["legacy_response_prefix_mapping_and_saved_hashes_equal"]
    assert receipt["official_detail_events_and_final_summary_verified"]
    assert receipt["original_output_budget"] == 1024
    assert receipt["scale"] == 1
    assert receipt["model_calls_by_audit"] == 0
    assert not receipt["arrival_offset_shifted_or_time_compressed"]
    assert not receipt["other_two_release_files_native_simulation_verified"]
    assert not receipt["external_pool_is_original_private_burstgpt_prompt"]
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["research_observation_or_analysis_migrated"]


def test_longbench_v2_capacity_blockers_are_not_filtered_into_completion():
    receipt = json.loads(
        (
            DOCS / "validation/longbench-v2-length-group-capacity-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["original_group_counts"] == {
        "short": 180,
        "medium": 215,
        "long": 108,
    }
    medium, long = receipt["groups"]
    assert medium["fixed_original_indices"] == [3, 5, 8]
    assert medium["full_original_prompt_tokens"] == [149662, 138653, 68762]
    assert long["fixed_original_indices"] == [0, 2, 7]
    assert long["full_original_prompt_tokens"] == [295877, 530991, 815430]
    assert receipt["deployed_context_limit"] == 131072
    assert receipt["original_output_budget"] == 128
    assert receipt["selection_frozen_before_capacity_checks"]
    assert receipt["original_three_short_tasks_reused_not_rerun"]
    assert not any(group["group_ready_for_execution"] for group in receipt["groups"])
    assert not receipt["blocked_tasks_truncated_replaced_or_skipped_for_completion"]
    assert not receipt["source_checkbox_eligible"]
    assert receipt["model_calls"] == 0


def test_qasper_complete_native_tasks_are_not_full_longbench_or_research_handoff():
    receipt = json.loads(
        (DOCS / "validation/longbench-qasper-all200-native-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["original_unique_tasks"] == 200
    assert receipt["new_native_tasks"] == 197
    assert receipt["historical_native_tasks_reused"] == 3
    assert receipt["all_original_inputs_references_and_request_token_counts_verified"]
    assert receipt["all_saved_response_hashes_and_official_scores_recalculated"]
    assert receipt["installed_ci_payloads_equal"] == 32
    assert sum(receipt["terminal_counts"].values()) == 200
    assert receipt["official_qa_f1_percent"] == 50.7
    assert receipt["full_original_qasper_claim"]
    assert not receipt["full_longbench_other_subsuites_claim"]
    assert not receipt["single_uninterrupted_run_claim"]
    assert not receipt["research_side_exact_artifact_handoff_verified_by_this_audit"]
    assert not receipt["original_output_budget_or_inputs_changed"]
    assert not receipt["original_questions_replaced"]
    assert receipt["audit_model_calls"] == 0
    assert not receipt["performance_or_engine_speedup_claim"]


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


def test_tracelab_duckdb_asset_acquisition_is_not_conversion_or_replay():
    receipt = json.loads(
        (DOCS / "validation/tracelab-duckdb-original-20261024.json").read_text("utf-8")
    )
    assert receipt["asset_id"] == 488289840
    assert receipt["asset_bytes_verified"] == 160182272
    assert not receipt["duckdb_open_or_csv_conversion_verified"]
    assert not receipt["native_replay_executed"]
    assert receipt["model_calls"] == 0
    assert not receipt["source_completion_checked"]
    assert not receipt["raw_payloads_public"]


def test_tracelab_full_metadata_index_is_not_native_replay():
    receipt = json.loads(
        (DOCS / "validation/tracelab-full-index-20261024.json").read_text("utf-8")
    )
    assert receipt["prepared_rows"] == 665453
    assert receipt["full_source_index_prepared"]
    assert receipt["prepare_resolve_roundtrips"] == 6
    assert receipt["fixed_original_positions"] == [0, 1, 2, 665450, 665451, 665452]
    assert receipt["model_request_rendering_refused"]
    assert receipt["model_calls"] == receipt["native_parent_tasks_executed"] == 0
    assert not receipt["source_completion_checked"]


def test_tracelab_full_official_csv_preparation_is_not_model_replay():
    receipt = json.loads(
        (DOCS / "validation/tracelab-official-csv-preparation-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["source_unmodified"] and receipt["original_database_unchanged"]
    assert receipt["original_database_rows"] == 665453
    assert len(receipt["source_files"]) == 6
    assert receipt["isolated_environment"]
    assert receipt["packages"] == {"duckdb": "1.5.3", "numpy": "2.4.6"}
    assert receipt["package_versions_match_upstream_lock"]
    assert not receipt["full_upstream_uv_environment_installed"]
    assert receipt["pip_check_passed"]
    assert receipt["repeat_outputs_byte_identical"]
    assert len(receipt["runs"]) == 2
    assert all(
        run["process_returncode"] == 0 and run["csv_rows"] == 665453
        for run in receipt["runs"]
    )
    assert receipt["runs"][0]["csv_sha256"] == receipt["runs"][1]["csv_sha256"]
    assert receipt["configuration"]["max_sessions"] is None
    assert not receipt["original_sessions_limited_or_replaced"]
    assert receipt["upstream_field_clamping_and_synthetic_arrivals_preserved"]
    assert receipt["model_calls"] == 0
    assert not receipt["native_replay_executed"]
    assert not receipt["source_completion_checked"]


def test_tracelab_failed_first_locked_build_remains_failed_evidence():
    receipt = json.loads(
        (DOCS / "validation/tracelab-original-build-attempt-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["status"] == "OFFICIAL_LOCKED_BUILD_FAILED"
    assert receipt["process_returncode"] == 101
    assert receipt["locked_build"] and receipt["original_source_and_lock_unchanged"]
    assert receipt["isolated_prefix_cache_and_target"]
    assert not receipt["global_toolchain_or_model_environment_changed"]
    assert receipt["binary_sha256"] is None
    assert not receipt["native_replay_executed"]
    assert receipt["model_calls"] == 0


def test_tracelab_independent_text_pool_is_not_original_trace_prompts():
    receipt = json.loads(
        (DOCS / "validation/tracelab-independent-text-pool-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["zip_bytes"] == 322592222
    assert receipt["original_text_bytes"] == 1000000000
    assert receipt["full_utf8_decode_and_zip_crc_verified"]
    assert not receipt["token_pool_created_or_capacity_verified"]
    assert not receipt["original_private_trace_prompts"]
    assert not receipt["new_benchmark_or_dataset_inventory_entry"]
    assert not receipt["native_replay_executed"]
    assert receipt["model_calls"] == 0
    assert not receipt["raw_payloads_public"]


def test_tracelab_locked_build_and_cpu_dry_run_are_not_npu_replay():
    build = json.loads(
        (DOCS / "validation/tracelab-original-locked-build-20261024.json").read_text(
            "utf-8"
        )
    )
    dry = json.loads(
        (DOCS / "validation/tracelab-original-dry-run-20261024.json").read_text("utf-8")
    )
    assert build["status"] == "OFFICIAL_LOCKED_BUILD_SUCCEEDED"
    assert build["process_returncode"] == 0
    assert build["original_source_and_lock_unchanged"] and build["locked_build"]
    assert build["isolated_prefix_cache_and_target"]
    assert not build["global_toolchain_or_model_environment_changed"]
    assert build["native_library_binding"]["package_version"] == "3.6.4"
    assert build["binary_sha256"] == dry["binary_sha256"]
    assert dry["full_native_dry_run_returncode"] == 0
    assert dry["sample_native_dry_run_returncode"] == 0
    assert dry["full_sessions"] == 8058 and dry["full_steps"] == 665453
    assert dry["fixed_sample_sessions"] == 3
    assert dry["fixed_sample_capacity"] == {
        "steps": 20,
        "max_full_prompt_plus_original_output": 46646,
        "steps_exceeding_131072": 0,
    }
    assert dry["full_capacity"]["steps_exceeding_131072"] == 336898
    assert dry["complete_rounds_preserved"]
    assert not dry["context_overflow_skip_flag_enabled"]
    assert not dry["tokenizer_and_text_pool_loaded_by_dry_run"]
    assert not dry["token_pool_capacity_verified"]
    assert dry["model_calls"] == dry["native_model_attempted_steps"] == 0
    assert not dry["mandatory_server_cache_probe_verified"]
    assert not dry["native_replay_executed"]
    assert not dry["source_completion_checked"]


def test_tracelab_default_token_pool_cpu_diagnostic_is_not_original_runner():
    receipt = json.loads(
        (DOCS / "validation/tracelab-original-token-pool-cpu-20261024.json").read_text(
            "utf-8"
        )
    )
    assert receipt["status"] == "ORIGINAL_TOKEN_POOL_CPU_DIAGNOSTIC_PASSED"
    assert receipt["process_returncode"] == 0
    assert receipt["original_source_and_tokenizer_unchanged"]
    assert "Standalone CPU wrapper" in receipt["scope"]
    assert receipt["requested_default_pool_tokens"] == 100000000
    assert receipt["actual_pool_tokens"] == 100000000
    assert receipt["native_auto_pool_limit_matches_full_and_fixed_sample"]
    dry = json.loads(
        (DOCS / "validation/tracelab-original-dry-run-20261024.json").read_text("utf-8")
    )
    assert (
        max(
            2 * dry["full_capacity"]["max_full_prompt_plus_original_output"],
            dry["full_sessions"],
            100000000,
        )
        == receipt["requested_default_pool_tokens"]
    )
    assert not receipt["pool_parameters_reduced"]
    assert len(receipt["original_lock_built_dependency_rlib_sha256"]) == 6
    assert receipt["model_calls"] == 0
    assert not receipt["native_replay_executed"]
    assert not receipt["mandatory_server_cache_probe_verified"]
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


def test_locomo_historical_supplement_does_not_rewrite_category5_preflight():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["LOCOMO"]
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


def test_locomo_delivery_requires_native_all_categories_and_exact_handoff():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["LOCOMO"]
    assert item["checked"] and item["status"] == "AVAILABLE"
    delivery = json.loads((ROOT / item["delivery_receipt"]).read_text("utf-8"))
    native = json.loads(
        (ROOT / item["native_execution_audit_receipt"]).read_text("utf-8")
    )
    handoff = json.loads(
        (ROOT / item["research_exact_artifact_handoff_receipt"]).read_text("utf-8")
    )
    assert (
        delivery["verified_unique_original_tasks"]
        == native["verified_unique_original_tasks"]
        == 108
    )
    assert native["new_unique_native_original_tasks"] == 99
    assert native["historical_unique_native_original_tasks_reused"] == 9
    assert native["original_category_coverage"] == {
        "1": 3,
        "2": 3,
        "3": 96,
        "4": 3,
        "5": 3,
    }
    assert native["all_native_queries_original_replay_equal"]
    assert native["all_saved_responses_and_scores_equal"]
    assert native["native_original32_token_budget_and_sleep_retained"]
    assert native["audit_model_calls"] == 0
    assert native["saved_responses_replayed_not_model_rerun"] == 99
    assert handoff["original_full_conversation_question_and_reference_equal"]
    assert handoff["original_cat5_reference_rows_preserved"] == 3
    assert not handoff["missing_answer_field_added_to_raw_data_or_research_record"]
    assert not handoff["observation_or_analysis_migrated_into_benchmark"]
    assert not delivery["full1986_native_execution_claim"]
    assert not delivery["upstream_merged_fix_claim"]
    assert not delivery["image_byte_task_or_rag_validation_claim"]
    assert not delivery["raw_payloads_published"]


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


def test_runtime_prerequisite_rechecks_never_upgrade_blocked_sources():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    receipt = json.loads(
        (ROOT / ledger["runtime_prerequisite_latest"]).read_text("utf-8")
    )
    assert receipt["new_model_calls"] == receipt["new_source_completion_checks"] == 0
    assert receipt["code_execution_isolation"]["returncode"] == 1
    assert not receipt["code_execution_isolation"]["generated_code_executed"]
    assert not receipt["code_execution_isolation"]["host_security_policy_changed"]
    assert receipt["paws_official_archive_head"]["http_status"] == 403
    assert not receipt["paws_official_archive_head"]["dataset_acquired"]
    assert not receipt["tracelab_cache_gate"]["actual_cache_probe_executed"]
    assert not receipt["tracelab_cache_gate"]["future_probe_failure_or_fix_claimed"]
    # Later real runs may legitimately complete a source; this read-only receipt
    # must never itself become its execution/delivery evidence.
    assert all(
        ledger["delivery_inventory"][source].get("delivery_receipt")
        != ledger["runtime_prerequisite_latest"]
        for source in receipt["source_ids"]
    )


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
