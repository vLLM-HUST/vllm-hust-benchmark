"""Keep current delivery checkboxes distinct from historical reader evidence."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_crosscodeeval_native_delivery_keeps_original_profiles_and_ci_consumer_scope():
    receipt = json.loads(
        (
            DOCS / "validation/crosscodeeval-original-acquisition-20261026.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["source_checkbox_eligible"] is True
    native = receipt["public_native_entry_validation"]
    assert native["original_positions_per_language"] == [3, 4, 5]
    assert (
        native["original_cases_generated"]
        == native["original_cases_scored_by_public_entry"]
        == 12
    )
    assert native["previous_zero_through_two_repeated"] is False
    assert native["publisher_generation_and_scoring_python_unchanged"] is True
    assert native["full_corpus_native_or_quality_acceptance_claim"] is False
    consumer = receipt["fixed_ci_consumer_closure"]
    assert consumer["both_ci_conclusions"] == "success"
    assert consumer["installed_payloads_verified"] == 44
    assert consumer["unchanged_previously_full_audited_payloads"] == 43
    assert consumer["reused_full_index_rows"] == 69494
    assert consumer["public_and_research_roundtrips"] == 12
    assert consumer["new_model_calls"] == 0
    assert consumer["research_default_dependency_changed"] is False


def test_fleurs_native_delivery_is_all_locale_sampled_asr_not_full_test_or_paper():
    receipt = json.loads(
        (DOCS / "validation/fleurs-native-delivery-20261026.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["source_checkbox_eligible"]
    native = receipt["native_execution"]
    assert (
        native["native_locales"] == 102 and native["completed_original_samples"] == 306
    )
    assert native["original_positions_per_locale"] == [0, 1, 2]
    assert native["original_full_waveforms_and_annotations_verified"]
    assert not native["audio_clipped_resampled_or_rewritten"]
    assert not native["language_forced_or_failed_cases_replaced"]
    assert not native["full_77810_native_claim"]
    assert receipt["scoring"]["independent_saved_output_regrade_cases"] == 306
    assert receipt["scoring"]["new_model_calls_during_audit"] == 0
    assert not receipt["scoring"]["conformer_training_or_original_paper_baseline_claim"]
    consumer = receipt["fixed_ci_consumer"]
    assert consumer["installed_payloads_verified"] == 44
    assert consumer["public_and_research_roundtrips"] == 9
    assert not consumer["research_default_dependency_changed"]
    assert not receipt["raw_audio_transcripts_or_predictions_published"]


def test_original_qasper_native_delivery_retains_zero_evidence_and_consumer_scope():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["QASPER"]
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert row["status"] == "AVAILABLE" and row["checked"]
    assert (
        receipt["source_checkbox_eligible"] and receipt["not_longbench_derived_qasper"]
    )
    assert receipt["original_population"]["questions"] == 5049
    native = receipt["native_profile"]
    assert native["original_split"] == "dev" and native["original_positions"] == [
        0,
        1,
        2,
    ]
    assert (
        native["model_calls"]
        == native["stop"]
        == native["valid_native_predictions"]
        == 3
    )
    assert not native["input_truncated"]
    assert not native["sibling_question_or_gold_annotations_in_input"]
    assert not native["predictions_repaired_retried_or_replaced"]
    grading = receipt["publisher_grading"]
    assert not grading["evaluator_modified"] and not grading["text_evidence_only"]
    assert grading["standalone_original_cli_exit_code"] == 0
    assert grading["inprocess_and_standalone_cli_scores_identical"]
    assert grading["scores"]["Evidence F1"] == 0
    assert not grading["quality_acceptance_or_paper_baseline_claim"]
    consumer = receipt["fixed_provider_current_research_consumer"]
    assert consumer["installed_noneditable_payloads_byte_verified"] == 42
    assert consumer["current_public_research_roundtrips"] == 9
    assert consumer["existing_full5049_field_index_and_media_checks_not_rerun"]
    assert not consumer["research_default_dependency_changed"]


def test_longalpaca_original_corpus_delivery_is_not_fake_quality_benchmark():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["LONGALPACA"]
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert row["status"] == "AVAILABLE" and row["checked"]
    assert receipt["source_checkbox_eligible"]
    preparation = receipt["full_preparation_and_fidelity"]
    assert preparation["original_rows_and_index"] == 12000
    assert preparation["original_source_file_preserved"] == 3231
    assert not preparation["new_feature_or_relationship_inference"]
    native = receipt["native_execution"]
    assert native["original_ranges"] == [[0, 2], [7324, 7326]]
    assert (
        native["model_calls"] == native["unique_original_cases"] == native["stop"] == 6
    )
    assert native["nonempty_input_branch_native_cases"] == 2
    assert not native["input_truncated"]
    assert not native["reference_or_provenance_injected_into_input"]
    assert not native["retry_or_failed_case_replacement"]
    assert not native["full12000_generation_claim"] and not native["quality_scored"]
    consumer = receipt["fixed_provider_and_research_consumer"]
    assert consumer["actual_installed_noneditable_payloads_verified"] == 42
    assert len(consumer["public_research_resolved_positions"]) == 12
    assert not consumer["research_default_dependency_changed"]


def test_dureader_full_original_demo_delivery_keeps_failure_and_model_scope():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["DUREADER"]
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert row["status"] == "AVAILABLE" and row["checked"]
    assert receipt["source_checkbox_eligible"]
    profile = receipt["native_profile"]
    assert profile["model_calls"] == profile["stop"] == 100
    assert (
        profile["valid_native_predictions"] + profile["invalid_predictions_retained"]
        == 100
    )
    assert profile["invalid_predictions_retained"] == 6
    assert not profile["invalid_output_repaired_retried_or_swapped"]
    assert not profile["input_truncated"]
    assert profile["original_documents"] == 453
    assert not receipt["model"]["publisher_baseline_replication"]
    assert not receipt["grading"]["quality_acceptance_claim"]
    assert receipt["grading"]["undefined_metric_value"] is None
    assert not receipt["grading"]["scoring_formula_normalization_and_answers_changed"]
    assert (
        receipt["fixed_provider_and_research_consumer"][
            "all_original_rows_and_index_verified"
        ]
        == 200
    )
    assert (
        len(
            receipt["fixed_provider_and_research_consumer"][
                "public_research_resolved_positions"
            ]
        )
        == 12
    )
    assert not receipt["fixed_provider_and_research_consumer"][
        "research_default_dependency_changed"
    ]


def test_ds1000_real_pandas_scoring_is_not_all_library_or_delivery_completion():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["DS1000"]
    receipt = json.loads(
        (ROOT / row["native_functional_scoring_receipt"]).read_text("utf-8")
    )
    assert row["status"] == "AVAILABLE" and row["checked"]
    assert receipt["original_problem_ids"] == [0, 1, 2]
    assert (
        receipt["original_data_rows_equal_between_existing_hf_asset_and_author_archive"]
        == 1000
    )
    assert receipt["official_timeout_seconds"] == 120
    assert receipt["official_max_workers"] == 16
    assert receipt["author_summary"]["count"] == 3
    assert receipt["author_summary"]["mean_as_publisher_formatted"] == 0.667
    assert receipt["runtime"]["pandas"] == "1.5.3"
    assert receipt["new_model_calls"] == 0
    assert not receipt["candidate_repaired_or_replaced"]
    assert not receipt["full1000_functional_execution_claim"]
    assert not receipt["source_checkbox_eligible"]
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    assert set(delivery["fixed_sample_per_library"]) == {
        "Pandas",
        "Numpy",
        "Matplotlib",
        "Tensorflow",
        "Scipy",
        "Sklearn",
        "Pytorch",
    }
    assert delivery["unique_original_tasks_generated_and_functionally_scored"] == 24
    assert delivery["new_original_model_calls"] == 21
    assert delivery["prior_original_model_calls_reused"] == 3
    assert (
        delivery["fixed_consumer"][
            "all_original_records_public_and_research_direct_equal"
        ]
        == 1000
    )
    assert not delivery["full1000_native_execution_claim"]


def test_humaneval_real_official_scoring_retains_failure_and_delivery_gap():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["HUMANEVAL"]
    receipt = json.loads(
        (ROOT / row["native_functional_scoring_receipt"]).read_text("utf-8")
    )
    assert row["status"] == "AVAILABLE" and row["checked"]
    assert receipt["generated_code_executed"]
    assert receipt["passed_by_original_task_order"] == [True, True, False]
    assert receipt["official_pass_at_1"] == 2 / 3
    assert receipt["new_model_calls"] == 0
    assert not receipt["failure_replaced_or_repaired"]
    assert not receipt["official_evaluator_modified"]
    assert receipt["isolation_probe"]["host_uid"] == 65534
    assert receipt["isolation_probe"]["no_new_privs"]
    assert not receipt["source_checkbox_eligible"]
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    assert delivery["public_scorer_server_validation"]["passed"] == [True, True, False]
    assert (
        delivery["fixed_consumer_validation"]["full_original_records_prepared"] == 164
    )
    assert (
        delivery["fixed_consumer_validation"][
            "full_original_records_public_and_research_direct_equal"
        ]
        == 164
    )
    assert not delivery["model_quality_pass_claim"]


def test_flores_third_direction_is_new_positions_not_duplicate_sample_count():
    receipt = json.loads(
        (
            DOCS / "validation/flores200-third-direction-native-audit-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["original_indices"] == [3, 4, 5]
    assert receipt["total_completed_model_calls_across_three_directions"] == 9
    assert receipt["unique_parallel_sentence_positions_across_three_directions"] == 6
    assert receipt["original_inputs_and_references_equal"]
    assert receipt["new_model_calls_by_audit"] == 0
    assert not receipt["source_checkbox_eligible"]


def test_burstgpt_scaled_tool_audit_does_not_claim_original_timing_or_private_prompts():
    receipt = json.loads(
        (DOCS / "validation/burstgpt-scaled-tool-progress-20261025.json").read_text(
            "utf-8"
        )
    )
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["release1_rerun"]
    audit = receipt["release2_audit"]
    assert audit["scale"] == 1000000 and audit["original_indices"] == list(range(20))
    assert audit["original_output_budget"] == 1024
    assert (
        audit["native_model_calls_reused"] == 20 and audit["model_calls_by_audit"] == 0
    )
    assert audit["source_original_trace_and_external_pool_unchanged"]
    assert audit["all_native_requests_equal_original_publisher_query_generator"]
    assert audit["official_detail_events_and_final_summary_verified"]
    assert not audit["original_timing_performance_claim"]
    assert not audit["external_pool_is_original_private_burstgpt_prompt"]
    assert not receipt["whole_source_delivery_claim"]


def test_flores_original_archive_is_not_native_delivery_or_publisher_checksum():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["FLORES200"]
    receipt = json.loads(
        (ROOT / row["acquisition_progress_receipt"]).read_text("utf-8")
    )
    if row["checked"]:
        # Acquisition evidence stays frozen. Later native completion requires
        # a separate receipt; downloading the archive alone is never enough.
        assert row["delivery_receipt"] != row["acquisition_progress_receipt"]
        delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
        assert delivery["source_checkbox_eligible"]
        assert delivery["native"]["translations_reused"] >= 3
        assert (
            delivery["fixed_consumer"]["native_results_public_and_research_roundtrip"]
            >= 3
        )
    else:
        assert row["delivery_receipt"] is None
    assert not receipt["source_checkbox_eligible"] and receipt["model_calls"] == 0
    assert not receipt["publisher_checksum_available"]
    assert not receipt["observed_sha256_is_publisher_checksum"]
    assert not receipt["public_native_entry_complete"]
    assert receipt["splits"]["dev"] == {"files": 204, "row_counts": [997]}
    assert receipt["splits"]["devtest"] == {"files": 204, "row_counts": [1012]}
    assert receipt["splits"]["test"]["files"] == 0
    assert receipt["model_preparation"][
        "original_language_codes_without_model_language_token"
    ] == ["arb_Latn", "min_Arab", "sat_Olck"]
    assert not receipt["model_preparation"]["unsupported_language_codes_substituted"]


def test_billsum_delivery_binds_native_model_original_fields_and_fixed_consumer():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["BILLSUM"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    receipt = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert receipt["source_checkbox_eligible"]
    assert receipt["model"] == "google/pegasus-billsum"
    assert receipt["model_input_fields"] == ["text"]
    assert receipt["original_fields_preserved"] == ["text", "summary", "title"]
    assert not receipt["reference_summary_or_title_in_model_input"]
    assert receipt["primary_original_indices_per_evaluation_split"] == {
        "test": [0, 1, 2],
        "ca_test": [0, 1, 2],
    }
    assert receipt["cpu_ci_conclusion"] == receipt["root_ci_conclusion"] == "success"
    assert receipt["public_generation_main_ast_identical_to_fixed_entry"]
    assert receipt["fixed_rescore_new_model_calls"] == 0
    assert all(
        item["original_indices"] == [3, 4, 5] and item["model_calls"] == 0
        for item in receipt["fixed_entry_rescore"]
    )
    fixed = receipt["fixed_consumer"]
    assert (
        fixed["fixed_ci_wheel_claim"] and fixed["installed_payload_files_equal"] == 39
    )
    assert fixed["public_index_rows"] == fixed["research_direct_prepare_rows"] == 23455
    assert (
        fixed["research_import_rows"] == fixed["public_and_research_roundtrips"] == 18
    )
    assert fixed["research_source_files_unchanged"] == 40
    assert not fixed["research_observation_or_analysis_migrated"]
    assert not receipt["research_default_dependency_lock_updated"]
    assert not receipt["full_evaluation_suites_inference_claim"]
    assert not receipt["full_raw_archive_field_preservation_claim"]
    assert not receipt["original_paper_preprocessing_or_training_reproduction_claim"]


def test_multinews_delivery_preserves_native_pairs_and_fixed_consumer():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["MULTINEWS"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    handoff = json.loads(
        (ROOT / row["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
    )
    native = json.loads(
        (ROOT / row["public_native_and_rescore_receipt"]).read_text("utf-8")
    )
    original = json.loads((ROOT / row["native_sample_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    assert delivery["original_author_loader_fields_equal_for_all_rows"]
    assert delivery["original_NEWLINE_CHAR_restoration_preserved"]
    assert not delivery["full5622_test_inference_claim"]
    assert not delivery["full_document_long_context_inference_claim"]
    assert (
        native["public_primary_rescore"]["rouge"]
        == original["original_hf_score"]["rouge"]
    )
    assert native["original_blank_document_edge_score"]["original_indices"] == [4736]
    assert native["original_blank_document_edge_score"]["scoring_complete"]
    assert handoff["installed_payload_files_equal"] == 39
    assert (
        handoff["original_rows_preserved"]
        == handoff["public_index_rows"]
        == handoff["research_direct_prepare_rows"]
        == 56216
    )
    assert handoff["research_source_files_equal_installed"] == 40
    assert handoff["cpu_ci_conclusion"] == handoff["root_ci_conclusion"] == "success"
    assert not handoff["research_observation_or_analysis_migrated"]


def test_xsum_original_samples_preserve_blank_articles_and_native_scoring():
    receipt = json.loads(
        (DOCS / "validation/xsum-original-native-and-rescore-20261025.json").read_text(
            "utf-8"
        )
    )
    original = receipt["original_hf_recipe_scoring"]
    assert receipt["primary_frozen_original_indices"] == [0, 1, 2]
    assert original["model_calls"] == 0 and original["native_responses_reused"] == 3
    assert original["estimated_full_native_test_seconds"] > 1800
    assert (
        original["original_postprocess_text_executed"]
        and original["metric_source_unmodified"]
    )
    assert receipt["public_primary_rescore"]["rouge"] == original["rouge"]
    assert receipt["original_blank_document_rows_per_split"] == {
        "train": 28,
        "validation": 5,
        "test": 1,
    }
    assert receipt["original_blank_document_edge_score"]["original_indices"] == [11155]
    assert receipt["original_blank_document_edge_score"]["scoring_complete"]
    assert not receipt["full11334_test_inference_claim"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["XSUM"]
    if row["checked"]:
        delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
        handoff = json.loads(
            (ROOT / row["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
        )
        assert delivery["source_checkbox_eligible"]
        assert not delivery["full11334_test_inference_claim"]
        assert (
            handoff["original_rows_preserved"]
            == handoff["public_index_rows"]
            == handoff["research_direct_prepare_rows"]
            == 226711
        )
        assert handoff["installed_payload_files_equal"] == 38
        assert (
            handoff["cpu_ci_conclusion"] == handoff["root_ci_conclusion"] == "success"
        )


def test_validation_receipts_have_no_duplicate_json_keys():
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate receipt key: {key}")
            result[key] = value
        return result

    for path in (DOCS / "validation").glob("*.json"):
        json.loads(path.read_text("utf-8"), object_pairs_hook=unique_object)


def test_xlsum_delivery_closes_all_frozen_language_scopes_without_reruns():
    delivery = json.loads(
        (DOCS / "validation/xlsum-native-delivery-20261025.json").read_text("utf-8")
    )
    audit = json.loads(
        (ROOT / delivery["final_frozen_selection_audit"]).read_text("utf-8")
    )
    fixed = json.loads(
        (ROOT / delivery["fixed_ci_research_handoff"]).read_text("utf-8")
    )
    assert delivery["source_checkbox_eligible"]
    assert (
        audit["status"]
        == "ALL45_FROZEN_PRIMARY_SELECTION_NATIVE_RESULTS_AND_AUTHOR_SCORING_AUDITED"
    )
    assert audit["language_variants"] == len(audit["checks"]) == 45
    assert (
        audit["primary_original_positions"]
        == delivery["native_primary_original_positions"]
        == 6173
    )
    assert audit["full_variant_positions"] == 6068 and audit["full_variants"] == 10
    assert audit["other_fixed3_variants"] == 35
    assert (
        sum(
            row[
                "all_selected_positions_original_fields_encoder_ids_generated_tokens_decoding_verified"
            ]
            for row in audit["checks"]
        )
        == 6173
    )
    assert all(
        row["scoring_verified"] and row["author_rouge_reused"]
        for row in audit["checks"]
    )
    assert audit["model_calls"] == audit["new_scoring_runs"] == 0
    assert fixed["data_commit"] == delivery["data_commit"]
    assert fixed["installed_payload_files_equal"] == 35
    assert (
        fixed["prior_full_original_rows_audit_reused"]
        == delivery["original_data_rows_preserved"]
        == 114198
    )
    assert not delivery["all_original_rows_model_inference_claim"]
    assert delivery["not_long_context_serving_validation"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    assert ledger["delivery_inventory"]["XLSUM"]["checked"]


def test_librispeech_native_samples_and_raw_binding_are_not_public_delivery():
    receipt = json.loads(
        (
            DOCS
            / "validation/librispeech-native-samples-and-candidate-binding-20261025.json"
        ).read_text("utf-8")
    )
    assert receipt["model_calls"] == 8 and receipt["device"] == "npu:1"
    assert receipt["model_parameters_verified_on_npu"]
    assert receipt["native_encoder_decoder_outputs_verified_on_npu"]
    assert not receipt["notebook_30_second_clipping_used"]
    assert receipt["client_isolated_pip_check"] == "No broken requirements found."
    for split in receipt["splits"]:
        assert split["selection_mode"] == "PRESELECTED_ORIGINAL_FIRST3_PLUS_LONGEST"
        assert split["estimated_full_warm_execution_seconds"] > 1800
        assert split["selected_original_positions"] == sorted(
            split["preselected_original_indices_before_model_outputs"]
        )
        assert (
            split["completed_original_positions"]
            == split["selected_original_positions"]
        )
        assert split["longest_audio_seconds"] > 30
        assert split["longest_original_audio_complete_input_verified"]
    raw = receipt["raw_asset_binding_audit"]
    assert (
        sum(
            row["all_original_utterances_and_transcripts_preserved"]
            for row in raw["checks"]
        )
        == 5559
    )
    assert (
        sum(
            row["native_selected_audio_bindings_byte_identical"]
            for row in raw["checks"]
        )
        == 8
    )
    assert raw["model_calls"] == raw["new_scoring_runs"] == 0
    assert not receipt["source_checkbox_eligible"]
    assert not receipt["public_native_entry_complete"]
    assert not receipt["fixed_ci_artifact_verified"]
    assert not receipt["research_consumption_verified"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["LIBRISPEECH"]
    if row["checked"]:
        delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
        handoff = json.loads(
            (ROOT / row["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
        )
        assert delivery["source_checkbox_eligible"]
        assert not delivery["full_5559_utterance_inference_claim"]
        assert not delivery["streaming_asr_or_live_conversation_claim"]
        assert handoff["original_rows_preserved"] == 5559
        assert handoff["installed_payload_files_equal"] == 38
        assert handoff["research_source_files_equal_installed"] == 40
        assert (
            handoff["cpu_ci_conclusion"] == handoff["root_ci_conclusion"] == "success"
        )
        assert delivery["primary_frozen_native_selection"] == {
            scope["split"]: scope["selected_original_positions"]
            for scope in receipt["splits"]
        }


def test_stsb_delivery_closes_fixed_installation_and_native_evaluation():
    receipt = json.loads(
        (DOCS / "validation/stsb-native-delivery-20261025.json").read_text("utf-8")
    )
    fixed = json.loads(
        (ROOT / receipt["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
    )
    native = json.loads(
        (ROOT / receipt["native_execution_and_rescore_receipt"]).read_text("utf-8")
    )
    assert receipt["source_checkbox_eligible"]
    assert receipt["full_native_evaluation_pairs"] == {"dev": 1500, "test": 1379}
    assert receipt["original_train_rows_retained_not_evaluated"] == 5749
    assert fixed["data_commit"] == receipt["data_commit"]
    assert fixed["root_ci_conclusion"] == fixed["cpu_ci_conclusion"] == "success"
    assert fixed["installed_payload_files_equal"] == 37
    assert (
        fixed["fixed_provider_original_rows"]
        == fixed["research_direct_prepare_rows"]
        == 8628
    )
    assert fixed["public_roundtrips_per_split"] == {"train": 6, "dev": 6, "test": 6}
    assert fixed["research_index_import_per_split"] == {"train": 3, "dev": 3, "test": 3}
    assert fixed["installed_provider_original_scoring_reverified_without_model_calls"]
    assert fixed["model_calls"] == fixed["new_scoring_runs"] == 0
    assert receipt["public_runner_sha256"] == native["public_runner_sha256"]
    assert not receipt["continuous_reference_labels_exposed_to_model"]
    assert not receipt["research_observation_or_analysis_migrated"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    assert ledger["delivery_inventory"]["STSB"]["checked"]


def test_stsb_native_original_pairs_and_candidate_consumption_do_not_claim_fixed_artifact():
    native = json.loads(
        (DOCS / "validation/stsb-original-native-and-rescore-20261025.json").read_text(
            "utf-8"
        )
    )
    run = native["original_native_run"]
    assert run["status"] == "ORIGINAL_FULL_SPLITS_EXECUTED_AND_SCORED"
    assert (
        run["model_parameters_verified_on_npu"]
        and run["model_forward_embeddings_verified_on_npu"]
    )
    assert run["encoder_native_max_length"] == 384
    assert [(row["split"], row["original_pairs"]) for row in run["splits"]] == [
        ("dev", 1500),
        ("test", 1379),
    ]
    assert len(run["encoding_calls"]) == 4
    assert native["original_metrics_equal_cached_rescore"]
    assert not native["cached_rescore_model_execution"]
    wheel = native["publisher_wheel_source_audit"]
    assert len(wheel["installed_source_files"]) == 6
    assert not wheel["installed_source_modified"] and wheel["model_calls"] == 0
    assert all(
        row["normalized_only_line_endings_equals_git_source"]
        for row in wheel["installed_source_files"]
    )
    consumer = json.loads(
        (DOCS / "validation/stsb-candidate-consumer-20261025.json").read_text("utf-8")
    )
    assert (
        consumer["all_original_rows_preserved"]
        == consumer["public_index_rows"]
        == consumer["research_direct_prepare_rows"]
        == 8628
    )
    assert consumer["public_first_tail_roundtrips_per_split"] == {
        "train": 6,
        "dev": 6,
        "test": 6,
    }
    assert consumer["research_index_import_per_split"] == {
        "train": 3,
        "dev": 3,
        "test": 3,
    }
    for evidence in [native, consumer]:
        assert (
            not evidence["source_checkbox_eligible"]
            and not evidence["fixed_ci_artifact_verified"]
        )
        assert not evidence["research_observation_or_analysis_migrated"]


def test_beir_delivery_closes_original_execution_fixed_artifact_and_consumer():
    receipt = json.loads(
        (DOCS / "validation/beir-native-delivery-20261025.json").read_text("utf-8")
    )
    fixed = json.loads((ROOT / receipt["fixed_ci_research_handoff"]).read_text("utf-8"))
    assert receipt["source_checkbox_eligible"]
    assert fixed["status"] == "FIXED_CI_PROVIDER_AND_RESEARCH_CONSUMPTION_VERIFIED"
    assert fixed["data_commit"] == receipt["data_commit"]
    assert fixed["installed_payload_files_equal"] == 36
    assert fixed["research_source_files_equal_installed"] == 40
    assert len(fixed["checks"]) == 9
    assert sum(c["public_first_tail_roundtrip"] for c in fixed["checks"]) == 54
    assert sum(c["research_index_import"] for c in fixed["checks"]) == 27
    assert sum(c["research_direct_prepare_resolve"] for c in fixed["checks"]) == 27
    assert fixed["model_calls"] == fixed["new_scoring_runs"] == 0
    assert receipt["unique_original_test_query_positions"] == 623
    assert receipt["rerank_candidate_pairs"] == 62300
    assert (
        receipt["newly_covered_unique_query_positions_relative_to_prior_native_run"]
        == 0
    )
    assert receipt["environment"]["backend"] == "torch_npu"
    assert not receipt["environment"]["serving_environment_modified"]
    import hashlib

    actual = hashlib.sha256(
        (ROOT / receipt["public_runner"]).read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert actual == receipt["public_runner_sha256"]
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    for run in receipt["final_public_entry_native_protocols"]:
        assert run["mode"] == "run" and run["generation_performed_this_run"]
        assert run["status"] == "ORIGINAL_FULL_TEST_EXECUTED_AND_SCORED"
        assert run["device"] == "npu:1"
        assert run["model_parameters_and_embeddings_verified_on_npu"]
        assert len(run["protocols"]) == 2
        assert (
            run["original_test_queries"]
            == {"BEIR_SCIFACT": 300, "BEIR_NFCORPUS": 323}[run["source_id"]]
        )
        assert ledger["delivery_inventory"][run["source_id"]]["checked"]
    assert not receipt["research_observation_or_analysis_migrated"]
    assert not receipt["engine_performance_or_scientific_conclusion_claim"]


def test_beir_full_native_and_candidate_consumer_are_not_fixed_ci_completion():
    native = json.loads(
        (DOCS / "validation/beir-original-full-native-20261025.json").read_text("utf-8")
    )
    assert (
        len(native["actual_native_protocols"])
        == len(native["public_rescore_protocols"])
        == 4
    )
    assert native["device"] == "npu:1"
    for protocol in native["actual_native_protocols"]:
        assert protocol["status"] == "ORIGINAL_FULL_TEST_EXECUTED_AND_SCORED"
        match = next(
            row
            for row in native["public_rescore_protocols"]
            if row["source_id"] == protocol["source_id"]
            and row["protocol"] == protocol["protocol"]
        )
        assert (
            match["metrics"] == protocol["metrics"] and match["mrr"] == protocol["mrr"]
        )
        assert not match["generation_performed_this_run"]
    consumer = json.loads(
        (DOCS / "validation/beir-candidate-consumer-20261025.json").read_text("utf-8")
    )
    assert len(consumer["sources"]) == 2 and consumer["model_calls"] == 0
    assert sum(len(row["assets"]) for row in consumer["sources"]) == 9
    for source in consumer["sources"]:
        for asset in source["assets"]:
            assert asset["all_original_rows_preserved"] == asset["public_index_rows"]
            assert (
                asset["research_index_import_rows"]
                == asset["research_direct_prepare_resolve_rows"]
                == 3
            )
    for evidence in [native, consumer]:
        assert not evidence["source_checkbox_eligible"]
        assert not evidence["fixed_ci_artifact_verified"]
        assert not evidence["research_observation_or_analysis_migrated"]


def test_xlsum_fixed_ci_consumer_and_clean_environment_are_separate_from_full_delivery():
    fixed = json.loads(
        (DOCS / "validation/xlsum-fixed-ci-research-handoff-20261025.json").read_text(
            "utf-8"
        )
    )
    assert fixed["status"] == "FIXED_CI_PROVIDER_AND_RESEARCH_CONSUMPTION_VERIFIED"
    expected_wheel = "4f7a2be0b0a39e02fd586d730b5ff40550bfa74ee8a9e86c00c36d66d0886cf7"  # pragma: allowlist secret - public CI software artifact digest
    assert fixed["data_wheel_sha256"] == expected_wheel
    assert fixed["installed_payload_files_equal"] == 35
    assert fixed["research_source_files_equal_installed"] == 40
    assert fixed["language_variants"] == len(fixed["checks"]) == 45
    assert fixed["prior_full_original_rows_audit_reused"] == 114198
    assert sum(c["public_first_tail_roundtrip"] for c in fixed["checks"]) == 270
    assert sum(c["research_direct_prepare_resolve"] for c in fixed["checks"]) == 135
    assert fixed["model_calls"] == fixed["scoring_runs"] == 0
    assert not fixed["research_observation_or_analysis_migrated"]
    assert not fixed["research_default_dependency_lock_updated"]
    assert not fixed["source_checkbox_eligible"]
    environment = json.loads(
        (DOCS / "validation/xlsum-clean-client-environment-20261025.json").read_text(
            "utf-8"
        )
    )
    assert environment["pip_check"] == "No broken requirements found."
    assert environment["original_offset_selected_before_output"] == 3
    assert environment["model_calls"] == 1
    assert environment["model_parameters_all_npu_fp32"]
    assert environment["output_device"] == "npu:0"
    assert environment["encoder_tokens"] == 512 and environment["decoder_tokens"] <= 84
    assert environment["original_tokenizer_transformers"] == "4.11.3"
    assert not environment["serving_engine_imported"]
    assert not environment["global_serving_environment_modified"]
    assert not environment["source_checkbox_eligible"]


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
        receipt["data_commit"]
        == "39556148d18a30bb1f6d3ec04cba663ab8d67db6"  # pragma: allowlist secret - public CI commit
    )
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
        current = ledger["delivery_inventory"][sid]
        if sid == "HUMANEVAL":
            native = json.loads(
                (ROOT / current["native_functional_scoring_receipt"]).read_text("utf-8")
            )
            assert native["generated_code_executed"]
            assert native["isolation_probe"]["status"] == "PASSED_ON_REPLACEMENT_SERVER"
        elif sid == "DS1000":
            delivery = json.loads(
                (ROOT / current["delivery_receipt"]).read_text("utf-8")
            )
            assert delivery["source_checkbox_eligible"]
            assert delivery["six_other_libraries"][
                "author_16_worker_concurrency_preflight_passed"
            ]
        else:
            assert not current["checked"]
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
    current = ledger["delivery_inventory"]["CHARTQA"]
    assert current["original_processor_preflight"]
    # Frozen CPU preparation is not runtime proof. Current delivery must have
    # independent real native generation, original scoring and consumer evidence.
    assert current["checked"]
    delivery = json.loads((ROOT / current["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    native = delivery["native_validation"]
    assert [batch["positions"] for batch in native["batches"]] == [
        [0, 1, 2],
        [1250, 1251, 1252],
    ]
    assert all(batch["stop"] == 3 for batch in native["batches"])
    assert native["both_saved_batches_public_verifier_passed"]
    assert native["original_encoder_and_scorer_unmodified"]
    assert native["verification_new_model_calls"] == native["paid_judge_calls"] == 0
    assert not native["full_test_inference_or_performance_claim"]
    assert (
        delivery["provider_and_research"]["public_and_fixed_research_roundtrips"] == 6
    )
    assert delivery["provider_and_research"]["full_original_index_rows"] == 2500


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
