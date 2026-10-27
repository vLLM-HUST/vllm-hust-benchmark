"""Keep dataset checkboxes aligned with source identities and actual evidence."""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scienceworld_delivery_requires_native_scores_and_actual_ci_consumer():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["SCIENCEWORLD"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    assert delivery["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert delivery["original_assets"]["original_rows"] == 7207
    assert delivery["native_profile"]["selected_original_episodes"] == 9
    assert delivery["native_profile"]["original_model_call_attempts"] == 464
    assert delivery["native_profile"]["max_generation_tokens"] == 50
    assert delivery["native_profile"]["environment_step_limit"] == 100
    assert delivery["native_profile"]["valid_model_outputs_rerun"] == 0
    assert len(delivery["saved_score_consistency"]["episodes"]) == 9
    assert delivery["saved_score_consistency"]["model_calls"] == 0
    actual = delivery["exact_ci_and_research"]
    assert actual["installed_payloads"] == 51
    assert actual["reused_actual_ci_research_readbacks"] == 9
    assert actual["reused_original_full_fields_and_index"] == 7207
    assert actual["new_native_inference_calls"] == 0
    assert (
        delivery["software_ci"]["tests"]
        == delivery["software_ci"]["lint"]
        == delivery["software_ci"]["cpu_correctness"]
        == "success"
    )
    assert delivery["strict_environment_replay"]["status"] == "FAILED_NOT_WAIVED"
    assert not delivery["quality_claim"]


def test_azure_lmm_completion_is_only_a_verified_trace_asset():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["AZURE_LMM_2025"]
    if not row["checked"]:
        return
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    assert delivery["delivery_kind"] == "metadata_asset_not_native_model_benchmark"
    assert delivery["original_asset"]["rows_all_fields_verified_and_indexed"] == 1000000
    assert (
        delivery["exact_ci_and_research"]["installed_payloads_byte_equal_ci_wheel"]
        == 47
    )
    assert delivery["exact_ci_and_research"][
        "all_ci_payloads_byte_equal_candidate_full_audit"
    ]
    assert delivery["exact_ci_and_research"]["fixed_actual_ci_research_roundtrips"] == 6
    assert (
        delivery["model_calls"] == delivery["synthetic_prompts_or_media_created"] == 0
    )
    assert not delivery["original_request_replay_claim"]
    assert not delivery["visual_benchmark_or_quality_claim"]


def test_repobench_completion_requires_native_and_fixed_original_scoring():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["REPOBENCH"]
    if not row["checked"]:
        return
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    assert delivery["source_checkbox_eligible"]
    profile = delivery["native_profile"]
    assert (
        profile["selected_tasks"]
        == profile["saved_and_independently_audited_tasks"]
        == 90
    )
    assert profile["fixed_strata"] == 30 and profile["per_stratum"] == 3
    assert (
        profile["valid_generations_rerun"] == profile["new_model_calls_in_audit"] == 0
    )
    assert profile["saved_token_decoding_original_next_line_and_ground_truth_verified"]
    assert (
        delivery["public_and_research"]["full_prior_fields_and_index_rows_reused"]
        == 49684
    )
    assert delivery["public_and_research"]["fixed_research_roundtrips_reused"] == 60
    scoring = delivery["scoring"]
    assert (
        scoring["python_hash_seed"] == 0 and scoring["two_repeat_frozen_reports_equal"]
    )
    assert (
        scoring["legacy_unseeded_reports_retained"]
        and not scoring["quality_acceptance_claim"]
    )
    assert (
        scoring["python"]["exact_match_percent"]
        == scoring["java"]["exact_match_percent"]
        == 0.0
    )


def test_infinitebench_does_not_inherit_unrelated_qrecc_storage_blocker():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    inventory = ledger["delivery_inventory"]
    assert "QReCC" not in (inventory["INFINITEBENCH"]["blocker"] or "")
    assert inventory["QRECC"]["checked"] is False


def test_gpqa_fixed_artifact_and_native_main_do_not_hide_skipped_task():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["GPQA"]
    fixed = json.loads(
        (ROOT / row["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
    )
    main = json.loads((ROOT / row["main_native_audit"]).read_text("utf-8"))
    publisher = json.loads(
        (ROOT / row["publisher_py39_verification_receipt"]).read_text("utf-8")
    )
    assert fixed["final_ci_artifact_verified"]
    assert fixed["installed_payloads_equal"] == 34
    assert fixed["raw_records_all_fields_and_original_permutation_verified"] == 1192
    assert (
        fixed["research_index_import_rows"]
        == fixed["research_direct_prepare_resolve_rows"]
        == 9
    )
    assert main["completed_generations"] == 447
    assert (
        main["actual_new_model_requests"]
        + main["byte_equivalent_previous_requests_reused"]
        == 447
    )
    assert main["publisher_skipped_original_offsets"] == [69]
    assert not main["source_checkbox_eligible"]
    assert publisher["pandas"] == "1.5.2" and publisher["new_model_calls"] == 0
    assert publisher["python"].startswith("3.9.")
    if row["checked"]:
        delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
        assert delivery["all_original_question_positions_have_native_model_response"]
        assert delivery["source_checkbox_eligible"]


def test_ruler_delivery_requires_native_metrics_and_final_artifact_handoff():
    folder = ROOT / "docs/workload-data"
    ledger = json.loads((folder / "dataset-validation-status.json").read_text("utf-8"))
    row = ledger["delivery_inventory"]["RULER"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    handoff = json.loads(
        (ROOT / row["fixed_ci_research_handoff_receipt"]).read_text("utf-8")
    )
    assert delivery["native_requests_completed_and_audited"] == 2530
    assert (
        delivery["original_task_types_verified"] == handoff["original_task_types"] == 13
    )
    assert handoff["raw_records_reconstructed_equal_total"] == 7500
    assert handoff["public_roundtrip_rows_total"] == 90
    assert handoff["research_index_import_rows_total"] == 45
    assert handoff["research_direct_prepare_resolve_rows_total"] == 45
    assert not handoff["new_candidate_not_final_ci_artifact"]
    assert handoff["model_calls"] == 0
    assert not delivery["unmodified_publisher_qa2_generator_claim"]
    assert not delivery["full_original_length_grid_validated"]
    assert not delivery["research_observation_or_analysis_migrated"]
    assert not delivery["raw_payloads_published"]


def test_longwriter_delivery_preserves_corpus_and_non_scoring_boundary():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["LONGWRITER"]
    assert row["checked"] and row["status"] == "AVAILABLE"
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    native = json.loads((ROOT / row["range_runner_native_audit"]).read_text("utf-8"))
    handoff = json.loads(
        (ROOT / row["research_exact_artifact_handoff_receipt"]).read_text("utf-8")
    )
    assert delivery["fixed_original_indices"] == [0, 1, 2, 5997, 5998, 5999]
    assert native["verified_unique_original_tasks"] == 6
    assert native["previous_unique_original_tasks_reused"] == 3
    assert [sample["original_row_index"] for sample in native["samples"]] == [
        5997,
        5998,
        5999,
    ]
    assert native["generation_config"] == {"temperature": 0.5, "max_tokens": 32768}
    assert native["new_range_runner_native_validated"]
    assert handoff["complete_inputs_and_independent_references_equal"]
    assert handoff["research_source_index_import_rows"] == 3
    assert handoff["research_direct_prepare_resolve_rows"] == 3
    assert not delivery["semantic_or_length_quality_claim"]
    assert not delivery["is_longbench_write_evaluation"]
    assert not delivery["full_corpus_native_completed"]
    assert not native["request_wire_capture_claim"]


def test_structeval_checkpoint_counts_only_audited_terminal_protocols():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["STRUCTEVAL"]
    checkpoint = json.loads(
        (ROOT / row["completed_protocols_checkpoint_chain"][0]).read_text("utf-8")
    )
    assert not row["checked"]
    assert checkpoint["verified_unique_native_tasks"] == 480
    assert checkpoint["previous_unique_original_tasks_reused"] == 339
    assert checkpoint["new_unique_original_tasks_audited"] == 141
    assert {item["protocol"] for item in checkpoint["protocols"]} == {
        "JSON-to-YAML",
        "TOML-to-JSON",
        "TOML-to-YAML",
    }
    assert all(
        item["new_original_tasks"] == 47
        and item["previous_original_tasks_reused"] == 3
        and item["all_original_protocol_tasks"] == 50
        and item["official_saved_output_rescore_equal"]
        for item in checkpoint["protocols"]
    )
    assert checkpoint["running_sibling_jobs_excluded"]
    assert checkpoint["audit_model_calls"] == 0
    assert not checkpoint["full_source_completed"]
    assert not checkpoint["original_prompts_output_budget_and_scoring_changed"]


def test_qasper_delivery_separates_full_native_run_and_sampled_research_handoff():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["LONGBENCH"]
    assert row["checked"] and row["blocker"] is None
    delivery = json.loads((ROOT / row["delivery_receipt"]).read_text("utf-8"))
    native = json.loads(
        (
            directory
            / "validation"
            / delivery["native_execution_and_official_scoring_receipt"]
        ).read_text("utf-8")
    )
    handoff = json.loads(
        (
            directory
            / "validation"
            / delivery["exact_installed_research_handoff_receipt"]
        ).read_text("utf-8")
    )
    assert native["original_unique_tasks"] == 200
    assert native["historical_native_tasks_reused"] == 3
    assert native["new_native_tasks"] == 197
    assert native["all_saved_response_hashes_and_official_scores_recalculated"]
    assert handoff["installed_provider_payload_files_equal"] == 32
    assert handoff["research_installed_python_files_equal"] == 40
    assert handoff["fixed_original_indices"] == [0, 1, 2]
    assert handoff["public_prepare_resolve_rows"] == 3
    assert handoff["research_source_index_import_rows"] == 3
    assert handoff["research_direct_prepare_resolve_rows"] == 3
    assert handoff["original_inputs_references_source_identity_and_relations_equal"]
    assert handoff["new_native_tasks_or_model_calls"] == 0
    assert not delivery["all200_research_handoff_claim"]
    assert not delivery["all_longbench_other_subsuites_completed"]
    assert not delivery["original_qasper_full_corpus_completed"]
    assert not handoff["research_features_or_observation_migrated_into_benchmark"]


def test_all_source_receipts_are_local_json_and_cover_the_referenced_source():
    """Check single/multi-source receipts without treating readers as task runs."""
    rows = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["sources"]
    for row in rows:
        if not row.get("receipt"):
            continue
        path = (ROOT / row["receipt"]).resolve()
        assert path.is_relative_to((ROOT / "docs/workload-data/validation").resolve())
        receipt = json.loads(path.read_text("utf-8"))
        ids = set(receipt.get("source_ids", []))
        if receipt.get("source_id"):
            ids.add(receipt["source_id"])
        ids.update(
            item["source_id"]
            for item in receipt.get("sources", [])
            if isinstance(item, dict) and "source_id" in item
        )
        assert row["source_id"] in ids


def test_status_summary_does_not_erase_native_subsets_or_upgrade_generation():
    summary = (ROOT / "docs/workload-data/VALIDATION_STATUS.zh-CN.md").read_text(
        "utf-8"
    )
    assert "WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md" in summary
    for receipt in (
        "agentbench-alfworld-public-handoff-npu-20261023.json",
        "tau2-reference-handoff-npu-20261022.json",
    ):
        assert receipt in summary
        evidence = json.loads(
            (ROOT / "docs/workload-data/validation" / receipt).read_text("utf-8")
        )
        assert evidence["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert "GAIA官方原版闭环仍未完成" in summary
    assert "缺隔离功能评分" in summary
    assert "零新模型/原生任务" in summary
    assert "AgentBench、GAIA、tau2官方执行器仍待逐环境接入" not in summary


def _script_hashes_for_line_endings(payload):
    """Accept only Git LF/CRLF transport variants, never alter old receipts."""
    lf = payload.replace(b"\r\n", b"\n")
    return {
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest(),
    }


def test_line_ending_compatibility_does_not_accept_changed_script_content():
    lf = b"print('fixture')\n"
    crlf = lf.replace(b"\n", b"\r\n")
    assert _script_hashes_for_line_endings(lf) == _script_hashes_for_line_endings(crlf)
    assert hashlib.sha256(lf).hexdigest() in _script_hashes_for_line_endings(crlf)
    assert hashlib.sha256(b"print('changed')\n").hexdigest() not in (
        _script_hashes_for_line_endings(lf)
    )


def test_partial_history_execution_and_retrieval_preparation_stay_distinct():
    directory = ROOT / "docs/workload-data"
    sources = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["sources"]
    rows = {row["source_id"]: row for row in sources}
    history = rows["OASST1"]
    receipt = json.loads((ROOT / history["receipt"]).read_text(encoding="utf-8"))
    assert history["selected_samples"] == history["completed_samples"] == [23, 28, 68]
    assert history["selected_samples"] == [
        row["row_index"] for row in receipt["samples"]
    ]
    assert history["server_execution"] == "COMPLETED_ORIGINAL_HISTORY_CONTINUATIONS"
    assert all(row["finish_reason"] == "stop" for row in receipt["samples"])
    assert receipt["quality_scored"] is False
    assert receipt["live_episode_validated"] is False
    assert history["checked"] is False
    retrieval = rows["QRECC"]
    assert retrieval["acquisition"] == "VERIFIED_ORIGINAL_TEST_ARCHIVE_AND_MEMBER"
    assert retrieval["read_adapter"] == "VERIFIED_INSTALLED_CLI_3_SAMPLES"
    assert retrieval["server_execution"] == retrieval["scoring"] == "NOT_RUN"
    assert retrieval["completed_samples"] == []
    assert retrieval["checked"] is False


def test_humaneval_generation_without_functional_scoring_stays_unchecked():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    row = next(r for r in ledger["sources"] if r["source_id"] == "HUMANEVAL")
    receipt = json.loads((ROOT / row["receipt"]).read_text(encoding="utf-8"))
    assert row["checked"] is False
    assert row["completed_samples"] == []
    assert row["scoring"] == "BLOCKED_NO_VERIFIED_SANDBOX"
    assert receipt["complete_task_validation"] is False
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["sandbox_probe"]["generated_code_executed"] is False
    assert len(receipt["samples"]) == 3
    # Historical execution hashed actual LF bytes. A Windows Git archive may
    # export CRLF; this check proves content continuity, not byte-identical replay.
    assert receipt["helper_sha256"] in _script_hashes_for_line_endings(
        (ROOT / "scripts/run_humaneval_original_generation.py").read_bytes()
    )


def test_checklist_covers_registered_sources_once():
    directory = ROOT / "docs" / "workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = ledger["sources"]
    assert len(rows) == len({row["source_id"] for row in rows})
    # The100-row sources array is a frozen historical stage, not a barrier to
    # implementing the already agreed118-source inventory incrementally.
    assert len(rows) == 100
    assert {row["source_id"] for row in rows} <= set(catalog["sources"])
    assert set(catalog["sources"]) <= set(ledger["delivery_inventory"])
    markdown = (directory / "DATASET_CHECKLIST.zh-CN.md").read_text(encoding="utf-8")
    table_rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in markdown.splitlines()
        if line.startswith("|")
    ]
    # Frozen sources preserves old read/prepare stages. Current delivery boxes
    # must follow the unique authoritative inventory, not that historical array.
    inventory = ledger["delivery_inventory"]
    current_rows = [
        cells for cells in table_rows if len(cells) == 3 and cells[0] in inventory
    ]
    assert len(current_rows) == len(inventory) == 118
    assert len({cells[0] for cells in current_rows}) == 118
    for source_id, item in inventory.items():
        (current,) = [cells for cells in current_rows if cells[0] == source_id]
        marker = (
            "—"
            if item["status"] == "CLASSIFICATION_ONLY"
            else "[x]"
            if item["checked"]
            else "[ ]"
        )
        assert current[1].startswith(marker)


def test_summary_counts_match_actual_checkboxes():
    directory = ROOT / "docs/workload-data"
    rows = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["sources"]
    table = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in (directory / "DATASET_CHECKLIST.zh-CN.md")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("|")
    ]
    for scope, title in (
        ("public", "公开候选来源"),
        ("enterprise_private", "企业私有来源"),
    ):
        selected = [row for row in rows if row["scope"] == scope]
        assert [
            title,
            str(len(selected)),
            str(sum(row["checked"] for row in selected)),
        ] in table


def test_checked_sources_require_execution_and_scoring_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    for row in ledger["sources"]:
        if not row["checked"]:
            continue
        assert row["acquisition"] == "VERIFIED"
        assert row["read_adapter"] == "VERIFIED"
        assert row["server_execution"] == "COMPLETED"
        assert row["scoring"] in {"COMPLETED", "NON_SCORING_INTEGRITY_VERIFIED"}
        assert (
            len(set(row["selected_samples"]))
            >= ledger["completion_rule"]["minimum_preselected_samples"]
        )
        assert set(row["completed_samples"]) == set(row["selected_samples"])
        assert row["receipt"]
        receipt = (ROOT / row["receipt"]).resolve()
        assert receipt.is_relative_to(ROOT.resolve())
        assert receipt.is_file()
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        assert evidence["source_id"] == row["source_id"]
        if row["scoring"] == "COMPLETED":
            assert evidence["status"] == "END_TO_END_EXECUTED_AND_SCORED"
        else:
            assert evidence["status"] in {
                "REQUESTS_EXECUTED_AND_INTEGRITY_VALIDATED_NO_QUALITY_ORACLE",
                "ORIGINAL_FIELDS_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE",
                "ORIGINAL_MESSAGES_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE",
            }
            assert evidence["quality_claim"] is False
            assert evidence["scoring"] == "NON_SCORING_INTEGRITY_VERIFIED"
            assert len(evidence["samples"]) == 3
            assert [s["record_id"] for s in evidence["samples"]] == evidence[
                "selected_samples"
            ]
            assert all(
                s["response_validation"]["status"] == "RESPONSE_INTEGRITY_VERIFIED"
                for s in evidence["samples"]
            )
        assert evidence["selected_samples"] == row["selected_samples"]
        assert evidence["server_inventory"]["backend"] == "vllm-ascend"


def test_arc_task_count_and_protocol_are_not_chat_or_full_suite_claims():
    evidence = json.loads(
        (
            ROOT / "docs/workload-data/validation/arc-challenge-npu-20261022.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["task_count"] == 3
    assert evidence["candidate_likelihood_requests"] == 12
    assert (
        evidence["model_answer_correct_count"]
        + evidence["model_answer_incorrect_count"]
        == 3
    )

    assert (
        evidence["protocol"]
        == "original multiple-choice loglikelihood; no chat conversion"
    )
    assert evidence["subsuite"] == "ARC-Challenge"
    # Historical receipt pins the helper that ran ARC-Challenge before the
    # shared ARC-Easy support was added.
    assert (
        evidence["helper_sha256"]
        == "f45dbe69e17309fcd74dd49fc3b74ce8dc6d7921a4c46c3268c1dcc593088568"  # pragma: allowlist secret - public historical runner file SHA256
    )


def test_enterprise_checks_are_not_quality_scores_and_preserve_full_inputs():
    directory = ROOT / "docs/workload-data/validation"
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in directory.glob("paio-*-npu-20261022.json")
    ]
    assert len(records) == 8
    assert sum(r["sample_count"] for r in records) == 24
    helper = hashlib.sha256(
        (ROOT / "scripts/run_source_request_samples.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest()
    checker = hashlib.sha256(
        (ROOT / "src/vllm_hust_benchmark/workload_response_validation.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest()
    for record in records:
        assert record["quality_claim"] is False
        assert record["production_trace_representativity"] == "NOT_ESTABLISHED"
        assert record["source_asset"]["matches_catalog_file_sha256"] is True
        assert record["software_provenance"]["model_execution_helper_sha256"] == helper
        assert record["software_provenance"]["response_validator_sha256"] == checker
        for sample in record["samples"]:
            assert sample["finish_reasons"] == ["stop"]
            assert sample["full_prompt_token_check"] == "MATCHED_SERVER_USAGE"
            assert sample["server_prompt_tokens"] == sample["preflight_prompt_tokens"]


def test_model_artifact_records_exact_publisher_weights_not_just_a_name():
    artifact = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/qwen35-27b-deployable-artifact-20261022.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["publisher_all_shards_verified"] is True
    assert len(artifact["files"]) == 11
    assert all(row["sha256"] == row["publisher_sha256"] for row in artifact["files"])
    assert artifact["publisher_runtime_files_verified"] is True
    assert all(row["publisher_matches"] for row in artifact["runtime_files"])
