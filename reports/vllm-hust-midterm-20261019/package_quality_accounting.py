"""Package token-extent qualification or paired full-answer quality evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=(
            "accounting",
            "accounting-v2",
            "accounting-v3",
            "attention",
            "moe-gdn",
            "recurrence",
            "vocabulary-prefill",
            "gdn-reference",
            "quality",
        ),
        required=True,
    )
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()
    if args.kind == "gdn-reference":
        directories = [
            "gdn-reference-r1",
            "gdn-reference-control-r1",
            "gdn-reference-r2",
            "gdn-reference-control-r2",
        ]
        sources = [
            "prepare_gdn_reference_plan.py",
            "prepare_gdn_reference_plan_full.py",
            "prepare_gdn_reference_controller.py",
            "prepare_gdn_reference_full.py",
            "gdn-reference-controller-build.json",
            "gdn-reference-full-build.json",
            "gdn-reference-plan-r1.json",
            "gdn-reference-plan-r2.json",
            "run_gdn_reference.py",
            "run_gdn_reference_full.py",
            "gdn_reference_probe.py",
            "gdn_reference_probe_full.py",
            "gdn_reference_probe.sh",
            "gdn_reference_probe_full.sh",
            "dcmi_hccs64.py",
            "dcmi_hccs_bandwidth.py",
            "measure_microflow.py",
            "audit_gdn_reference.py",
            "gdn-reference-summary-r2.json",
        ]
        summaries = {"reference.json": "gdn-reference-summary-r2.json"}
    elif args.kind == "vocabulary-prefill":
        directories = ["gdn-native-source-r8"]
        sources = [
            "audit_vocab_projection.py",
            "test_vocab_projection.py",
            "snapshot_gdn_native_sources.py",
            "audit_gdn_prefill_shapes.py",
            "test_gdn_prefill_shapes.py",
            "combine_vocab_accounting.py",
            "audit_gdn_host_schedule.py",
            "audit_token_accounting.py",
            "vocab-projection-summary-r8.json",
            "gdn-prefill-shapes-r8.json",
            "same-run-partial-accounting-r8-v2.json",
            "same-run-partial-accounting-r8.json",
            "host-attention-pairs-r8.json",
            "payload-receipt.json",
            "actual-runtime.json",
            "payload/betterscale/models/qwen35/__init__.py",
            "payload/betterscale/models/qwen35/small_fish_runtime.py",
            "payload/betterscale/models/qwen35/integration.py",
            "runtime/vllm_ascend/spec_decode/llm_base_proposer.py",
            "runtime/vllm_ascend/ops/vocab_parallel_embedding.py",
        ]
        summaries = {
            "vocabulary.json": "vocab-projection-summary-r8.json",
            "prefill-shapes.json": "gdn-prefill-shapes-r8.json",
            "same-run-catalog.json": "same-run-partial-accounting-r8-v2.json",
        }
    elif args.kind == "recurrence":
        directories = ["microflow-token-accounting-r8", "token-observer-v7"]
        sources = [
            "prepare_token_observer_v7.py",
            "measure_token_accounting_v7.py",
            "launch_token_accounting_v7.py",
            "runtime-token-accounting-v7.sh",
            "serve-token-accounting-v7.sh",
            "token-observer-build-v7.json",
            "test_gdn_slots.py",
            "audit_gdn_recurrence.py",
            "test_gdn_recurrence.py",
            "audit_gdn_host_schedule.py",
            "audit_token_accounting.py",
            "audit_host_attention_pairs.py",
            "derive_attention_pairs.py",
            "prepare_same_run_accounting.py",
            "same-run-accounting-build.json",
            "derive_moe_projection_accounting.py",
            "derive_moe_projection_accounting_r8.py",
            "derive_partial_operator_accounting.py",
            "derive_partial_operator_accounting_r8.py",
            "combine_same_run_accounting.py",
            "test_same_run_accounting.py",
            "gdn-recurrence-summary-r8.json",
            "host-attention-pairs-r8.json",
            "partial-operator-accounting-r8.json",
            "moe-projection-summary-r8.json",
            "same-run-partial-accounting-r8.json",
            "payload/betterscale/models/qwen35/decode_kv.py",
            "payload/betterscale/models/qwen35/service_metadata.py",
            "payload/betterscale/models/qwen35/host_metadata.py",
            "payload/betterscale/models/qwen35/mixed_core.py",
        ]
        summaries = {
            "recurrence.json": "gdn-recurrence-summary-r8.json",
            "attention-pairs.json": "host-attention-pairs-r8.json",
            "projections.json": "partial-operator-accounting-r8.json",
            "moe-projections.json": "moe-projection-summary-r8.json",
            "same-run-catalog.json": "same-run-partial-accounting-r8.json",
        }
    elif args.kind == "moe-gdn":
        directories = [
            "microflow-token-accounting-r6",
            "microflow-token-accounting-r7",
            "token-observer-v5",
            "token-observer-v6",
        ]
        sources = [
            "model_shape_inventory.py",
            "gdn_host_snapshot.py",
            "host_logits_snapshot.py",
            "host_mtp_snapshot.py",
            "attention_length_snapshot.py",
            "audit_loaded_moe.py",
            "derive_moe_projection_accounting.py",
            "audit_gdn_host_schedule.py",
            "audit_token_accounting.py",
            "test_model_shape_inventory.py",
            "test_moe_projection_accounting.py",
            "test_gdn_host_snapshot.py",
            "test_gdn_schedule_audit.py",
            "loaded-moe-summary-r6.json",
            "moe-projection-summary-r6.json",
            "gdn-host-summary-r7.json",
        ]
        for version in (5, 6):
            sources.extend(
                [
                    f"prepare_token_observer_v{version}.py",
                    f"runtime-token-accounting-v{version}.sh",
                    f"serve-token-accounting-v{version}.sh",
                    f"measure_token_accounting_v{version}.py",
                    f"launch_token_accounting_v{version}.py",
                    f"token-observer-build-v{version}.json",
                ]
            )
        summaries = {
            "loaded-moe.json": "loaded-moe-summary-r6.json",
            "moe-projections.json": "moe-projection-summary-r6.json",
            "gdn-schedule.json": "gdn-host-summary-r7.json",
        }
    elif args.kind == "attention":
        directories = [
            "microflow-token-accounting-r4",
            "microflow-token-accounting-r5",
            "token-observer-v4",
        ]
        sources = [
            "prepare_token_observer_v4.py",
            "attention_length_snapshot.py",
            "host_logits_snapshot.py",
            "host_mtp_snapshot.py",
            "runtime-token-accounting-v4.sh",
            "serve-token-accounting-v4.sh",
            "measure_token_accounting_v4.py",
            "launch_token_accounting_v4.py",
            "token-observer-build-v4.json",
            "test_attention_length_snapshot.py",
            "test_attention_length_audit.py",
            "audit_attention_lengths.py",
            "audit_attention_length_sources.py",
            "attention-length-source-audit.json",
            "derive_attention_pairs.py",
            "test_attention_pairs.py",
            "derive_partial_operator_accounting.py",
            "test_partial_operator_accounting.py",
            "attention-length-summary-r5.json",
            "attention-pairs-summary-r5.json",
            "partial-operator-accounting-r5.json",
            "audit_token_accounting.py",
            "hbm-profiler-relocation-20261020.sha256",
        ]
        summaries = {
            "length-summary.json": "attention-length-summary-r5.json",
            "pairs-summary.json": "attention-pairs-summary-r5.json",
            "partial-operators.json": "partial-operator-accounting-r5.json",
        }
    elif args.kind == "accounting-v3":
        directories = ["microflow-token-accounting-r3", "token-observer-v3"]
        sources = [
            "prepare_token_observer_v3.py",
            "host_logits_snapshot.py",
            "host_mtp_snapshot.py",
            "runtime-token-accounting-v3.sh",
            "serve-token-accounting-v3.sh",
            "measure_token_accounting_v3.py",
            "launch_token_accounting_v3.py",
            "token-observer-build-v3.json",
            "test_host_logits_snapshot.py",
            "audit_logits_accounting.py",
            "audit_token_accounting.py",
            "logits-accounting-summary-r3.json",
        ]
        summaries = {"logits-summary.json": "logits-accounting-summary-r3.json"}
    elif args.kind == "accounting-v2":
        directories = ["microflow-token-accounting-r2", "token-observer-v2"]
        sources = [
            "prepare_token_observer_v2.py",
            "host_mtp_snapshot.py",
            "runtime-token-accounting-v2.sh",
            "serve-token-accounting-v2.sh",
            "measure_token_accounting_v2.py",
            "launch_token_accounting_v2.py",
            "token-observer-build-v2.json",
            "test_host_mtp_snapshot.py",
            "test_token_observer_v2.py",
            "audit_token_accounting_v2.py",
            "audit_token_accounting.py",
            "token-accounting-step-summary-r2.json",
        ]
        summaries = {"step-summary.json": "token-accounting-step-summary-r2.json"}
    elif args.kind == "accounting":
        directories = ["microflow-token-accounting-r1", "token-observer"]
        sources = [
            "prepare_token_observer.py",
            "runtime-token-accounting.sh",
            "serve-token-accounting.sh",
            "measure_token_accounting.py",
            "launch_token_accounting.py",
            "token-observer-build.json",
            "test_token_observer.py",
            "audit_token_accounting.py",
            "audit_token_accounting_drained.py",
            "token-accounting-summary-r1.json",
            "token-accounting-drained-summary-r1.json",
        ]
        summaries = {
            "request-window-summary.json": "token-accounting-summary-r1.json",
            "drained-summary.json": "token-accounting-drained-summary-r1.json",
        }
    else:
        assert args.run, "Quality packaging requires explicit run directories"
        directories = args.run
        sources = [
            "quality_client.py",
            "quality_client_v2.py",
            "quality-client.sh",
            "quality-client-v2.sh",
            "quality_client_v3.py",
            "quality_client_v3base.py",
            "quality-client-v3.sh",
            "prepare_quality_v3.py",
            "quality-v3-build.json",
            "prepare_quality_controllers.py",
            "prepare_quality_v2.py",
            "quality-controller-build.json",
            "quality-controller-build-v2.json",
            "measure_quality_b0.py",
            "measure_quality_b0_v2.py",
            "measure_quality_microflow.py",
            "measure_quality_microflow_v2.py",
            "measure_quality_kvmat.py",
            "measure_quality_b0_v3.py",
            "measure_quality_microflow_v3.py",
            "measure_quality_kvmat_v3.py",
            "prepare_quality_kvmat.py",
            "quality-kvmat-build.json",
            "recovered-mmlu-cases.json",
            "verified-mmlu-quality-inputs.json",
            "verify_mmlu_quality_inputs.py",
            "recover_quality_wire_inputs.py",
            "quality-wire-inputs.json",
            "quality-wire-source-audit.json",
            "audit_quality_wire_sources.py",
            "test_quality_client.py",
            "audit_quality_pair.py",
            "audit_quality_finishes.py",
            "baseline.py",
            "baseline.sh",
            "runtime-kvmat.sh",
            "launch_kvmat.py",
            "serve-kvmat.sh",
            "runtime-microflow.sh",
            "launch_microflow.py",
            "serve-microflow.sh",
            "microflow-source-receipt.json",
            "microflow-manager.json",
            "microflow-manager-receipt.json",
            "kvmat-source-receipt.json",
            "kvmat-managed-source-receipt.json",
            "materialized.200.jsonl",
            "actual-runtime.json",
        ]
        summaries = {}
    selected = {}
    for directory in directories:
        for path in (args.root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                selected[path.relative_to(args.root).as_posix()] = path
    for source in sources:
        selected[f"source/{source}"] = args.root / source
    args.output.mkdir(exist_ok=False)
    members = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative, path in sorted(selected.items()):
            members[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative, recursive=False)
    payload = buffer.getvalue()
    parts = []
    for index, offset in enumerate(range(0, len(payload), 900_000)):
        part = payload[offset : offset + 900_000]
        name = f"{args.kind}-evidence.tar.gz.part{index:03d}"
        (args.output / name).write_bytes(part)
        parts.append({"file": name, "sha256": hashlib.sha256(part).hexdigest()})
    (args.output / "archive.json").write_text(
        json.dumps(
            {
                "parts": parts,
                "concatenated_sha256": hashlib.sha256(payload).hexdigest(),
                "members": members,
            },
            indent=2,
        )
        + "\n"
    )
    for name, source in summaries.items():
        (args.output / name).write_bytes((args.root / source).read_bytes())


if __name__ == "__main__":
    main()
